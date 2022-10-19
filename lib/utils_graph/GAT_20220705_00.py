import os
import sys
import cv2
import copy
import time
import json
import math
import random
import pickle
import triangle
import datetime

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

import imgaug as ia
import imgaug.augmenters as iaa

import torch
import torch.nn as nn
import torch.utils.data
import torch.nn.parallel
import torch.optim as optim
import torch.nn.functional as F
import torchvision.utils as vutils
import torchvision.models as models
from torch.autograd import Variable, grad
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, utils

from torch_geometric.data import Data
from torch_geometric.nn import MessagePassing
from torch_geometric.nn import GCNConv, GATConv
from torch_geometric.utils import remove_self_loops, add_self_loops
from torch_geometric.utils import softmax as torch_geo_softmax

# from PIL import Image
from random import shuffle
from tqdm import tqdm, trange
from scipy.spatial import distance
from collections import OrderedDict
from shapely.geometry import Polygon
from IPython.display import clear_output
from PIL import Image, ImageDraw, ImageFont

from lib.utils.basic import resize_short, resize_long, rot_c, rot_cc, rad2vec, rasterize
from lib.utils.basic import line_suppress, junc_suppress
from lib.utils.basic import dist2line, is_intersect
from lib.detection_v1.dl_models import count_parameters
from lib.detection_v1.utils_contrib import seed_everything
from lib.augment import Augmenter, aug_config
from lib.augment.utils import room_names
from lib.utils.model_hourglass_hawp_pixshuffle_iROI1 import LineSegDetector
from lib.utils.dataset import MyDataset, collate_batch, move_to_device
from lib.utils_graph.building_funcs import (
    refurbish, tri2seg,
    build_torch_geo_data, build_torch_geo_data_non_dual,
    build_graph_gt, build_graph_pred, build_aligned_graph, build_mapped_connections,
)
from lib.utils.model_hourglass_hawp_pixshuffle_iROI1 import build_hourglass

class CusGATConvHead(MessagePassing):
    
    def __init__(self, in_channels, out_channels, edge_in_channels, edge_out_channels):
        super().__init__(aggr='add')
        self.FC_neighbours = nn.Linear(in_channels, out_channels, bias=True)
        self.FC_phi_i = nn.Linear(out_channels, out_channels, bias=True)
        self.FC_phi_j = nn.Linear(out_channels, out_channels, bias=True)
        self.edge_linear = nn.Linear(out_channels * 2 + edge_in_channels, edge_out_channels, bias=True)
        self.attn = nn.Linear(2 * out_channels + edge_in_channels, out_channels)
        self.negative_slope = 0.2
        self.reset_parameters()

    def reset_parameters(self):
        self.FC_neighbours.reset_parameters()
        self.FC_phi_i.reset_parameters()
        self.FC_phi_j.reset_parameters()

    def forward(self, x, edge_index, edge_attr):
        row, col = edge_index
        x_fc = self.FC_neighbours(x)
        updated_pt_attr = self.propagate(edge_index, x=x_fc, edge_attr=edge_attr)
        updated_edge_attr = self.edge_linear(torch.concat([
            updated_pt_attr[edge_index[0]], updated_pt_attr[edge_index[1]], edge_attr
        ], dim=1))
        return updated_pt_attr, updated_edge_attr

    def message(self, edge_index_i, edge_index_j, x_i, x_j, edge_attr):
        x_phi_i = self.FC_phi_i(x_i)
        x_phi_j = self.FC_phi_j(x_j)
        x = torch.cat([x_phi_i, x_phi_j, edge_attr], dim=-1)
        alpha = self.attn(x)
        alpha = F.leaky_relu(alpha, self.negative_slope)
        alpha = torch_geo_softmax(alpha, edge_index_i)
        return alpha * x_j
    
    def update(self, aggr_out):
        return aggr_out

class CusGATConv(nn.Module):
    
    def __init__(self, in_channels, out_channels, edge_in_channels, edge_out_channels, n_heads, device):
        super().__init__()
        self.heads = nn.ModuleList([CusGATConvHead(
            in_channels, out_channels, edge_in_channels, edge_out_channels
        ).to(device) for _ in range(n_heads)])
        self.FC_pt = nn.Linear(in_channels + out_channels * n_heads, out_channels)
        self.FC_edge = nn.Linear(edge_in_channels + edge_out_channels * n_heads, edge_out_channels)
        self.to(device)

    def forward(self, _input):
        x, edge_index, edge_attr = _input
        head_res = [head(x, edge_index, edge_attr) for head in self.heads]
        pt_attrs = [x] + [r[0] for r in head_res]
        edge_attrs = [edge_attr] + [r[1] for r in head_res]
        
        pt_cat = torch.cat(pt_attrs, dim=1)
        edge_cat = torch.cat(edge_attrs, dim=1)
        
        out_pt = self.FC_pt(pt_cat)
        out_edge = self.FC_edge(edge_cat)
        return out_pt, edge_index, out_edge

class CusGATP_Net(nn.Module):
    
    def __init__(self, config):
        super().__init__()
        for key, value in config.items():
            setattr(self, key, value)
        
        self.feat_extractor = build_hourglass(
            head_size  = self.hour_glass_params["head_size"],
            inplanes   = self.hour_glass_params["inplanes"],
            num_feats  = self.hour_glass_params["num_feats"],
            num_stacks = self.hour_glass_params["num_stacks"],
            depth      = self.hour_glass_params["depth"],
            num_blocks = self.hour_glass_params["num_blocks"],
        )
        self.fc_loi = nn.Conv2d(self.feat_extractor.out_feature_channels, self.dim_loi, 1)
        self.pool3 = nn.MaxPool2d(kernel_size=(1,3))
        self.pool1d = nn.MaxPool1d(2, 2)
        self.fc_hour = nn.Sequential(
            nn.Linear(self.dim_loi * self.n_pts0 // 2, self.dim_hour_fc),
            nn.ReLU(inplace=True),
            nn.Linear(self.dim_hour_fc, self.dim_hour_fc),
            nn.ReLU(inplace=True),
            nn.Linear(self.dim_hour_fc, self.dim_hour_out),
        )
        self.fc_node = nn.Linear(self.in_dim, self.hidden_dim)
        
        self.register_buffer(
            'tspan_0', torch.linspace(0, 1, self.n_pts0)[None,None,:].to(self.device)
        )
        self.register_buffer(
            'tspan_1', (torch.linspace(-1, 1, self.n_pts1) * self.span_1 / 2)[None,None,:].to(self.device)
        )
        self.register_buffer(
            'rot_mat_90', torch.from_numpy(np.array([[0,-1],[1,0]], np.float32)).to(self.device)
        )
        self.register_buffer(
            'rot_mat_counter_90', torch.from_numpy(np.array([[0,1],[-1,0]], np.float32)).to(self.device)
        )
        
        self.fc_edge = nn.Linear(self.edge_in_dim, self.edge_hidden_dim)
        self.conv_in = CusGATConv(
            self.hidden_dim, self.hidden_dim,
            self.edge_hidden_dim + self.dim_hour_out, self.edge_hidden_dim, self.n_heads, self.device
        )
        self.hiddens = nn.Sequential(*[CusGATConv(
            self.hidden_dim, self.hidden_dim,
            self.edge_hidden_dim, self.edge_hidden_dim, self.n_heads, self.device
        ) for _ in range(self.n_hiddens)])
        self.conv_out = CusGATConv(
            self.hidden_dim, self.out_dim,
            self.edge_hidden_dim, self.edge_out_dim, self.n_heads, self.device
        )
        self.final = nn.Sigmoid()
        
        self.to(self.device)
    
    def pooling(self, features_per_image, lines_per_im):
        """
        features_per_image: (C, S0, S1)
        lines_per_im: (N, 4)
        """
        h,w = features_per_image.size(1), features_per_image.size(2)
        U,V = lines_per_im[:,:2], lines_per_im[:,2:]
        line_dir = F.normalize(V-U, p=2.0, dim=1)

        if np.random.random()<0.5:
            line_norm = torch.matmul(line_dir, self.rot_mat_90)
        else:
            line_norm = torch.matmul(line_dir, self.rot_mat_counter_90)

        dir_points = U[:,:,None]*self.tspan_0 + V[:,:,None]*(1-self.tspan_0)
        dir_points = dir_points.permute((0,2,1))

        norm_points = line_norm[:,:,None] * self.tspan_1
        norm_points = norm_points.permute((0,2,1))
        sampled_points = dir_points[:,:,None,:] + norm_points[:,None,:,:]

        sampled_points = sampled_points.reshape([-1,2])

        px, py = sampled_points[:,0],sampled_points[:,1]
        px0 = px.floor().clamp(min=0, max=w-1)
        py0 = py.floor().clamp(min=0, max=h-1)
        px1 = (px0 + 1).clamp(min=0, max=w-1)
        py1 = (py0 + 1).clamp(min=0, max=h-1)
        px0l, py0l, px1l, py1l = px0.long(), py0.long(), px1.long(), py1.long()

        xp = (
            features_per_image[:, py0l, px0l] * (py1-py) * (px1 - px) + \
            features_per_image[:, py1l, px0l] * (py - py0) * (px1 - px) + \
            features_per_image[:, py0l, px1l] * (py1 - py) * (px - px0) + \
            features_per_image[:, py1l, px1l] * (py - py0) * (px - px0)
        ).reshape(self.dim_loi, -1, self.n_pts0, self.n_pts1).permute(1,0,2,3)

        xp = self.pool3(xp).squeeze(3)
        xp = self.pool1d(xp)

        features_per_line = xp.view(xp.shape[0], -1)

        hs_hour = self.fc_hour(features_per_line)
        return hs_hour
    
    def forward(self, img_tensor, data):
        assert img_tensor.shape[0]==1
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        hourglass_outputs, features = self.feat_extractor(img_tensor)
        loi_features = self.fc_loi(features)
        
        pt0s = x[edge_index[0]] * loi_features.shape[2]
        pt1s = x[edge_index[1]] * loi_features.shape[2]
        lines_for_train = torch.concat([pt0s, pt1s], dim=1)
        pooled_feats = self.pooling(loi_features[0], lines_for_train)
        
        h_node = self.fc_node(x)
        h_edge = torch.concat([self.fc_edge(edge_attr), pooled_feats], dim=1)
        hn, _, he = self.conv_in((h_node, edge_index, h_edge))
        hn, _, he = self.hiddens((hn, edge_index, he))
        out_node, _, out_edge = self.conv_out((hn, edge_index, he))
        out_edge = self.final(out_edge)
        out_edge = transpose_avg(data, out_edge) # 确保是对称矩阵
        
        return hourglass_outputs, out_node, out_edge

def transpose_avg(data, out_edge):
    edge_mat = torch.zeros([data.x.shape[0], data.x.shape[0], out_edge.shape[1]]).to(out_edge.device)
    edge_mat[...,0] = 1
    edge_mat[data.edge_index[0], data.edge_index[1]] = out_edge
    edge_mat = (edge_mat + edge_mat.transpose(0, 1)) / 2
    return edge_mat[data.edge_index[0], data.edge_index[1]]

def check_cross(data, out_edge):
    conf = sorted(
        [(i,x) for i,x in enumerate((1-out_edge[:,0]).detach().cpu().numpy())],
        key=lambda z: z[1], reverse=True
    )
    cross_lines = set([])
    valid_lines = []
    for index, _ in conf:
        if torch.max(out_edge[index])==out_edge[index, 0]:
            continue
        L1 = [
            data.x[data.edge_index[0, index]].detach().cpu().numpy(),
            data.x[data.edge_index[1, index]].detach().cpu().numpy(),
        ]
        is_cross = False
        for l in valid_lines:
            point_indices = list(data.edge_index[:, index].detach().cpu().numpy())
            point_indices = point_indices + list(data.edge_index[:, l].detach().cpu().numpy())
            
            if len(set(point_indices))<4:
                continue

            L2 = [
                data.x[data.edge_index[0, l]].detach().cpu().numpy(),
                data.x[data.edge_index[1, l]].detach().cpu().numpy(),
            ]
            inter = is_intersect(L1, L2)
            d1, d2 = 1, 1
            if inter:
                inter = np.array(inter)
                d1 = dist2line(L1[0], L1[1], inter)
                d2 = dist2line(L2[0], L2[1], inter)
            if d1<1e-8 and d2<1e-8:
                is_cross = True
        
        if is_cross:
            cross_lines.add(index)
        else:
            valid_lines.append(index)
    return cross_lines, valid_lines

def post_process():
    # each line 4 argmax
    # 去除朝外的游荡线段
    return
