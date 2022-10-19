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

def preprocess(img, img_size=512):
    if img is None:
        return None
    img, _ = resize_long(img, img_size)
    bias = [(img_size - img.shape[0])//2, 0, (img_size - img.shape[1])//2, 0]
    bias[1], bias[3] = img_size - img.shape[0] - bias[0], img_size - img.shape[1] - bias[2]
    img = cv2.copyMakeBorder(img, bias[0], bias[1], bias[2], bias[3], cv2.BORDER_REPLICATE)
    return img

def build_torch_geo_data(point_list_hs, lines_dict_hs, mapped_connections, device, img_size=None):
    line_segs = {}
    line_segs_pred = {}
    for k in lines_dict_hs:
        line = lines_dict_hs[k]
        line_segs_pred[tuple(sorted(lines_dict_hs[k][:2]))] = lines_dict_hs[k][2]
    gt_segs = [tuple(sorted(mapped_connections[k][:2])) for k in mapped_connections]
    
    segs_hs = [tuple(sorted(lines_dict_hs[k][:2])) for k in lines_dict_hs]
    tri = triangle.triangulate({
        "vertices": point_list_hs,
        "segments": segs_hs
    })
    segments = [tuple(sorted(x)) for x in tri2seg(tri["triangles"])]
    segments_hull = [tuple(sorted(x)) for x in triangle.convex_hull(point_list_hs).tolist()]
    segments = list(set(segments + segments_hull))
    
    line_seg_keys = []
    for i in range(len(point_list_hs)-1):
        for j in range(i+1, len(point_list_hs)):
            line_seg_keys.append((i,j))
    
    for i,j in line_seg_keys:
        width = 0
        if np.random.random()<0.5:
            pt0 = point_list_hs[i]
            pt1 = point_list_hs[j]
        else:
            pt0 = point_list_hs[j]
            pt1 = point_list_hs[i]
        direction = pt1 - pt0
        center = (pt1 + pt0) / 2
        distance = np.linalg.norm(direction)
        direction = abs((direction / distance)[1])
        degree = np.clip(np.arccos(direction) / np.pi * 180, 0, 90)
        info = [1, 0, 0, 0, 0, center[0], center[1], pt0[0], pt0[1], pt1[0], pt1[1], direction, distance, width]
        if img_size is not None:
            for i_info in range(5, 11):
                info[i_info] = info[i_info] / img_size * 2 - 1
            info[12] = distance / img_size
        
        if (i,j) in line_segs_pred:
            info[0] = 0
            info[line_segs_pred[(i,j)][0]] = 1
            info[13] = line_segs_pred[(i,j)][3] # type_index, center, direction, width, distance
            line_segs[(i,j)] = info[:13] # 不要width
        elif (i,j) in segments:
            line_segs[(i,j)] = info[:13] # 不要width
        elif distance<8 or degree<5 or degree>85 or (distance<32 and (degree<10 or degree>80)):
            line_segs[(i,j)] = info[:13] # 不要width
    
    line_seg_keys = sorted(list(line_segs.keys()), key=lambda x: x[0]+x[1]*1e-6)
    
    edge_index = []
    edge_attr = []
    for ik, k in enumerate(line_seg_keys):
        for jk, k_ in enumerate(line_seg_keys):
            if ik==jk:
                continue
            if len(set(list(k)+list(k_)))<4:
                edge_index.append([ik, jk])
                if k[0]==k_[0]:
                    e_attr = point_list_hs[k[0]]
                elif k[0]==k_[1]:
                    e_attr = point_list_hs[k[0]]
                elif k[1]==k_[0]:
                    e_attr = point_list_hs[k[1]]
                elif k[1]==k_[1]:
                    e_attr = point_list_hs[k[1]]
                edge_attr.append(e_attr)
    edge_index = np.array(edge_index, int)
    edge_attr = np.array(edge_attr)
    if img_size is not None:
        edge_attr = edge_attr / img_size * 2 - 1
    edge_attr = edge_attr.astype(np.float32)
    
    point_attr = torch.tensor([line_segs[k] for k in line_seg_keys], dtype=torch.float).to(device)
    edge_index = torch.from_numpy(edge_index).long().T.to(device)
    edge_attr = torch.from_numpy(edge_attr).to(device)
    data = Data(x=point_attr, edge_index=edge_index, edge_attr=edge_attr)
    
    ys_dict = {}
    for k in mapped_connections:
        seg = tuple(mapped_connections[k][:2])
        ys_dict[seg] = mapped_connections[k][2][0]
        ys_dict[seg[::-1]] = mapped_connections[k][2][0]
    
    ys = np.zeros([len(line_seg_keys), 5])
    for i, seg in enumerate(line_seg_keys):
        if seg in ys_dict:
            ys[i,ys_dict[seg]] = 1
        else:
            ys[i,0] = 1
    ys = torch.from_numpy(ys.astype(np.float32)).to(device)
    return line_seg_keys, data, ys

class LineSegDetectorWrapper:
    
    def __init__(self, model_path, device):
        self.device = device
        hour_glass_params = {"inplanes": 128, "num_feats": 256, "num_stacks": 2, "depth": 4, "num_blocks": 1,}
        self.my_model = LineSegDetector(
            n_class=5, device=self.device, d_max=5, span_1=2, hour_glass_params=hour_glass_params,
        )
        self.my_model.load_state_dict(torch.load(model_path, map_location=str(self.device)))
        self.my_model.eval()
    
    def predict(self, xs):
        T_juncs = 0.15
        T_lines = 0.35
        
        with torch.no_grad():
            output_dict = self.my_model(xs, T_juncs=T_juncs, T_lines=T_lines)
        juncs_final = output_dict["juncs_final"].detach().cpu().numpy()
        juncs_score = output_dict["juncs_score"].detach().cpu().numpy()
        lines_final = output_dict["lines_final"].detach().cpu().numpy()
        score_final = output_dict["score_final"].detach().cpu().numpy()
        width_final = output_dict["width_final"].detach().cpu().numpy()

        # suppression
        lines_final, lines_final_indices = line_suppress(lines_final, T_seg_sup_dist=5, T_seg_sup_angle=20)
        score_final = np.array([score_final[i] for i in lines_final_indices])
        width_final = np.array([width_final[i] for i in lines_final_indices])
        
        juncs_final, juncs_score = junc_suppress(juncs_final, juncs_score)
        
        return juncs_final, juncs_score, lines_final, score_final, width_final

class GaANConvHead(MessagePassing):
    
    def __init__(self, in_channels, out_channels, edge_channels):
        super().__init__(aggr='add')
        self.FC_neighbours = nn.Linear(in_channels, out_channels, bias=True)
        self.FC_phi_i = nn.Linear(out_channels, out_channels, bias=True)
        self.FC_phi_j = nn.Linear(out_channels, out_channels, bias=True)
        self.attn = nn.Linear(2 * out_channels+edge_channels,out_channels)
        self.negative_slope = 0.2
        self.reset_parameters()

    def reset_parameters(self):
        self.FC_neighbours.reset_parameters()
        self.FC_phi_i.reset_parameters()
        self.FC_phi_j.reset_parameters()

    def forward(self, x, edge_index, edge_attr=None):
        row, col = edge_index
        x_fc = self.FC_neighbours(x)
        out = self.propagate(edge_index, x=x_fc, edge_attr=edge_attr)
        return out

    def message(self, edge_index_i, edge_index_j, x_i, x_j, edge_attr):
        x_phi_i = self.FC_phi_i(x_i)
        x_phi_j = self.FC_phi_j(x_j)
        if edge_attr is None:
            x = torch.cat([x_phi_i, x_phi_j], dim=-1)
        else:
            x = torch.cat([x_phi_i, x_phi_j, edge_attr], dim=-1)
        alpha = self.attn(x)
        alpha = F.leaky_relu(alpha, self.negative_slope)
        alpha = torch_geo_softmax(alpha, edge_index_i)
        return alpha * x_j
    
    def update(self, aggr_out):
        return aggr_out

class GaANConv(nn.Module):
    
    def __init__(self, in_channels, out_channels, edge_channels, n_heads, device):
        super().__init__()
        self.heads = nn.ModuleList([
            GaANConvHead(in_channels, out_channels, edge_channels).to(device) for _ in range(n_heads)
        ])
        self.FC = nn.Linear(in_channels + out_channels * n_heads, out_channels, bias=True)
        self.to(device)

    def forward(self, x, edge_index, edge_attr=None):
        head_res = [x] + [head(x, edge_index, edge_attr) for head in self.heads]
        out = torch.cat(head_res, dim=1)
        out = self.FC(out)
        return out

class GaANNet_pooling(nn.Module):
    
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
        self.fc_node = nn.Linear(self.in_dim, self.dim_hour_out)
        
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
        
        self.FC_edge = nn.Linear(self.edge_in_dim, self.edge_hidden_dim)
        self.conv_in = GaANConv(
            self.dim_hour_out * 2, self.hidden_dim, self.edge_hidden_dim, self.n_heads, self.device
        )
        self.hiddens = nn.ModuleList([GaANConv(
            self.hidden_dim, self.hidden_dim, self.edge_hidden_dim, self.n_heads, self.device
        ) for _ in range(self.n_hiddens)])
        self.conv_out = GaANConv(self.hidden_dim, self.out_dim, self.edge_hidden_dim, self.n_heads, self.device)
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
    
    def forward(self, img_tensor, x, edge_index, edge_attr=None):
        assert img_tensor.shape[0]==1
        hourglass_outputs, features = self.feat_extractor(img_tensor)
        loi_features = self.fc_loi(features)
        lines_for_train = []
        for d in x:
            lines_for_train.append((d[7:11] + 1) / 2 * loi_features.shape[2])
        lines_for_train = torch.stack(lines_for_train)
        pooled_feats = self.pooling(loi_features[0], lines_for_train)
        
        h_node = torch.concat([self.fc_node(x), pooled_feats], dim=1)
        
        h_edge = self.FC_edge(edge_attr) if edge_attr is not None else None
        h = self.conv_in(h_node, edge_index, h_edge)
        for conv_hidden in self.hiddens:
            h = conv_hidden(h, edge_index, h_edge)
        out = self.conv_out(h, edge_index, h_edge)
        out = self.final(out)
        return hourglass_outputs, out

def check_cross(line_seg_keys, point_list_hs, hs, max_lines=300):
    hs_np = hs.detach().cpu().numpy()
    cross_lines = set([])
    
    types = np.argmax(hs_np, axis=1)
    is_lines = set([i for i,x in enumerate(types) if x!=0])
    if len(is_lines)>max_lines:
        return False, cross_lines
    confs = sorted(
        [(i,x) for i,x in enumerate((1-hs_np[:,0])) if i in is_lines],
        key=lambda z: z[1], reverse=True
    )
    
    valid_lines = []
    for index1, _ in confs:
        L1 = [list(point_list_hs[line_seg_keys[index1][0]]), list(point_list_hs[line_seg_keys[index1][1]])]
        is_cross = False
        for index2 in valid_lines:
            point_indices = list(line_seg_keys[index1]) + list(line_seg_keys[index2])
            
            if len(set(point_indices))<4:
                continue

            L2 = [list(point_list_hs[line_seg_keys[index2][0]]), list(point_list_hs[line_seg_keys[index2][0]])]
            inter = is_intersect(L1, L2)
            
            d1, d2 = 1, 1
            if inter:
                inter = np.array(inter)
                d1 = dist2line(L1[0], L1[1], inter)
                d2 = dist2line(L2[0], L2[1], inter)
            if d1<1e-8 and d2<1e-8:
                is_cross = True
        if is_cross:
            cross_lines.add(index1)
        else:
            valid_lines.append(index1)
    
    return True, cross_lines

def findNewCycles(path, graph, cycles):
    start_node = path[0]
    next_node= None
    sub = []

    #visit each edge and each node of each edge
    for edge in graph:
        node1, node2 = edge
        if start_node in edge:
            if node1 == start_node:
                next_node = node2
            else:
                next_node = node1
            if not visited(next_node, path):
                    # neighbor node not on path yet
                    sub = [next_node]
                    sub.extend(path)
                    # explore extended path
                    findNewCycles(sub, graph, cycles);
            elif len(path) > 2  and next_node == path[-1]:
                    # cycle found
                    p = rotate_to_smallest(path);
                    inv = invert(p)
                    if isNew(p, cycles) and isNew(inv, cycles):
                        cycles.append(p)

def invert(path):
    return rotate_to_smallest(path[::-1])

#  rotate cycle path such that it begins with the smallest node
def rotate_to_smallest(path):
    n = path.index(min(path))
    return path[n:]+path[:n]

def isNew(path, cycles):
    return not path in cycles

def visited(node, path):
    return node in path

def check_ring(line_seg_keys, links_np):
    # DFS
    graph, cycles = [], []
    
    types = np.argmax(links_np, axis=1)
    graph = [l for l,t in zip(line_seg_keys, types) if t!=0]
    
    for edge in graph:
        for node in edge:
            findNewCycles([node], graph, cycles)
    
    rings = []
    for cycle in cycles:
        link_indices = []
        for i in range(len(cycle)):
            link = tuple(sorted([cycle[i-1], cycle[i]]))
            link_index = line_seg_keys.index(link)
            link_indices.append(link_index)
        rings.append(link_indices)
    return rings

def post_process():
    # each line 4 argmax
    # 去除朝外的游荡线段
    return