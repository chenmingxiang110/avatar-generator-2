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

# from PIL import Image
from random import shuffle
from tqdm import tqdm, trange
from scipy.spatial import distance
from collections import OrderedDict
from shapely.geometry import Polygon

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
from torch_geometric.utils import add_self_loops, degree

from lib.utils.basic import resize_short, resize_long, pt_dist

def build_graph_gt(lines, scale, bias):
    type_dict = {"wall": 1, "door": 2, "window": 3, "other": 4}
    point_list = [(int(np.round(l["start"][0])), int(np.round(l["start"][1]))) for l in lines]
    point_list = point_list + [(int(np.round(l["end"][0])), int(np.round(l["end"][1]))) for l in lines]
    point_list = np.array(list(set(point_list)))
    lines_dict = {}
    min_width = min([l['width'] for l in lines if l["width"] is not None])
    for i_l, l in enumerate(lines):
        type_index = type_dict[l['type']]
        width = l['width'] if l["width"] is not None else min_width
        direction = np.array(l["end"]) - np.array(l["start"])
        center = (np.array(l["end"]) + np.array(l["start"])) / 2
        distance = np.linalg.norm(direction)
        if distance==0:
            continue
        direction = abs((direction / distance)[1])
        center = center * scale + bias
        width = width * scale
        distance = distance * scale
        info = type_index, center, direction, width, distance
        pt0 = np.array([[int(np.round(l["start"][0])), int(np.round(l["start"][1]))]])
        pt1 = np.array([[int(np.round(l["end"][0])), int(np.round(l["end"][1]))]])
        dists = np.sum((pt0 - point_list) ** 2, axis=1), np.sum((pt1 - point_list) ** 2, axis=1)
        
        tmp0, tmp1 = np.argmin(dists[0]), np.argmin(dists[1])
        if tmp0==tmp1: continue
        lines_dict[i_l] = tmp0, tmp1, info
    point_list = point_list * scale + np.expand_dims(bias, 0)
    return point_list, lines_dict

def build_graph_pred(juncs_final, lines_final, score_final, width_final):
    point_list = np.copy(juncs_final)
    lines_dict = {}
    for i_line, line in enumerate(lines_final):
        pt_indices = (
            np.argmin(np.sum(np.abs(np.array([line[:2]]) - juncs_final), axis=1)),
            np.argmin(np.sum(np.abs(np.array([line[2:]]) - juncs_final), axis=1))
        )
        if pt_indices[0]==pt_indices[1]: continue
        type_index = np.argmax(score_final[i_line,1:])+1
        center = (np.array(line[:2]) + np.array(line[2:])) / 2
        direction = np.array(line[2:]) - np.array(line[:2])
        distance = np.linalg.norm(direction)
        direction = abs((direction / distance)[1])
        width = width_final[i_line]
        info = type_index, center.astype(float), direction, width, distance
        lines_dict[i_line] = pt_indices[0], pt_indices[1], info
    return point_list, lines_dict

def build_aligned_graph(
    juncs_final, juncs_score, lines_dict=None, score_n_width=None, T_align=5
):
    assert lines_dict is not None or score_n_width is not None
    if lines_dict is None:
        lines_final, score_final, width_final = score_n_width
        point_list, lines_dict = build_graph_pred(juncs_final, lines_final, score_final, width_final)
    else:
        point_list = np.copy(juncs_final)
    
    point_changed = True
    fix_rank_v = {i: 0 for i in range(len(point_list))}
    fix_rank_h = {i: 0 for i in range(len(point_list))}
    for i_iter in range(100):
        point_changed = False
        for key in lines_dict:
            line = list(point_list[lines_dict[key][0]]) + list(point_list[lines_dict[key][1]])
            abs_cos = point_list[lines_dict[key][1]] - point_list[lines_dict[key][0]]
            abs_cos = abs((abs_cos / np.linalg.norm(abs_cos))[1])
            direction = 0
            if abs_cos<0.5:
                direction = 1 # 竖直方向
            elif abs_cos>0.866:
                direction = 2 # 水平方向

            if direction==1 and line[1]!=line[3] and abs(line[1]-line[3])<T_align:
                point_changed = True
                if fix_rank_v[lines_dict[key][0]] > fix_rank_v[lines_dict[key][1]]:
                    point_list[lines_dict[key][1]][1] = point_list[lines_dict[key][0]][1]
                    fix_rank_v[lines_dict[key][1]] += 1
                elif fix_rank_v[lines_dict[key][0]] < fix_rank_v[lines_dict[key][1]]:
                    point_list[lines_dict[key][0]][1] = point_list[lines_dict[key][1]][1]
                    fix_rank_v[lines_dict[key][0]] += 1
                else:
                    if juncs_score[lines_dict[key][0]]>juncs_score[lines_dict[key][1]]:
                        point_list[lines_dict[key][1]][1] = point_list[lines_dict[key][0]][1]
                        fix_rank_v[lines_dict[key][1]] += 1
                    else:
                        point_list[lines_dict[key][0]][1] = point_list[lines_dict[key][1]][1]
                        fix_rank_v[lines_dict[key][0]] += 1
            elif direction==2 and line[0]!=line[2] and abs(line[0]-line[2])<T_align:
                point_changed = True
                if fix_rank_h[lines_dict[key][0]] > fix_rank_h[lines_dict[key][1]]:
                    point_list[lines_dict[key][1]][0] = point_list[lines_dict[key][0]][0]
                    fix_rank_h[lines_dict[key][1]] += 1
                elif fix_rank_h[lines_dict[key][0]] < fix_rank_h[lines_dict[key][1]]:
                    point_list[lines_dict[key][0]][0] = point_list[lines_dict[key][1]][0]
                    fix_rank_h[lines_dict[key][0]] += 1
                else:
                    if juncs_score[lines_dict[key][0]]>juncs_score[lines_dict[key][1]]:
                        point_list[lines_dict[key][1]][0] = point_list[lines_dict[key][0]][0]
                        fix_rank_h[lines_dict[key][1]] += 1
                    else:
                        point_list[lines_dict[key][0]][0] = point_list[lines_dict[key][1]][0]
                        fix_rank_h[lines_dict[key][0]] += 1
        if not point_changed:
            break
    
    return point_list, lines_dict

def build_mapped_connections(point_list_gt, lines_dict_gt, point_list_hs, lines_dict_hs, mapping_dist_T=15):
    
    def build_pt_mapping_gt_hs(point_list_gt, point_list_hs):
        mapping = {}
        for i_pt, pt in enumerate(point_list_gt):
            dists = np.sqrt(np.sum((np.array([pt]) - point_list_hs) ** 2, axis=1))
            index = np.argmin(dists)
            min_dist = dists[index]
            mapping[i_pt] = index if min_dist < mapping_dist_T else None
        return mapping
    
    def group_unmatch(connections):
        cs_ = copy.deepcopy(connections)
        cs_ = [x for x in cs_ if np.linalg.norm(point_list_gt[x[0]] - point_list_gt[x[1]])>1e-3]
        if len(cs_)<=1:
            return cs_
        
        for _ in range(100):
            changed = False
            for i in range(len(cs_)-1):
                if changed:
                    break
                line_i = cs_[i]
                pt0_i, pt1_i, type_index_i, width_i, dist_i = line_i
                vec_i = point_list_gt[pt1_i] - point_list_gt[pt0_i]
                vec_i = vec_i / np.linalg.norm(vec_i)
                for j in range(i+1, len(cs_)):
                    line_j = cs_[j]
                    pt0_j, pt1_j, type_index_j, width_j, dist_j = line_j
                    if pt0_i==pt0_j or pt1_i==pt0_j: # 有共点
                        vec_j = point_list_gt[pt1_j] - point_list_gt[pt0_j]
                        vec_j = vec_j / np.linalg.norm(vec_j)
                        cosine = np.abs(vec_i.dot(vec_j))
                        if cosine>0.95:
                            new_type_index = type_index_i if dist_i>dist_j else type_index_j
                            new_width = width_i if dist_i>dist_j else width_j
                            if pt0_i==pt0_j:
                                new_dist = np.linalg.norm(point_list_gt[pt1_i] - point_list_gt[pt1_j])
                                cs_[i] = [pt1_i, pt1_j, new_type_index, new_width, new_dist]
                            else:
                                new_dist = np.linalg.norm(point_list_gt[pt0_i] - point_list_gt[pt1_j])
                                cs_[i] = [pt0_i, pt1_j, new_type_index, new_width, new_dist]
                            cs_.pop(j)
                            changed = True
                            break
            if not changed:
                break
        return cs_

    pt_mapping_gt_hs = build_pt_mapping_gt_hs(point_list_gt, point_list_hs)
    mapped_pt_hs = list(set([pt_mapping_gt_hs[k] for k in pt_mapping_gt_hs]))
    mapped_connections = {}
    unmatched_connections = []
    for key in lines_dict_gt:
        l = lines_dict_gt[key]
        type_index, width = l[2][0], l[2][3]
        pt0, pt1 = pt_mapping_gt_hs[l[0]], pt_mapping_gt_hs[l[1]]
        if pt0 is not None and pt0==pt1:
            continue
        if pt0 is None or pt1 is None:
            unmatched_connections.append(sorted([l[0], l[1]]) + [l[2][0], l[2][3], l[2][4]])
            continue
        coords = point_list_hs[pt0], point_list_hs[pt1]
        center = (np.array(coords[0]) + np.array(coords[1])) / 2
        direction = np.array(coords[1]) - np.array(coords[0])
        distance = np.linalg.norm(direction)
        direction = abs((direction / distance)[1])
        info = type_index, center.astype(float), direction, width, distance
        mapped_connections[len(mapped_connections)] = pt0, pt1, info
    
    if len(unmatched_connections)>1: # 多于一个才有group的意义
        unmatched_connections = sorted(unmatched_connections, key=lambda x: x[0] + x[1] * 1e-6)
        grouped_unmatch = group_unmatch(unmatched_connections)
        for l in grouped_unmatch:
            type_index, width = l[2], l[3]
            pt0, pt1 = pt_mapping_gt_hs[l[0]], pt_mapping_gt_hs[l[1]]
            if pt0 is not None and pt0==pt1:
                continue
            if pt0 is None or pt1 is None:
                continue
            coords = point_list_hs[pt0], point_list_hs[pt1]
            center = (np.array(coords[0]) + np.array(coords[1])) / 2
            direction = np.array(coords[1]) - np.array(coords[0])
            distance = np.linalg.norm(direction)
            direction = abs((direction / distance)[1])
            info = type_index, center.astype(float), direction, width, distance
            mapped_connections[len(mapped_connections)] = pt0, pt1, info
    
    return mapped_pt_hs, mapped_connections

def tri2seg(tris):
    edges = []
    for tri in tris:
        for index0, index1 in [[tri[0], tri[1]], [tri[1], tri[2]], [tri[0], tri[2]]]:
            if (index0, index1) not in edges and (index1, index0) not in edges:
                edges.append((index0, index1))
    return edges

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
        pt0 = point_list_hs[i]
        pt1 = point_list_hs[j]
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
    return data, ys

def build_torch_geo_data_non_dual(point_list_hs, lines_dict_hs, mapped_connections, device, img_size):
    lines_dict_hs_ij = {}
    for k in lines_dict_hs:
        i,j,info = lines_dict_hs[k]
        lines_dict_hs_ij[(i,j)] = info
        lines_dict_hs_ij[(j,i)] = info
    mapped_connections_ij = {}
    for k in mapped_connections:
        i,j,info = mapped_connections[k]
        mapped_connections_ij[(i,j)] = info
        mapped_connections_ij[(j,i)] = info
    
    edge_index = []
    edge_attr = []
    ys = []
    for i in range(len(point_list_hs)-1):
        for j in range(i+1, len(point_list_hs)):
            # type_index, center.astype(float), direction, width, distance
            if (i,j) in lines_dict_hs_ij:
                info = lines_dict_hs_ij[(i,j)]
                ea_type, ea_width = info[0], info[3]
            else:
                ea_type, ea_width = 0, 0
            if (i,j) in mapped_connections_ij:
                info = mapped_connections_ij[(i,j)]
                y_type, y_width = info[0], info[3]
            else:
                y_type, y_width = 0, 0

            pt0 = point_list_hs[i]
            pt1 = point_list_hs[j]
            center = (pt0 + pt1) / 2
            direction = pt1 - pt0
            distance = np.linalg.norm(direction)
            direction = abs((direction / distance)[1])

            ea = [0, 0, 0, 0, 0, center[0], center[1], pt0[0], pt0[1], pt1[0], pt1[1], direction, distance]
            for i_info in range(5, 11):
                ea[i_info] = ea[i_info] / img_size * 2 - 1
            ea[12] = ea[12] / img_size
            
            y = [0, 0, 0, 0, 0]
            ea[ea_type] = 1
            y[y_type] = 1
            edge_index.append([i,j])
            edge_index.append([j,i])
            edge_attr.append(ea)
            edge_attr.append(ea)
            ys.append(y)
            ys.append(y)
    
    x = torch.from_numpy(np.array(point_list_hs / img_size).astype(np.float32)).to(device)
    edge_index = torch.from_numpy(np.array(edge_index)).long().T.to(device)
    edge_attr = torch.from_numpy(np.array(edge_attr, np.float32)).to(device)
    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
    ys = torch.from_numpy(np.array(ys, np.float32)).to(device)
    
    return data, ys

def refurbish(data, connections, img_size):
    nodes, edge_indices, edge_attrs, lines = [
        data.x.detach().cpu().numpy(),
        data.edge_index.detach().cpu().numpy().T,
        data.edge_attr.detach().cpu().numpy(),
        connections.detach().cpu().numpy(),
    ]
    
    refurbished_lines = []
    for edge_index, edge_attr, line in zip(edge_indices, edge_attrs, lines):
        refurbished_lines.append([
            (edge_attr[7:11] + 1) / 2 * img_size, np.argmax(line)
        ])
    return refurbished_lines
