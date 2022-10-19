import subprocess
import sys
import os
import cv2
import time
import json
import copy
import random
import pickle

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
import torch.nn.functional as F
import torchvision.utils as vutils
import torchvision.models as models

from tqdm import tqdm, trange
from skimage.draw import polygon

from lib.utils.basic import rasterize, softmax, build_one_hot

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

def fp_rasterize(batch_lines, shape, linewidth=10):
    
    def re_parse_lines(lines):
        res = [[[line[0], line[1]], [line[2], line[3]]] for line in lines]
        return res

    canvas = np.zeros([shape[0], shape[1], 3])
    wall_lines = re_parse_lines([x for x in batch_lines if x[-1]=="wall"  ])
    door_lines = re_parse_lines([x for x in batch_lines if x[-1]=="door"  ])
    win_lines  = re_parse_lines([x for x in batch_lines if x[-1]=="window"])
    canvas[...,0] = rasterize(wall_lines, shape[:2], linewidth=linewidth)
    canvas[...,1] = rasterize(door_lines, shape[:2], linewidth=linewidth)
    canvas[...,2] = rasterize(win_lines , shape[:2], linewidth=linewidth)
    return canvas

def create_polygon_from_vertices(shape, vertices):
    polygon_array = np.zeros(shape, 'uint8')
    rr, cc = polygon(vertices[:,0], vertices[:,1], polygon_array.shape)
    polygon_array[rr,cc] = 1
    return polygon_array

def point_suppression(points, T):
    def distance2(p0, p1):
        return (p0[0]-p1[0])**2+(p0[1]-p1[1])**2
    T2 = T**2
    points = sorted(points, key=lambda x: x[-1], reverse=True)
    points_ = []
    for p in points:
        flag = True
        for p_ in points_:
            if distance2(p, p_)<T2:
                flag=False
                break
        if flag:
            points_.append(p)
    return points_

def point_perpendicularize(points, T_pix, re_sort=False):
    if len(points)==0:
        return []
    points_s = sorted(copy.deepcopy(points), key=lambda x: x[-1], reverse=True) if re_sort else copy.deepcopy(points)
    res = [points_s[0]]
    for pt in points_s[1:]:
        lock_0 = False
        lock_1 = False
        for anchor in res:
            if not lock_0 and abs(pt[0]-anchor[0])<=T_pix:
                lock_0 = True
                pt[0] = anchor[0]
            if not lock_1 and abs(pt[1]-anchor[1])<=T_pix:
                lock_1 = True
                pt[1] = anchor[1]
        res.append(pt)
    return res

def get_label(pt0, pt1, batch_lines, eta):
    # 0: none, 1: wall, 2: door, 3: window
    label_dict = {"wall":1, "door":2, "window":3}
    best_dist = eta
    lbl = 0
    for line in batch_lines:
        l0 = np.array(line[0:2])
        l1 = np.array(line[2:4])
        d = min(
            np.linalg.norm(l0 - pt0) + np.linalg.norm(l1 - pt1),
            np.linalg.norm(l1 - pt0) + np.linalg.norm(l0 - pt1),
        )
        if d<=best_dist:
            best_dist = d
            lbl = label_dict[line[4]]
    return lbl

def parse_kps(kp_map, T_conf, T_dist):
    points = []
    for i in range(kp_map.shape[0]):
        for j in range(kp_map.shape[1]):
            if kp_map[i,j]>=T_conf:
                points.append([i,j,kp_map[i,j]])
    points = point_suppression(points, T_dist)
    return points

def parse_lns(points, ln_map, T_degree, T_prob, T_miniwall, mask_linewidth, is_softmax=True):
    line_matrix = np.zeros([len(points), len(points)], int)
    line_confidence = np.zeros([len(points), len(points)])
    if is_softmax:
        ln_prob = softmax(ln_map)
    else:
        ln_prob = build_one_hot(np.argmax(ln_map, axis=0), 4).transpose([2,0,1])
    T_cos = np.cos(T_degree / 180 * np.pi)
    for i in range(len(points)):
        directions = []
        indices = sorted([(
            j, np.linalg.norm(np.array([points[j][0]-points[i][0], points[j][1]-points[i][1]]))
        ) for j in range(len(points))], key=lambda x: x[1])
        # 由近及远
        for j, dist in indices:
            if i==j: continue
            di = np.array([points[j][0]-points[i][0], points[j][1]-points[i][1]])
            di = di / dist
            should_skip = False
            for di_ in directions:
                if np.sum(di * di_) > T_cos:
                    should_skip = True
                    break
            if should_skip:
                directions.append(di)
            else:
                directions.append(di) # 无论是否是线 都要加到已探知的方向中
                if i<j:
                    should_accept=False
                    if dist<T_miniwall and (points[j][0]==points[i][0] or points[j][1]==points[i][1]):
                        should_accept=True
                    _ln = rasterize([[
                        [points[i][0],points[i][1]],[points[j][0],points[j][1]]
                    ]], ln_map.shape[1:], linewidth=mask_linewidth)
                    probs = np.apply_over_axes(np.sum, ln_prob * _ln, [1,2]) / max(np.sum(_ln), 1e-8)
                    if should_accept or probs[0]<1-T_prob:
                        label = np.argmax(probs[1:])+1
                        line_matrix[i,j] = label
                        line_matrix[j,i] = label
                        line_confidence[i,j] = 1-probs[0]
                        line_confidence[j,i] = 1-probs[0]
    return line_matrix, line_confidence

def line_suppression(batch_lines, T_dist, T_degree):
    
    def get_distance2(line0, line1):
        d0 = np.sqrt((line0[0]-line1[0])**2+(line0[1]-line1[1])**2)
        d1 = np.sqrt((line0[2]-line1[2])**2+(line0[3]-line1[3])**2)
        return d0, d1
    
    def get_cos(line0, line1):
        if (line0[0]==line1[0] and line0[1]==line1[1]) or (line0[2]==line1[2] and line0[3]==line1[3]):
            norm0 = np.array([line0[2]-line0[0], line0[3]-line0[1]])
            norm0 = norm0/max(np.linalg.norm(norm0), 1e-8)
            norm1 = np.array([line1[2]-line1[0], line1[3]-line1[1]])
            norm1 = norm1/max(np.linalg.norm(norm1), 1e-8)
            return np.sum(norm0 * norm1)
        elif (line0[0]==line1[2] and line0[1]==line1[3]) or (line0[2]==line1[0] and line0[3]==line1[1]):
            norm0 = np.array([line0[2]-line0[0], line0[3]-line0[1]])
            norm0 = -norm0/max(np.linalg.norm(norm0), 1e-8)
            norm1 = np.array([line1[2]-line1[0], line1[3]-line1[1]])
            norm1 = norm1/max(np.linalg.norm(norm1), 1e-8)
            return np.sum(norm0 * norm1)
        return -float("inf")
    
    if len(batch_lines)==0:
        return []
    
    T_cos = np.cos(T_degree / 180 * np.pi)
    cos_80 = np.cos(80 / 180 * np.pi)
    batch_lines = sorted(batch_lines, key=lambda x: x[-1], reverse=True)
    res = [batch_lines[0]]
    for line in batch_lines[1:]:
        flag = True
        for line_ in res:
            d0, d1 = get_distance2(line, line_)
            if d0+d1<T_dist and get_cos(line, line_)>cos_80: # 避免suppress直角
                flag=False
                break
            if get_cos(line, line_)>T_cos:
                flag=False
                break
        if flag:
            res.append(line)
    return res

def build_batch_lines(points, line_matrix, line_confidence, T_dist, T_degree):
    batch_lines = []
    for i in range(len(points)-1):
        for j in range(i+1, len(points)):
            if line_matrix[i,j]!=0:
                conf = line_confidence[i,j]
                if line_matrix[i,j]==1:
                    label = "wall"
                elif line_matrix[i,j]==2:
                    label = "door"
                else:
                    label = "window"
                batch_lines.append([points[i][0], points[i][1], points[j][0], points[j][1], label, conf])
    batch_lines = line_suppression(batch_lines, T_dist, T_degree)
    return [x[:-1] for x in batch_lines]

def get_batch(batch_size, paths, shape, fix_param=False):
    xs = []
    ys = []
    for index in np.random.choice(len(paths), batch_size):
        path_json = paths[index]+".json"
        path_png = paths[index]+".png"
        path_pkl = paths[index]+".pkl"

        img = cv2.imread(path_png)
        img_ = cv2.resize(img_binarize(img, fix_param=fix_param), (shape[1], shape[0]))
        scale = [shape[0]/img.shape[0], shape[1]/img.shape[1]]
        with open(path_json, 'r') as f:
            info = json.load(f)
        with open(path_pkl, 'rb') as f:
            batch_lines_ = pickle.load(f)
        batch_lines_ = [[
            x[0] * scale[0], x[1] * scale[1],
            x[2] * scale[0], x[3] * scale[1], x[4],
        ] for x in batch_lines_]
        xs.append(img_)
        ys.append(batch_lines_)
    return np.array(xs), ys

def batch_process(paths, batch_size, shape, linewidth, fix_param=False):
    xs, ys = get_batch(batch_size, paths, shape, fix_param=fix_param)
    xs = np.expand_dims(xs, 1)
    ys_ln = np.concatenate([
        np.zeros([batch_size, 1, shape[0], shape[1]]),
        np.array([fp_rasterize(y, shape, linewidth=linewidth) for y in ys]).transpose([0,3,1,2])
    ], axis=1)
    ys_ln = np.argmax(ys_ln, axis=1)
    ys_kp = np.zeros([batch_size, shape[0], shape[1]])
    for i_batch in range(batch_size):
        for ln in ys[i_batch]:
            ys_kp[i_batch, int(np.round(ln[0])), int(np.round(ln[1]))] = 1
            ys_kp[i_batch, int(np.round(ln[2])), int(np.round(ln[3]))] = 1
        # ys_kp[i_batch] = cv2.dilate(ys_kp[0], np.ones((3,3)),iterations = 1)
    return xs, ys, ys_ln, ys_kp
