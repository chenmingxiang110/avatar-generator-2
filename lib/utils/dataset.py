"""
户型图分析
https://github.com/StanislasChaillou/OpenPlan

分割网络
https://github.com/Lextal/pspnet-pytorch
https://github.com/VainF/DeepLabV3Plus-Pytorch
"""

import os
import sys
import cv2
import copy
import time
import json
import random
import pickle
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
import torchvision.transforms as transforms

# shapely
import shapely.vectorized
import shapely
from shapely.geometry import Point
from shapely.geometry import Polygon

# from PIL import Image
from random import shuffle
from tqdm import tqdm, trange
from scipy.spatial import distance
from collections import OrderedDict
from IPython.display import clear_output
from PIL import Image, ImageDraw, ImageFont

from lib.utils.basic import resize_short, resize_long, rot_c, rot_cc, rad2vec, rasterize, rot_vec, proj_dist
from lib.utils.data_process import fp_rasterize, create_polygon_from_vertices
from lib.utils.plot import plot_lines, plot_bboxes, plot_bboxes_and_masks
from lib.augment import Augmenter, aug_config
from lib.detection_v1.dl_models import count_parameters
from lib.augment.utils import _build_from_labels, room_names, _addRandomText

class MyDataset(torch.utils.data.Dataset):

    def __init__(self, paths, room_names, mode, junc_scale_factor=4, afm_scale_factor=4, d_max=5, img_size=512, augmenter=None, img_augseq=None, is_CoarseDropout=False, font_zhs=None, char_sample=None, img_preprocess=None, augment_prob=0):
        if img_augseq is None:
            seq = [
                iaa.SomeOf((0, 4), [
                    # convert images into their superpixel representation
                    # sometimes(iaa.Superpixels(p_replace=(0, 1.0), n_segments=(20, 200))),
                    iaa.OneOf([
                        iaa.GaussianBlur((0, 2.0)), # blur images with a sigma between 0 and 3.0
                        iaa.AverageBlur(k=(2, 3)), # blur image using local means
                    ]),
                    iaa.Sharpen(alpha=(0, 1.0), lightness=(0.75, 1.5)), # sharpen images
                    # add gaussian noise to images
                    iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.05*255), per_channel=0.5),
                    # change brightness of images (by -10 to 10 of original value)
                    iaa.Add((-10, 10), per_channel=0.5),
                    iaa.AddToHueAndSaturation((-20, 20)), # change hue and saturation
                    iaa.LinearContrast((0.8, 1.2), per_channel=0.5), # improve or worsen the contrast
                ], random_order=True),
                iaa.SomeOf((0, 2), [
                    # search either for all edges or for directed edges,
                    # blend the result with the original image using a blobby mask
                    iaa.OneOf([
                        iaa.Sometimes(0.25, iaa.Invert(1.0)), # invert color channels
                        iaa.BlendAlphaSimplexNoise(
                            foreground=iaa.OneOf([
                                iaa.EdgeDetect(alpha=(0.0, 0.5)),
                                iaa.DirectedEdgeDetect(alpha=(0.0, 0.5), direction=(0.0, 1.0)),
                            ]),
                            upscale_method=["linear", "cubic"],
                            size_px_max=(2,16),
                        )
                    ]),
                    # either change the brightness of the whole image (sometimes
                    # per channel) or change the brightness of subareas
                    iaa.OneOf([
                        iaa.Multiply((0.8, 1.2), per_channel=0.5),
                        iaa.BlendAlphaFrequencyNoise(
                            exponent=(-4, 0),
                            foreground=iaa.Multiply((0.8, 1.2), per_channel=True)
                        )
                    ]),
                ], random_order=True),
            ]
            if is_CoarseDropout:
                seq.append(iaa.Sometimes(0.35, iaa.OneOf([
                    iaa.CoarseSaltAndPepper(0.06, size_percent=(0.015, 0.045), per_channel=0.5),
                    iaa.CoarseDropout(0.06, size_percent=(0.015, 0.045), per_channel=0.5),
                ])))
            self.img_augseq = iaa.Sequential(seq, random_order=True)
        else:
            self.img_augseq = copy.deepcopy(img_augseq)
        
        self.img_preprocess = img_preprocess
        self.font_zhs = font_zhs
        self.char_sample = char_sample
        self.paths = paths
        self.augmenter = augmenter
        self.augment_prob = augment_prob
        self.img_size = img_size
        self.room_names = room_names
        self.junc_scale_factor = junc_scale_factor
        self.afm_scale_factor = afm_scale_factor
        self.d_max = d_max
        self.eps = 1e-4
        assert mode in [
            "basic", "instance", "semantic", "semantic_binary", "scale",
            "hawp", "hawp_line", "hawp_seg", "hawp_bseg", "line"
        ]
        self.mode = mode
    
    def rotate(self, vec, angle):
        rad = angle/180*np.pi
        qx = np.cos(rad) * vec[0] - np.sin(rad) * vec[1]
        qy = np.sin(rad) * vec[0] + np.cos(rad) * vec[1]
        return np.round(np.array([qx, qy]), 3)
        
    def mrcnn_labels(self, img, labels):
        _boxes, _labels, _masks = [], [], []
        for room in labels["Rooms"]:
            seg = room["polygon"]
            label = room_names.index(room["type"])
            mask = create_polygon_from_vertices(img.shape[:2], np.array(seg, int))
            xmin, ymin = min([x[0] for x in seg]), min([x[1] for x in seg])
            xmax, ymax = max([x[0] for x in seg]), max([x[1] for x in seg])
            if xmax-xmin<1e-4 or ymax-ymin<1e-4:
                continue
            box = [ymin, xmin, ymax, xmax]
            
            mask, scale = resize_long(mask, self.img_size)
            bias = [(self.img_size - mask.shape[0])//2, 0, (self.img_size - mask.shape[1])//2, 0]
            bias[1], bias[3] = self.img_size - mask.shape[0] - bias[0], self.img_size - mask.shape[1] - bias[2]
            mask = cv2.copyMakeBorder(mask, bias[0], bias[1], bias[2], bias[3], cv2.BORDER_CONSTANT, value=0)
            
            box[1] = box[1] * scale + bias[0]
            box[3] = box[3] * scale + bias[0]
            box[0] = box[0] * scale + bias[2]
            box[2] = box[2] * scale + bias[2]
            
            _boxes.append(box)
            _labels.append(label)
            _masks.append(mask)
        
        res = (
            torch.from_numpy(np.array(_boxes).astype(np.float32)),
            torch.from_numpy(np.array(_labels).astype(np.int64)),
            torch.from_numpy(np.array(_masks).astype(np.float32)),
        )
        return res
    
    def scale_labels(self, labels):
        return torch.from_numpy(np.array(labels['Scale']).astype(np.float32))
    
    def get_semantics(self, img, labels, is_binary=False, pt_rad=4):
        pt_rad = max(pt_rad, 1)
        
        seg_rooms = np.zeros([img.shape[0], img.shape[1], len(self.room_names)])
        for room in labels["Rooms"]:
            seg = room["polygon"]
            layer = self.room_names.index(room["type"])
            mask = create_polygon_from_vertices(img.shape[:2], np.array(seg, int))
            seg_rooms[...,layer] = seg_rooms[...,layer] + mask
        seg_rooms[...,0] = np.sum(seg_rooms, axis=2)==0
        seg_rooms = np.clip(seg_rooms.astype(float), 0, 1)
        seg_rooms = seg_rooms / np.sum(seg_rooms, axis=2, keepdims=True)

        type_dict = {"wall": 1, "door": 2, "window": 3, "other": 4}
        seg_lines = np.zeros([img.shape[0], img.shape[1], 5])
        seg_pts = np.zeros([img.shape[0], img.shape[1]])
        for line in labels["Lines"]:
            pt0, pt1 = np.array(line["start"]), np.array(line["end"])
            for pt in [pt0, pt1]:
                pt_ = int(np.round(pt[0])), int(np.round(pt[1]))
                for pt_r in range(pt_[0]-pt_rad, pt_[0]+pt_rad+1):
                    tmp_c = pt_rad - abs(pt_r - pt_[0])
                    for pt_c in range(pt_[1]-tmp_c, pt_[1]+tmp_c+1):
                        seg_pts[np.clip(pt_r, 0, img.shape[0]-1), np.clip(pt_c, 0, img.shape[0]-1)] = 1
            layer = type_dict[line["type"]]
            direction = pt1 - pt0
            length = np.linalg.norm(direction)
            if length==0: continue
            direction = direction / length
            direction90 = self.rotate(direction, 90)
            width = 120/labels["Scale"] if line["width"] is None else line["width"]
            seg = [
                pt0 + direction90 * width / 2,
                pt0 - direction90 * width / 2,
                pt1 - direction90 * width / 2,
                pt1 + direction90 * width / 2,
            ]
            mask = create_polygon_from_vertices(img.shape[:2], np.array(seg, int))
            seg_lines[...,layer] = seg_lines[...,layer] + mask
        seg_lines[...,0] = np.sum(seg_lines, axis=2)==0
        seg_lines = np.clip(seg_lines.astype(float), 0, 1)
        seg_lines = seg_lines / np.sum(seg_lines, axis=2, keepdims=True)
        
        seg_rooms, _ = resize_long(seg_rooms, self.img_size)
        bias = [(self.img_size - seg_rooms.shape[0])//2, 0, (self.img_size - seg_rooms.shape[1])//2, 0]
        bias[1], bias[3] = self.img_size - seg_rooms.shape[0] - bias[0], self.img_size - seg_rooms.shape[1] - bias[2]
        seg_rooms = cv2.copyMakeBorder(seg_rooms, bias[0], bias[1], bias[2], bias[3], cv2.BORDER_CONSTANT, value=0)
        
        seg_lines, _ = resize_long(seg_lines, self.img_size)
        bias = [(self.img_size - seg_lines.shape[0])//2, 0, (self.img_size - seg_lines.shape[1])//2, 0]
        bias[1], bias[3] = self.img_size - seg_lines.shape[0] - bias[0], self.img_size - seg_lines.shape[1] - bias[2]
        seg_lines = cv2.copyMakeBorder(seg_lines, bias[0], bias[1], bias[2], bias[3], cv2.BORDER_CONSTANT, value=0)
        
        seg_pts, _ = resize_long(seg_pts, self.img_size)
        bias = [(self.img_size - seg_pts.shape[0])//2, 0, (self.img_size - seg_pts.shape[1])//2, 0]
        bias[1], bias[3] = self.img_size - seg_pts.shape[0] - bias[0], self.img_size - seg_pts.shape[1] - bias[2]
        seg_pts = cv2.copyMakeBorder(seg_pts, bias[0], bias[1], bias[2], bias[3], cv2.BORDER_CONSTANT, value=0)
        
        seg_pts = torch.from_numpy(np.array([seg_pts]).astype(np.float32))
        if is_binary:
            seg_rooms = torch.from_numpy(seg_rooms.transpose([2,0,1]).astype(np.float32))
            seg_lines = torch.from_numpy(seg_lines.transpose([2,0,1]).astype(np.float32))
        else:
            seg_rooms = torch.from_numpy(np.argmax(seg_rooms, axis=2).astype(np.int64))
            seg_lines = torch.from_numpy(np.argmax(seg_lines, axis=2).astype(np.int64))
        
        return seg_rooms, seg_lines, seg_pts
    
    def get_batch_lines(self, img, labels):
        _scale = self.img_size / max(img.shape[0], img.shape[1])
        if img.shape[0] > img.shape[1]:
            bias = [0, (img.shape[0] - img.shape[1]) / 2 * _scale]
        else:
            bias = [(img.shape[1] - img.shape[0]) / 2 * _scale, 0]
        
        type_dict = {"wall": 0, "door": 1, "window": 2, "other": 3}
        batch_lines = []
        for line in labels["Lines"]:
            if line["width"] is None: # doors and windows do not have width
                batch_lines.append(line["start"]+line["end"]+[120/labels["Scale"], type_dict[line["type"]]])
            else:
                batch_lines.append(line["start"]+line["end"]+[line["width"], type_dict[line["type"]]])
        for i in range(len(batch_lines)):
            batch_lines[i][0] = batch_lines[i][0] * _scale + bias[0]
            batch_lines[i][1] = batch_lines[i][1] * _scale + bias[1]
            batch_lines[i][2] = batch_lines[i][2] * _scale + bias[0]
            batch_lines[i][3] = batch_lines[i][3] * _scale + bias[1]
            batch_lines[i][4] = batch_lines[i][4] * _scale
        return batch_lines
    
    def get_hawp(self, img, labels, batch_lines, pts):
        jloc_gt = np.zeros([
            1, 512 // self.junc_scale_factor, 512 // self.junc_scale_factor
        ])
        joff_gt = np.zeros([
            2, 512 // self.junc_scale_factor, 512 // self.junc_scale_factor
        ])
        afm = np.zeros([
            4, 512 // self.afm_scale_factor, 512 // self.afm_scale_factor
        ])
        line_type_gt = np.zeros([
            1, 512 // self.afm_scale_factor, 512 // self.afm_scale_factor
        ], int)
        line_width_gt = np.zeros([
            1, 512 // self.afm_scale_factor, 512 // self.afm_scale_factor
        ])
            
        afm[0] = -1
        for pt in pts:
            b_center = np.array([pt[0] / self.junc_scale_factor, pt[1] / self.junc_scale_factor], int)
            offset = (pt - b_center * self.junc_scale_factor - self.junc_scale_factor / 2) / self.junc_scale_factor
            jloc_gt[0, b_center[0], b_center[1]] = 1
            joff_gt[:, b_center[0], b_center[1]] = offset

        x_vec = np.array([1,0])
        
        for line in batch_lines:
            pt0 = np.array([line[0], line[1]]) / self.afm_scale_factor
            pt1 = np.array([line[2], line[3]]) / self.afm_scale_factor
            line_dir = pt1 - pt0
            line_len = np.linalg.norm(line_dir)
            
            if line_len<=self.eps:
                continue
            
            line_dir = line_dir/line_len
            line_norm = rot_vec(line_dir, 90)
            line_type = line[5]
            dist_box_pts = np.array([
                pt0 + line_norm * (self.d_max + self.eps), pt1 + line_norm * (self.d_max + self.eps),
                pt1 - line_norm * (self.d_max + self.eps), pt0 - line_norm * (self.d_max + self.eps),
            ])
            dist_box_poly = Polygon(dist_box_pts)
            
            rcs = []
            for r in range(int(np.ceil(min(dist_box_pts[:,0]))), int(max(dist_box_pts[:,0]))+1):
                for c in range(int(np.ceil(min(dist_box_pts[:,1]))), int(max(dist_box_pts[:,1]))+1):
                    if r>=0 and r<afm.shape[1] and c>=0 and c<afm.shape[2]:
                        rcs.append([r,c])
            if len(rcs)==0:
                continue
            
            rcs = np.array(rcs, int)
            rcs_valid = shapely.vectorized.contains(dist_box_poly, rcs[:,0], rcs[:,1])
            rcs = [tuple(rc) for i_rc, rc in enumerate(rcs) if rcs_valid[i_rc]]
            
            for r, c in rcs:
                p = np.array([r, c])
                d = proj_dist(pt0, pt1, p)
                if d>self.d_max: continue
                p_vec = line_norm if np.sum((p-pt0) * line_norm)<0 else -line_norm
                theta = np.arccos(np.clip(np.dot(p_vec, x_vec), -1, 1))
                if p_vec[1]<0:
                    theta = -theta
                v0, v1 = pt0 - p, pt1 - p
                sin_vec = np.cross(p_vec, v0) / np.linalg.norm(v0)
                if sin_vec>0:
                    vl, vr = v0, v1
                else:
                    vl, vr = v1, v0
                theta_l = np.arccos(np.clip(np.dot(p_vec,vl)/np.linalg.norm(vl), -1, 1))
                theta_r = np.arccos(np.clip(np.dot(p_vec,vr)/np.linalg.norm(vr), -1, 1))

                theta_norm = theta / 2 / np.pi + 0.5
                theta_l_norm = np.clip(theta_l * 2 / np.pi, 0, 1)
                theta_r_norm = np.clip(theta_r * 2 / np.pi, 0, 1)
                if afm[0,r,c]<0 or d<afm[0,r,c]:
                    # line_type: 1 wall, 2 door, 3 window
                    line_type_gt[0,r,c] = line_type + 1
                    line_width_gt[0,r,c] = line[4]
                    afm[:,r,c] = np.array([d, theta_norm, theta_l_norm, theta_r_norm])
        res = [
            torch.from_numpy(jloc_gt.astype(np.float32)),
            torch.from_numpy(joff_gt.astype(np.float32)),
            torch.from_numpy(afm.astype(np.float32)),
            torch.from_numpy(line_type_gt),
        ]
        return res
    
    def get_hawp2(self, img, labels, batch_lines, pts):
        lines_junc_idx0 = np.sum((pts[:, None] - batch_lines[None, :, :2]) ** 2, axis=-1).argmin(0)
        lines_junc_idx1 = np.sum((pts[:, None] - batch_lines[None, :, 2:4]) ** 2, axis=-1).argmin(0)
        lines_width = batch_lines[:, 4]
        lines_type = batch_lines[:, 5]
        
        edges_positive = [
            (a,b,c,d+1) if a<b else (b,a,c,d+1) for a,b,c,d in zip(lines_junc_idx0, lines_junc_idx1, lines_width, lines_type)
        ]
        edges_positive_set = set([x[:2] for x in edges_positive])
        edges_negative = []
        for i in range(len(pts)-1):
            for j in range(i+1,len(pts)):
                if (i,j) not in edges_positive_set:
                    edges_negative.append((i,j,0,0))
        edges_positive = np.array(edges_positive)
        edges_negative = np.array(edges_negative)
        
        num_static_pos_lines = 300
        num_static_neg_lines = 40
        
        lines_neg = np.concatenate((
            pts[edges_negative[:,0]], pts[edges_negative[:,1]], edges_negative[:, 2:]
        ), axis=-1)
        lpos = np.random.permutation(batch_lines)[:num_static_pos_lines]
        lpos[:,5] = lpos[:,5]+1
        lneg = np.random.permutation(lines_neg)[:num_static_neg_lines]
        
        lbl_mat = np.zeros([pts.shape[0]+1, pts.shape[0]+1], int)
        width_mat = np.zeros([pts.shape[0]+1, pts.shape[0]+1])
        if len(edges_positive)>0:
            width_mat[edges_positive[:,0].astype(int), edges_positive[:,1].astype(int)] = edges_positive[:,2]
            width_mat[edges_positive[:,1].astype(int), edges_positive[:,0].astype(int)] = edges_positive[:,2]
            lbl_mat[edges_positive[:,0].astype(int), edges_positive[:,1].astype(int)] = edges_positive[:,3]
            lbl_mat[edges_positive[:,1].astype(int), edges_positive[:,0].astype(int)] = edges_positive[:,3]
        
        lpre = np.concatenate((lpos, lneg),axis=0)
        _swap = np.random.random(len(lpre))>0.5
        lpre[_swap] = lpre[_swap][:,[2,3,0,1,4,5]]
        
        lpre_width = lpre[:,4].astype(np.float32)
        lpre_label = lpre[:,5].astype(int)
        lpre = lpre[:,:4].astype(np.float32)
        
        lpre, lpre_width, lpre_label, lbl_mat, width_mat = [
            torch.from_numpy(lpre),
            torch.from_numpy(lpre_width),
            torch.from_numpy(lpre_label),
            torch.from_numpy(lbl_mat),
            torch.from_numpy(width_mat),
        ]
        
        return lpre, lpre_width, lpre_label, lbl_mat, width_mat

    def __getitem__(self, index):
        do_augment = np.random.random()<self.augment_prob
        
        img_path, label_path = self.paths[index]
        with open(label_path, 'r') as f:
            labels = json.load(f)
        
        try:
            if self.augmenter is not None:
                if do_augment:
                    folder = "/".join(img_path.split("/")[:-1])
                    paths_dirs = folder.split("/")
                    for i, _dir in enumerate(paths_dirs):
                        if "opensource_aug" in _dir:
                            paths_dirs[i] = "opensource"
                    folder = "/".join(paths_dirs)

                    if "fp_300k_data" in img_path:
                        img = self.augmenter.augment(folder, labels)
                    else:
                        img = cv2.imread(img_path)
                else:
                    img = cv2.imread(img_path)
            else:
                img = cv2.imread(img_path)
        except Exception as e:
            return None, None
        if img is None:
            return None, None
        
        lbl_dict = {}
        
        lbl_dict["img_path"], lbl_dict["label_path"] = img_path, label_path
        if self.img_preprocess is not None:
            lbl_dict["img_preprocess"] = self.img_preprocess(cv2.imread(img_path))
        
        # if "basic" == self.mode:
        #     lbl_dict["img_path"], lbl_dict["label_path"] = img_path, label_path
        if "instance" == self.mode:
            lbl_dict["boxes"], lbl_dict["labels"], lbl_dict["masks"] = self.mrcnn_labels(img, labels)
        if "semantic" == self.mode or "hawp_seg" == self.mode:
            lbl_dict["seg_rooms"], lbl_dict["seg_lines"], lbl_dict["seg_pts"] = self.get_semantics(img, labels, is_binary=False)
        if "semantic_binary" == self.mode or "hawp_bseg" == self.mode:
            lbl_dict["seg_rooms"], lbl_dict["seg_lines"], lbl_dict["seg_pts"] = self.get_semantics(img, labels, is_binary=True)
        if "scale" == self.mode:
            lbl_dict["scale"] = self.scale_labels(labels)
        if "hawp_line" == self.mode or "hawp" == self.mode or "hawp_seg" == self.mode or "hawp_bseg" == self.mode:
            assert self.img_size==512
            batch_lines = np.array(self.get_batch_lines(img, labels))
            pts = np.array(list(set(
                [(line[0], line[1]) for line in batch_lines] + [(line[2], line[3]) for line in batch_lines]
            )))
            lbl_dict["batch_lines"] = batch_lines
            lbl_dict["batch_juncs"] = pts
            lbl_dict["junc_scale_factor"] = self.junc_scale_factor
            lbl_dict["afm_scale_factor"] = self.afm_scale_factor
            
            [
                lbl_dict["jloc_gt"], lbl_dict["joff_gt"], lbl_dict["afm"], lbl_dict["line_type"]
            ] = self.get_hawp(img, labels, batch_lines, pts)
            [
                lbl_dict["lpre"], lbl_dict["lpre_width"], lbl_dict["lpre_label"], lbl_dict["lbl_mat"], lbl_dict["width_mat"]
            ] = self.get_hawp2(img, labels, batch_lines, pts)
        if "line" == self.mode:
            lbl_dict["batch_lines"] = self.get_batch_lines(img, labels)
            pts = np.array(list(set(
                [(line[0], line[1]) for line in lbl_dict["batch_lines"]] + \
                [(line[2], line[3]) for line in lbl_dict["batch_lines"]]
            )))
            lbl_dict["batch_juncs"] = pts
        
        img, _ = resize_long(img, self.img_size)
        bias = [(self.img_size - img.shape[0])//2, 0, (self.img_size - img.shape[1])//2, 0]
        bias[1], bias[3] = self.img_size - img.shape[0] - bias[0], self.img_size - img.shape[1] - bias[2]
        img = cv2.copyMakeBorder(img, bias[0], bias[1], bias[2], bias[3], cv2.BORDER_REPLICATE)
        if do_augment:
            img = self.img_augseq(images=[img])[0]
            if (self.font_zhs is not None) and (self.char_sample is not None):
                img = _addRandomText(img, np.random.choice(self.font_zhs), self.char_sample, textSize=(4,12))
            if np.random.random()<0.5: # 降低分辨率
                shrink_img_size = np.random.random() * 1 + 1.5
                shrink_img_size = int(self.img_size / shrink_img_size)
                img = cv2.resize(img, (shrink_img_size, shrink_img_size))
                img = cv2.resize(img, (self.img_size, self.img_size))
        img = (img.transpose([2,0,1]) / 255.).astype(np.float32)
        
        return img, lbl_dict

    def __len__(self):
        return len(self.paths)

def collate_batch(batch):
    label_list, img_list, = [], []
   
    for _img, _label in batch:
        if _img is not None and _label is not None:
            label_list.append(_label)
            img_list.append(_img)
    
    img_list = torch.from_numpy(np.array(img_list).astype(np.float32))
    return img_list, label_list

def move_to_device(xs, targets, device):
    xs = xs.to(device)
    for target in targets:
        for key in target:
            if (key not in ["batch_lines", "batch_juncs", "img_path", "label_path"]) and ("scale_factor" not in key):
                target[key] = target[key].to(device)
    return xs, targets