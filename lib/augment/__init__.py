import os
import sys
import cv2
import copy
import time
import json
import random
import pickle

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import imgaug.augmenters as iaa

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
import torch.nn.functional as F
import torchvision.utils as vutils
import torchvision.models as models

from random import shuffle
from tqdm import tqdm, trange
from scipy.spatial import distance
from collections import OrderedDict
from shapely.geometry import Polygon
from IPython.display import clear_output
from PIL import Image, ImageDraw, ImageFont

# import kefp
# import extract
# from extract.Room import Room
# from extract.io.JsonEncoder import JsonEncoder

from lib.utils.basic import resize_short, resize_long, rot_c, rot_cc, rad2vec, rasterize
from lib.utils.data_process import fp_rasterize, create_polygon_from_vertices
from lib.utils.plot import plot_lines, plot_bboxes, plot_bboxes_and_masks
from lib.augment.utils import _augment, _customize, _build_from_labels, _get_ruler_pts

aug_config = {
    "label_dict": {
        "Living Room": ["Living Room", "客厅", "起居室"],
        "Dining Room": ["Dining Room", "Dining", "餐厅", "用餐区", "餐饮区"],
        "Bedroom": ["Bedroom", "Master Bedroom", "Cloakroom", "Walk-in Closet", "卧室", "主卧", "次卧", "衣帽间"],
        "Library / Reading Room": ["Library", "Reading Room", "书房", "书房", "办公室"],
        "Bathroom": ["Bathroom", "Washroom", "洗手间", "卫生间", "淋浴间"],
        "Kitchen": ["Kitchen", "厨房"],
        "Balcony": ["Balcony", "阳台"],
        "Corridor / Verandah": ["Hallway", "Corridor", "Verandah", "过道", "走廊"],
        "Multifunctional Room": [
            "Multifunctional Room", "Washing Machine", "Storage Room", "Storage", "Basement", "Loft", "Attic",
            "多功能间", "保姆间", "洗衣房", "娱乐区", "健身区", "影音区", "储物间", "储藏室", "阁楼", "地下室"
        ],
        "Porch": ["Entrance", "Porch", "入户花园", "玄关", "门厅"],
    },
    "mode_probs": { # the sum will be normalized to 1
        "raw": 3, "custom": 1,
    },
    "style_probs": { # the sum will be normalized to 1
        "raw": 1, "bw": 1, "binary": 1, "edge": 1, "semi-edge": 1, "fill": 1,
    },
    "accessory_probs": { # each one between 0 and 1
        "deco": 0.5,
        "label": 0.5,
        "ruler": 0.75,
        "pattern": 0.5,
        "random_font_color": 0.67,
        "random_line_color": 0.67,
        "random_bg_color": 0.67,
        "is_double_bg": 0.75,
        "custom_ruler": 0.67,
        "custom_detail_ruler": 0.5,
        "custom_room_label": 0.5,
        "custom_room_bg_color": 0.5,
        "random_text": 0.75,
    },
}

class Augmenter:
    
    def __init__(self, config, font_zhs=None, char_sample=None):
        self.font_zhs = font_zhs
        self.char_sample = char_sample
        self.label_dict = copy.deepcopy(config["label_dict"])
        self.mode_probs = copy.deepcopy(config["mode_probs"])
        self.style_probs = copy.deepcopy(config["style_probs"])
        self.accessory_probs = copy.deepcopy(config["accessory_probs"])
        
        self.mode_names = [k for k in self.mode_probs]
        self.mode_weights = np.array([self.mode_probs[k] for k in self.mode_probs])
        assert np.min(self.mode_weights)>=0
        mode_probs_sum = np.sum(self.mode_weights)
        assert mode_probs_sum>0
        self.mode_weights = self.mode_weights / mode_probs_sum
        
        self.style_names = [k for k in self.style_probs]
        self.style_weights = np.array([self.style_probs[k] for k in self.style_probs])
        assert np.min(self.style_weights)>=0
        style_probs_sum = np.sum(self.style_weights)
        assert style_probs_sum>0
        self.style_weights = self.style_weights / style_probs_sum
        
    def augment(self, folder, labels):
        _mode = self.mode_names[np.random.choice(len(self.mode_weights), 1, p=self.mode_weights)[0]]
        _style = self.style_names[np.random.choice(len(self.style_weights), 1, p=self.style_weights)[0]]
        
        res = None
        
        c = tuple(np.random.random(3) * 0.6) \
            if np.random.random()<self.accessory_probs["random_line_color"] else (0,0,0)
        font_color = (np.array(c)*255).astype(int)
        bg_color = (np.random.random(3)*128+128).astype(int) \
            if np.random.random()<self.accessory_probs["random_bg_color"] else (255,255,255)
        bg_color2 = (np.random.random(3)*128+128).astype(int) \
            if np.random.random()<self.accessory_probs["random_bg_color"] else (255,255,255)
        if _mode=="raw":
            has_deco    = np.random.random()<self.accessory_probs["deco"]
            has_labels  = np.random.random()<self.accessory_probs["label"]
            has_pattern = np.random.random()<self.accessory_probs["pattern"]
            has_frame   = np.random.random()<self.accessory_probs["ruler"]
        else:
            has_deco    = False
            has_labels  = True
            has_pattern = False
            has_frame   = True
        _label_dict = {}
        _color_dict = {key: tuple(
            [np.random.randint(128) for _ in range(3)]
        ) for key in ["wall", "window", "door"]}
        ruler_type = None
        detail_prob = 0

        if has_labels and np.random.random()<self.accessory_probs["custom_room_label"]:
            has_labels = False
            _label_dict = self.label_dict
        if _style!="raw" and np.random.random()<self.accessory_probs["custom_room_bg_color"]:
            has_color = [key for key in self.label_dict]
            shuffle(has_color)
            has_color = has_color[:int(len(has_color)*0.67)]
            for key in has_color:
                _color_dict[key] = tuple([np.random.randint(128)+128 for _ in range(3)])
        if has_frame and np.random.random()<self.accessory_probs["custom_ruler"]:
            has_frame = False
            ruler_type = np.random.randint(4)
        if np.random.random()<self.accessory_probs["custom_detail_ruler"]:
            detail_prob = 0.3
        rand_text = np.random.random()<self.accessory_probs["random_text"]

        if _mode=="raw":
            res = _augment(
                folder, labels, style=_style, color=c,
                has_deco=has_deco, has_labels=has_labels, has_pattern=has_pattern, has_frame=has_frame
            )
        else:
            res = _build_from_labels(labels, _color_dict)
        
        res = _customize(
            res, labels, label_dict=_label_dict, color_dict=_color_dict, ruler_type=ruler_type,
            detail_prob=detail_prob, font_color=tuple(font_color.tolist()), bg_color=bg_color,
            font_zh=np.random.choice(self.font_zhs), char_sample=self.char_sample, rand_text=rand_text
        )
        
        if res is not None:
            if np.random.random()<self.accessory_probs["is_double_bg"]:
                res_ = np.ones(res.shape) * np.array([[bg_color2]])
                res_ = res_.astype(np.uint8)
                xs, ys = _get_ruler_pts(labels, T=80)
                x_top = np.random.randint(max(xs[0]-80, 50))
                x_bottom = np.random.randint(min(xs[-1]+80, res.shape[0]-50), res.shape[0])
                y_left = np.random.randint(max(ys[0]-80, 50))
                y_right = np.random.randint(min(ys[-1]+80, res.shape[1]-50), res.shape[1])
                res_[x_top:x_bottom, y_left:y_right] = res[x_top:x_bottom, y_left:y_right]
                res = res_
        
        return res
