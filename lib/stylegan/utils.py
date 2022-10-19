import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

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
from torch.autograd import Variable, grad
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, utils

# from PIL import Image
from random import shuffle
from tqdm import tqdm, trange
from scipy.spatial import distance
from collections import OrderedDict
from shapely.geometry import Polygon
from IPython.display import clear_output
from PIL import Image, ImageDraw, ImageFont

from lib.utils.basic import resize_short, resize_long, rot_c, rot_cc, rad2vec, rasterize
from lib.detection_v1.dl_models import count_parameters
from lib.detection_v1.utils_contrib import seed_everything

from lib.stylegan.model import StyledGenerator, Discriminator

def img2square(data, size=512):
    img_path, lbl_path = data
    
    img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)

    if len(img.shape)==2:
        img = np.array([img, img, img]).transpose([1,2,0])
    if img.shape[2]==4: # 如果透明 则转换成白色
        B, G, R, A = cv2.split(img)
        alpha = A / 255

        R = (255 * (1 - alpha) + R * alpha).astype(np.uint8)
        G = (255 * (1 - alpha) + G * alpha).astype(np.uint8)
        B = (255 * (1 - alpha) + B * alpha).astype(np.uint8)

        img = cv2.merge((B, G, R))
    
    if lbl_path is None:
        padding = np.random.randint(31)
        img = cv2.copyMakeBorder(
            img, padding, padding, padding, padding, cv2.BORDER_CONSTANT, value=(255,255,255)
        )
        pts = np.array([[padding, padding], [img.shape[0]-padding, img.shape[1]-padding]], int)
    else:
        pts = []
        if lbl_path[-4:]=="json":
            with open(lbl_path, 'r') as f:
                lbl = json.load(f)
            for l in lbl["Lines"]:
                pts.append(l["start"])
                pts.append(l["end"])
        else:
            with open(lbl_path, 'rb') as handle:
                lbl = pickle.load(handle)
            scale = np.max(img.shape) / 480
            pts = lbl["boxes"].reshape([-1, 2]) * scale
            pts = pts[:,::-1]
        pts = np.array(pts).astype(int)
    
    bbox = np.min(pts[:,0]), np.max(pts[:,0]), np.min(pts[:,1]), np.max(pts[:,1])
    center = int((bbox[0]+bbox[1]) / 2), int((bbox[2]+bbox[3]) / 2)
    half_size = max(center[0] - bbox[0], center[1] - bbox[2])
    half_size_0, half_size_1 = [
        half_size * (np.random.random() * 0.2 + 1.15),
        half_size * (np.random.random() * 0.2 + 1.15)
    ]
    
    if center[0]-half_size_0<0:
        half_size_0 = center[0]
    if center[0]+half_size_0>=img.shape[0]:
        half_size_0 = img.shape[0] - center[0] - 1
    if center[1]-half_size_1<0:
        half_size_1 = center[1]
    if center[1]+half_size_1>=img.shape[1]:
        half_size_1 = img.shape[1] - center[1] - 1
    
    _slice = np.array([
        center[0] - half_size_0, center[0] + half_size_0 + 1,
        center[1] - half_size_1, center[1] + half_size_1 + 1,
    ]).astype(int)
    
    img_slice = np.copy(img[_slice[0]:_slice[1], _slice[2]:_slice[3]])
    shape = max(img_slice.shape[:2])
    b0, b2 = (shape - img_slice.shape[0]) // 2, (shape - img_slice.shape[1]) // 2
    b1, b3 = (shape - img_slice.shape[0]) - b0, (shape - img_slice.shape[1]) - b2
    bias = b0, b1, b2, b3
    img_slice = cv2.copyMakeBorder(img_slice, b0, b1, b2, b3, cv2.BORDER_REPLICATE)
    
    info = (_slice, bias, shape)
    img_resize = cv2.resize(img_slice, (size,size))
    return img_resize, info

def get_square_fp_img(data_srcs, size=512):
    data_src = data_srcs[np.random.choice(len(data_srcs), 1)[0]]
    data = data_src[np.random.choice(len(data_src), 1)[0]]
    img, info = img2square(data, size=size)
    if np.random.random()<0.5:
        img = img.transpose([1,0,2])
    if np.random.random()<0.5:
        img = img[::-1]
    if np.random.random()<0.5:
        img = img[:,::-1]
    return img/255., info

def get_random_batch(data_srcs, batch_size, size):
    imgs = []
    for _ in range(batch_size * 10):
        if len(imgs)>=batch_size:
            break
        try:
            img, info = get_square_fp_img(data_srcs, size=size)
            assert img.shape==(size, size, 3)
        except Exception as e:
            continue
        imgs.append(img)
    return np.array(imgs, np.float32)*2-1

def requires_grad(model, flag=True):
    for p in model.parameters():
        p.requires_grad = flag
    return

def accumulate(model1, model2, decay=0.999):
    par1 = dict(model1.named_parameters())
    par2 = dict(model2.named_parameters())

    for k in par1.keys():
        par1[k].data.mul_(decay).add_(1 - decay, par2[k].data)
    return

def adjust_lr(optimizer, lr):
    for group in optimizer.param_groups:
        mult = group.get('mult', 1)
        group['lr'] = lr * mult
    return

class my_args:
    def __init__(self, d):
        for k, v in d.items():
            setattr(self, k, v)