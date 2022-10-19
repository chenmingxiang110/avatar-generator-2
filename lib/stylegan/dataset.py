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
from lib.stylegan.utils import img2square

class MyDataset(torch.utils.data.Dataset):

    def __init__(self, data_srcs, size, seq=None, augment_prob=0):
        if seq is None:
            self.seq = iaa.Sequential([
                iaa.SomeOf((0, 5), [
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
                    iaa.Multiply((0.8, 1.2), per_channel=0.5),
                    iaa.LinearContrast((0.8, 1.2), per_channel=0.5), # improve or worsen the contrast
                ], random_order=True),
            ], random_order=True)
        else:
            self.seq = seq
        self.data_srcs = data_srcs
        self.size = size
        self.augment_prob = augment_prob

    def __getitem__(self, index):
        do_augment = np.random.random()<self.augment_prob
        data = self.data_srcs[index]
        img, _ = img2square(data, size=self.size)
        if np.random.random()<0.5:
            img = img.transpose([1,0,2])
        if np.random.random()<0.5:
            img = img[::-1]
        if np.random.random()<0.5:
            img = img[:,::-1]
        if do_augment:
            img = self.seq(images=[img])[0]
        img = img/255.
        return img

    def __len__(self):
        return len(self.data_srcs)

def collate_batch(batch):
    _batch = torch.from_numpy((np.array(batch)*2-1).transpose([0,3,1,2]).astype(np.float32))
    return _batch