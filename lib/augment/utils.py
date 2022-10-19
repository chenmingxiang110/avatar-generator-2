import subprocess
import sys
import os
import cv2
import time
import json
import random
import pickle

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import imgaug.augmenters as iaa

from random import shuffle
from tqdm import tqdm, trange
from scipy.spatial import distance
from collections import OrderedDict
from shapely.geometry import Polygon
from IPython.display import clear_output
from PIL import Image, ImageDraw, ImageFont

from lib.utils.basic import resize_short, resize_long, rot_c, rot_cc, rad2vec, rasterize
from lib.utils.data_process import fp_rasterize, create_polygon_from_vertices
from lib.utils.plot import plot_lines, plot_bboxes, plot_bboxes_and_masks

room_class = {
    -1:0, 1:1, 37:1, 44:1, 2:2, 35:2, 3:3, 38:3, 39:3, 15:3,
    4:4, 41:4, 42:4, 43:4, 40:5, 10:5, 11:5, 30:5, 31:5,
    32:5, 33:5, 34:5, 14:5, 16:5, 20:5, 45:5, 5:6, 6:6, 7:6, 8:7,
    9:7, 25:8, 26:8, 27:8, 12:9, 13:9, 29:9, 22:9, 17:10, 21:10,
    18:11, 19:12, 24:13, 28:14, 23:15, 0:16, 36:16,
}

room_names = [
    "None", "Living Room", "Dining Room", "Bedroom", "Library / Reading Room",
    "Multifunctional Room", "Bathroom", "Kitchen", "Porch", "Balcony", "Garden",
    "Garage", "Elevator", "Stairs", "Void", "Corridor / Verandah", "Others"
]

room_name_dict = {}
room_name_inv_dict = {}
for i, name in enumerate(room_names):
    room_name_dict[i] = name
    room_name_inv_dict[name] = i

def _img_binarize(img, style=None, has_pattern=False, fix_param=False):
    
    def _rand(_min, _max):
        return _min+np.random.random()*(_max-_min)
    
    def _rand_p(base, index_min, index_max):
        return base**_rand(index_min, index_max)
    
    if style=="raw":
        return img
    
    if has_pattern:
        power = 0.35 if fix_param else _rand(0.25, 0.5)
        _max = 227.5 if fix_param else _rand(225, 230)
        _max+= (power-0.25)/0.5 * 20
    else:
        power, _max = (1.0, 210) if fix_param else (_rand_p(2, -2, 2), _rand(200, 220))
    bw = (1 - np.clip(np.max(img, axis=2)/_max, 0, 1)) ** (power)
    bw = bw/np.max(bw)
    if style=="bw":
        return bw
    
    T_line = 0.8 if fix_param else _rand(0.75, 0.85)
    
    img_ = (np.max(img, axis=2)/255.<T_line).astype(float)
    if style=="binary":
        if has_pattern:
            return np.clip(img_ + bw, 0, 1)
        else:
            return img_
    
    edges = cv2.Canny(np.clip(img_ * 256, 0, 255).astype(np.uint8),100,200)
    sigma = 1 if fix_param else _rand_p(2, 0, 2)
    ksize = 3
    edges = cv2.GaussianBlur(edges, (ksize,ksize), sigma)
    edges = np.clip(edges/np.max(edges) * (np.random.random()*1+1), 0, 1)
    if style=="edge":
        return edges
    
    # fill
    fill_value = 0.5 if fix_param else _rand(0.25, 0.75)
    img_filled = np.clip(edges + bw * fill_value, 0, 1)
    return img_filled

def _augment(
    folder, labels,
    has_deco=True, has_labels=True, has_pattern=False,
    style=None, color=(0,0,0), bg_color=(1,1,1), has_frame=False, fix_param=False
):
    
    def post_process(img):
        img_b = [np.clip((1-img*(1-c))*256, 0, 255).astype(np.uint8) for c in color][::-1]
        img_b = np.array(img_b).transpose([1,2,0])
        return img_b
    
    suffix = ['.jpg', '_noItems.jpg', '_noLabels.jpg', '_noLabels_noItems.jpg']
    paths = folder + '/' + folder.split("/")[-1]
    paths = [paths+x for x in suffix]
    
    possible_styles = ["raw", "bw", "binary", "edge", "semi-edge", "fill"]
    if style is None:
        style = np.random.choice(possible_styles)
    assert style in possible_styles
    
    if style=="semi-edge":
        if has_deco:
            img = cv2.imread(paths[2])
        else:
            img = cv2.imread(paths[3])
        if has_labels:
            img_1 = cv2.imread(paths[1])
        else:
            img_1 = cv2.imread(paths[3])
            
        if not has_deco:
            img_b = _img_binarize(img_1, style="bw", has_pattern=has_pattern, fix_param=fix_param)
        else:
            img_b = _img_binarize(img, style="edge", has_pattern=has_pattern, fix_param=fix_param)
            img_b1 = _img_binarize(img_1, style="bw", has_pattern=has_pattern, fix_param=fix_param)
            img_b = np.clip(img_b + img_b1, 0, 1)
    else:
        if has_deco:
            if has_labels:
                img = cv2.imread(paths[0])
            else:
                img = cv2.imread(paths[2])
        else:
            if has_labels:
                img = cv2.imread(paths[1])
            else:
                img = cv2.imread(paths[3])
        
        img_b = _img_binarize(img, style=style, has_pattern=has_pattern, fix_param=fix_param)
    
    if style!="raw":
        img_b = post_process(img_b)
    
    valid_area = [
        min([x["start"][0] for x in labels["Lines"]] + [x["end"][0] for x in labels["Lines"]]) - 15,
        min([x["start"][1] for x in labels["Lines"]] + [x["end"][1] for x in labels["Lines"]]) - 15,
        max([x["start"][0] for x in labels["Lines"]] + [x["end"][0] for x in labels["Lines"]]) + 15,
        max([x["start"][1] for x in labels["Lines"]] + [x["end"][1] for x in labels["Lines"]]) + 15,
    ]
    valid_area[0] = int(max(valid_area[0], 0))
    valid_area[1] = int(max(valid_area[1], 0))
    valid_area[2] = int(min(valid_area[2], img.shape[0]))
    valid_area[3] = int(min(valid_area[3], img.shape[1]))
    
    if has_frame:
        res = np.copy(img[...,0]) / 255.
        res = [np.clip((1-(1-res)*(1-c))*256, 0, 255).astype(np.uint8) for c in color][::-1]
        res = np.array(res).transpose([1,2,0])
    else:
        res = (np.ones(img.shape) * 255).astype(np.uint8)
    res[valid_area[0]:valid_area[2], valid_area[1]:valid_area[3]] = img_b[
        valid_area[0]:valid_area[2], valid_area[1]:valid_area[3]
    ]
    
    res = res * np.array([[bg_color]])
    res = res.astype(np.uint8)
    return res

def _addRandomText(img, font_zh, char_sample, textSize=(5,80)):
    # 添加随机文字
    n_text = np.random.randint(1,4)
    n_text = 3
    for _ in range(n_text):
        log0, log1 = np.log(textSize[0]), np.log(textSize[1])
        textSize_ = int(np.round(np.exp(np.random.random() * (log1 - log0) + log0)))
        n_line = np.random.randint(1,9)
        n_char = np.random.randint(6,21)
        line_gap = np.random.randint(textSize_/2)
        textColor = int(min(255, 2**(np.random.random()*3+5)))
        location = np.random.randint(img.shape[0]), np.random.randint(img.shape[1])
        
        for i_line in range(n_line):
            index = np.random.randint(len(char_sample)-n_char+1)
            text = char_sample[index:index+n_char]
            loc = location[0]+(textSize_+line_gap)*i_line, location[1]
            img = _cv2AddChineseText(
                img, text, loc[::-1], font_zh, textColor=(textColor,textColor,textColor), textSize=textSize_
            )
    
    return img

def _cv2AddChineseText(img, text, position, font_zh, textColor=(0, 255, 0), textSize=30):
    if (isinstance(img, np.ndarray)):  # 判断是否OpenCV图片类型
        img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    # 创建一个可以在给定图像上绘图的对象
    draw = ImageDraw.Draw(img)
    # 字体的格式
    fontStyle = ImageFont.truetype(
        font_zh, textSize, encoding="utf-8")
    # 绘制文本
    draw.text(position, text, textColor, font=fontStyle)
    # 转换回OpenCV格式
    return cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)

def _get_ruler_pts(labels, T):

    def suppress(xs, T):
        if len(xs)>2:
            for i in range(1, len(xs)-1):
                if xs[i]-xs[i-1]<T:
                    xs[i] = xs[i-1]
            if xs[-1]-xs[-2]<T:
                xs[-2] = xs[-1]
        xs = sorted(list(set(xs)))
        return xs

    xs = sorted(list(set(
        [int(x["start"][0]) for x in labels["Lines"]]+[int(x["end"][0]) for x in labels["Lines"]]
    )))
    ys = sorted(list(set(
        [int(x["start"][1]) for x in labels["Lines"]]+[int(x["end"][1]) for x in labels["Lines"]]
    )))
    for _ in range(5):
        xs = suppress(xs, T)
    for _ in range(5):
        ys = suppress(ys, T)
    return xs, ys

def _customize(
    img, labels, label_dict=None, color_dict=None, ruler_type=None, detail_prob=0,
    font_color=(0,0,0), bg_color=(255,255,255), font_zh=None, char_sample=None, rand_text=False,
):
    """
    example:
        label_dict = {
            "Living Room": ["Wohnzimmer"], # a list of all possible mappings
            "Dining Room": ["Esszimmer"],
            "Bedroom": ["Schlafzimmer"],
            "Library / Reading Room": ["Studierzimmer"],
            "Bathroom": ["Badezimmer"],
            "Kitchen": ["Kuche"],
            "Balcony": ["Balkon"],
        }
        color_dict = {
            "Dining Room": (255,255,220),
            "Bedroom": (255,220,255),
            "Kitchen": (220,255,255),
        }
    ruler_type == 0: all horizontal ruler texts
    ruler_type == 1: horizontal and vertical ruler texts
    font_color: max==255
    """
    
    font_size = 0.3 + np.random.random() * 0.5
    detail_font_size = max(font_size * (0.5 + np.random.random() * 0.5), 0.25)
    line_width = 2
    thickness = int(max(font_size, 1))
    line_color = font_color if font_color is not None else (0,0,0)
    line_color = np.array(line_color)[::-1]
    
    img_ = img.copy()
    valid_map = np.mean(img_, axis=2, keepdims=True)!=255
    
    if color_dict is not None:
        # adding background colors
        for room in labels["Rooms"]:
            if room["type"] in color_dict and color_dict[room["type"]] is not None:
                cv2.drawContours(
                    img_, [np.array(room["polygon"]).astype(int)[:,::-1]], -1, color_dict[room["type"]], -1
                )
    img_ = img_ * (1-valid_map) + img * valid_map
    img_ = img_.astype(np.uint8)
    
    if font_color is not None:
        # adding customized labels
        for room in labels["Rooms"]:
            if room["type"] not in label_dict:
                continue
            label = np.random.choice(label_dict[room["type"]]) if room["type"] in label_dict else room["type"]
            text_size = "".join([
                str(np.round(room["size"][0]/1000, 2)), "m x ", str(np.round(room["size"][1]/1000, 2)), "m"
            ])
            
            pos = tuple([int(x) for x in np.mean(room["polygon"], axis=0)][::-1])
            w, h = cv2.getTextSize(text_size, cv2.FONT_HERSHEY_SIMPLEX, font_size, thickness)[0]
            bias = [np.random.randint(21)-10 for _ in range(2)]
            h = max(int(h*2*font_size), 16)
            pos_label = (pos[0]-np.random.randint(w//2)+bias[0], pos[1]-h+bias[1])
            pos_size = (pos[0]-w//2+bias[0], pos[1]-h//2+int(1.5*h)+bias[1])
            
            if font_zh is None:
                cv2.putText(
                    img_, label, pos_label, cv2.FONT_HERSHEY_SIMPLEX,
                    font_size, font_color[::-1], thickness
                )
            else:
                # in case the text is in Chinese
                img_ = _cv2AddChineseText(img_, label, pos_label, font_zh, textColor=font_color, textSize=h)
            
            cv2.putText(
                img_, text_size, pos_size, cv2.FONT_HERSHEY_SIMPLEX,
                font_size, font_color[::-1], thickness
            )
    
    if ruler_type is not None:
        xs, ys = _get_ruler_pts(labels, T=80)
        x_top = max(xs[0]-80, 50) - line_width//2
        x_bottom = min(xs[-1]+80, img.shape[0]-50) - line_width//2
        y_left = max(ys[0]-80, 50) - line_width//2
        y_right = min(ys[-1]+80, img.shape[1]-50) - line_width//2
        
        ruler_gap = np.random.randint(5,25)
        ruler_tick_length_0 = np.random.randint(75)
        ruler_tick_length_1 = np.random.randint(ruler_gap, 75)
        use_short_tick = np.random.random()<0.5
        
        img_[x_top:x_top+line_width, ys[0]:ys[-1]] = line_color
        img_[x_bottom:x_bottom+line_width, ys[0]:ys[-1]] = line_color
        img_[xs[0]:xs[-1], y_left:y_left+line_width] = line_color
        img_[xs[0]:xs[-1], y_right:y_right+line_width] = line_color
        if ruler_type>1:
            img_[x_top+ruler_gap:x_top+ruler_gap+line_width, ys[0]:ys[-1]] = line_color
            img_[x_bottom-ruler_gap:x_bottom-ruler_gap+line_width, ys[0]:ys[-1]] = line_color
            img_[xs[0]:xs[-1], y_left+ruler_gap:y_left+ruler_gap+line_width] = line_color
            img_[xs[0]:xs[-1], y_right-ruler_gap:y_right-ruler_gap+line_width] = line_color
        
        for i,y in enumerate(ys):
            y0 = y-line_width//2
            y1 = y0+line_width
            x0 = x_top if use_short_tick else max(x_top-ruler_tick_length_0, 0)
            x1 = x_top+ruler_gap+line_width//2 if use_short_tick else x_top+ruler_tick_length_1+line_width//2
            img_[x0:x1, y0:y1] = line_color
            x0 = x_bottom-ruler_gap if use_short_tick else x_bottom-ruler_tick_length_1
            x1 = x_bottom+line_width if use_short_tick else min(x_bottom+line_width+ruler_tick_length_0, img.shape[0])
            img_[x0:x1, y0:y1] = line_color
            
            if i!=0:
                label = str(int((ys[i]-ys[i-1]) * labels["Scale"]))
                w, h = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_size, thickness)[0]
                cv2.putText(
                    img_, label, ((ys[i]+ys[i-1]-w)//2, x_top-5), cv2.FONT_HERSHEY_SIMPLEX,
                    font_size, (font_color if font_color is not None else (0,0,0))[::-1], thickness
                )
                cv2.putText(
                    img_, label, ((ys[i]+ys[i-1]-w)//2, x_bottom+line_width+h+5), cv2.FONT_HERSHEY_SIMPLEX,
                    font_size, (font_color if font_color is not None else (0,0,0))[::-1], thickness
                )
        
        if ruler_type%2==1:
            img_vert = (np.ones([img_.shape[1], img_.shape[0], img_.shape[2]]) * 255).astype(np.uint8)
        
        for i,x in enumerate(xs):
            x0 = x-line_width//2
            x1 = x0+line_width
            y0 = y_left if use_short_tick else max(y_left-ruler_tick_length_0, 0)
            y1 = y_left+ruler_gap+line_width//2 if use_short_tick else y_left+ruler_tick_length_1+line_width//2
            img_[x0:x1, y0:y1] = line_color
            y0 = y_right-ruler_gap if use_short_tick else y_right-ruler_tick_length_1
            y1 = y_right+line_width if use_short_tick else min(y_right+line_width+ruler_tick_length_0, img.shape[1])
            img_[x0:x1, y0:y1] = line_color
            
            if i!=0:
                label = str(int((xs[i]-xs[i-1]) * labels["Scale"]))
                w, h = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_size, thickness)[0]
                if ruler_type%2==1:
                    cv2.putText(
                        img_vert, label, (
                            (xs[i]+xs[i-1]-w)//2, img_vert.shape[0]-y_right-line_width-5
                        ), cv2.FONT_HERSHEY_SIMPLEX,
                        font_size, (font_color if font_color is not None else (0,0,0))[::-1], thickness
                    )
                    img_vert = cv2.rotate(img_vert, 1)
                    cv2.putText(
                        img_vert, label, (
                            img_vert.shape[1]-(xs[i]+xs[i-1]+w)//2, y_left-5
                        ), cv2.FONT_HERSHEY_SIMPLEX,
                        font_size, (font_color if font_color is not None else (0,0,0))[::-1], thickness
                    )
                    img_vert = cv2.rotate(img_vert, 1)
                else:
                    cv2.putText(
                        img_, label, (y_left-w-5, (xs[i]+xs[i-1])//2), cv2.FONT_HERSHEY_SIMPLEX,
                        font_size, (font_color if font_color is not None else (0,0,0))[::-1], thickness
                    )
                    cv2.putText(
                        img_, label, (y_right+line_width+5, (xs[i]+xs[i-1])//2), cv2.FONT_HERSHEY_SIMPLEX,
                        font_size, (font_color if font_color is not None else (0,0,0))[::-1], thickness
                    )
        
        if ruler_type%2==1:
            img_vert = cv2.rotate(img_vert, 0)
            img_vert_valid_map = np.mean(img_vert, axis=2, keepdims=True)!=255
            img_ = img_ * (1-img_vert_valid_map) + img_vert * img_vert_valid_map
            
    if detail_prob>0:
        if ruler_type is None: # 若不是 None 则已经在上一步计算过了
            xs, ys = _get_ruler_pts(labels, T=80)
        x_min, x_max, y_min, y_max = xs[0], xs[-1], ys[0], ys[-1]
        _b = int(np.mean(img_.shape[:2])/60)
        b = np.random.random() * 5 + _b
        
        polygons = [x["polygon"] for x in labels["Rooms"]]
        for poly in polygons:
            if np.random.random()>detail_prob: continue
            for i in range(len(poly)-1):
                line = [poly[i], poly[i+1]]
                if abs(line[0][0]-line[1][0])<2:
                    x0 = line[0][0]
                    x1 = line[0][0]
                    y0 = min(line[0][1], line[1][1])
                    y1 = max(line[0][1], line[1][1])
                    label = str(int((y1-y0) * labels["Scale"]))
                    if x0-b<x_min:
                        x0, x1 = x0+b, x0+b
                    elif x0+b>x_max:
                        x0, x1 = x0-b, x0-b
                    else:
                        if np.random.random()<0.5:
                            x0, x1 = x0-b, x0-b
                        else:
                            x0, x1 = x0+b, x0+b
                    if np.random.random()<0.5:
                        img_ = np.ascontiguousarray(img_, dtype=np.uint8)
                        w, h = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, detail_font_size, thickness)[0]
                        text_pos = (
                            int((y0+y1)/2),
                            int((x0+x1)/2-line_width) if np.random.random()<0.5 else int((x0+x1)/2+h+line_width)
                        )
                        cv2.putText(
                            img_, label, text_pos, cv2.FONT_HERSHEY_SIMPLEX, detail_font_size,
                            (font_color if font_color is not None else (0,0,0))[::-1], thickness
                        )
                    x0, x1 = x0-line_width/2, x1+line_width/2
                    x0, x1, y0, y1 = int(x0), int(x1), int(y0), int(y1)
                    img_[x0:x1, y0:y1] = np.ones(img_.shape[2]) * line_color
                    img_[x0-_b//2:x1+_b//2, y0:y0+line_width] = np.ones(img_.shape[2]) * line_color
                    img_[x0-_b//2:x1+_b//2, y1-line_width:y1] = np.ones(img_.shape[2]) * line_color
                elif abs(line[0][1]-line[1][1])<2:
                    x0 = min(line[0][0], line[1][0])
                    x1 = max(line[0][0], line[1][0])
                    y0 = line[0][1]
                    y1 = line[0][1]
                    label = str(int((x1-x0) * labels["Scale"]))
                    if y0-b<y_min:
                        y0, y1 = y0+b, y0+b
                    elif y0+b>y_max:
                        y0, y1 = y0-b, y0-b
                    else:
                        if np.random.random()<0.5:
                            y0, y1 = y0+b, y0+b
                        else:
                            y0, y1 = y0-b, y0-b
                    if np.random.random()<0.5:
                        img_ = np.ascontiguousarray(img_, dtype=np.uint8)
                        w, h = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, detail_font_size, thickness)[0]
                        text_pos = (
                            int(y0+line_width) if np.random.random()<0.5 else int(y0-w-line_width),
                            int((x0+x1)/2)
                        )
                        cv2.putText(
                            img_, label, text_pos, cv2.FONT_HERSHEY_SIMPLEX, detail_font_size,
                            (font_color if font_color is not None else (0,0,0))[::-1], thickness
                        )
                    y0, y1 = y0-line_width/2, y1+line_width/2
                    x0, x1, y0, y1 = int(x0), int(x1), int(y0), int(y1)
                    img_[x0:x1, y0:y1] = np.ones(img_.shape[2]) * line_color
                    img_[x0:x0+line_width, y0-_b//2:y1+_b//2] = np.ones(img_.shape[2]) * line_color
                    img_[x1-line_width:x1, y0-_b//2:y1+_b//2] = np.ones(img_.shape[2]) * line_color
    
    img_ = img_ * np.array(bg_color)/255.
    img_ = img_.astype(np.uint8)
    if char_sample is not None and rand_text:
        img_ = _addRandomText(img_, font_zh, char_sample)
    return img_

def _build_from_labels(labels, color_dict):
    img_ = (np.ones([720,1080,3]) * 255).astype(np.uint8)
    
    # adding background colors
    for room in labels["Rooms"]:
        if room["type"] in color_dict and color_dict[room["type"]] is not None:
            cv2.drawContours(
                img_, [np.array(room["polygon"]).astype(int)[:,::-1]], -1, color_dict[room["type"]], -1
            )
    img_ = img_/255.

    
    for _type in ["door", "wall", "window"]:
        raster = np.zeros(img_.shape[:2])
        for line in labels["Lines"]:
            if line["type"]!=_type: continue
            line_ = [line["start"],line["end"]]
            linewidth = line["width"] if line["width"] is not None else 8
            _raster = rasterize([line_], img_.shape[:2], linewidth=linewidth)
            raster = raster + _raster
        raster_valid = np.clip(raster, 0, 1)>0.5
        raster_valid = np.repeat(raster_valid, img_.shape[2]).reshape(img_.shape)
        raster = raster_valid * np.reshape(color_dict[_type], [1,1,3]) / 255.
        img_ = img_ * (1-raster_valid) + raster
    
    img_ = np.clip(img_ * 255, 0, 255).astype(np.uint8)
    
    return (img_[...,::-1]).copy()