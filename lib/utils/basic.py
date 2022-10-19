import os
import cv2
import json
import time
import copy
import datetime

import numpy as np
import matplotlib.pyplot as plt

from matplotlib.collections import LineCollection
from tqdm import tqdm, trange
from shapely.geometry import Point, Polygon

color_list = np.array([
    (0.0, 0.0, 0.0),
    (1.0, 0.0, 0.0),
    (1.0, 0.5, 0.5),
    (1.0, 0.5, 0.0),
    (1.0, 1.0, 0.0),
    (0.5, 1.0, 0.5),
    (0.5, 1.0, 0.0),
    (0.0, 0.5, 0.0),
    (0.0, 1.0, 1.0),
    (0.0, 0.5, 0.8),
    (0.5, 0.5, 1.0),
    (0.0, 0.0, 1.0),
    (0.0, 0.0, 0.5),
    (0.5, 0.0, 0.5),
    (0.5, 0.0, 1.0),
    (1.0, 0.0, 1.0),
    (1.0, 0.0, 0.5),
    (0.7, 0.7, 0.7),
]) * 0.75 + 0.25

def pt_dist(pt0, pt1):
    return np.sqrt(np.sum((pt0 - pt1) ** 2))

def rasterize(lines, shape, **kwargs):
    """Rasterizes an array of lines onto an array of a specific shape using
    Matplotlib. The output lines are antialiased.

    Be wary that the line coordinates are in terms of (i, j), _not_ (x, y).

    Args: 
        lines: (line x end x coords)-shaped array of floats
        shape: (rows, columns) tuple-like

    Returns:
        arr: (rows x columns)-shaped array of floats, with line centres being
        1. and empty space being 0.
    """
    lines, shape = np.array(lines), np.array(shape)
    if len(lines)==0:
        return np.zeros(shape)

    # Flip from (i, j) to (x, y), as Matplotlib expects
    lines = lines[:, :, ::-1]

    # Create our canvas
    fig = plt.figure()
    fig.set_size_inches(shape[::-1]/fig.get_dpi() + np.array([1e-8, 1e-8])) # 加一个微小量免得因为数值问题跟我发疯

    # Here we're creating axes that cover the entire figure
    ax = fig.add_axes([0, 0, 1, 1])
    ax.axis('off')

    # And now we're setting the boundaries of the axes to match the shape
    ax.set_xlim(0, shape[1])
    ax.set_ylim(0, shape[0])
    ax.invert_yaxis()

    # Add the lines
    lines = LineCollection(lines, color='k', **kwargs)
    ax.add_collection(lines)

    # Then draw and grab the buffer
    fig.canvas.draw_idle()
    arr = (np.frombuffer(fig.canvas.get_renderer().buffer_rgba(), np.uint8)
                        .reshape((*shape, 4))
                        [:, :, :3]
                        .mean(-1))

    # And close the figure for all the IPython folk out there
    plt.close()

    # Finally, flip and reverse the array so empty space is 0.
    return 1-arr/255.

def rot_c(img):
    return img.transpose([1,0,2])[:,::-1]

def rot_cc(img):
    return img.transpose([1,0,2])[::-1]

def rad2vec(rad):
    r = np.sin(rad)
    c = np.cos(rad)
    vec = np.round(np.array([r,c]), 4)
    return vec

def rot_vec(vec, degree):
    theta = np.deg2rad(degree)
    rot_mat = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    return np.round(np.dot(rot_mat, vec), 4)

def proj_dist(a, b, p):
    return np.abs(np.cross(a-b, a-p)) / np.linalg.norm(b-a)

def softmax(x):
    """
    x: (channel, dim0, dim1)
    """
    x_ = np.exp(x-np.max(x, axis=0, keepdims=True))
    x_ = x_/np.sum(x_, axis=0, keepdims=True)
    return x_

def resize_short(img, size, interpolation=cv2.INTER_LINEAR):
    if img.shape[0]<img.shape[1]:
        return resize_height(img, size, interpolation=interpolation)
    else:
        return resize_width(img, size, interpolation=interpolation)

def resize_long(img, size, interpolation=cv2.INTER_LINEAR):
    if img.shape[0]>img.shape[1]:
        return resize_height(img, size, interpolation=interpolation)
    else:
        return resize_width(img, size, interpolation=interpolation)

def resize_height(img, size, interpolation=cv2.INTER_LINEAR):
    scale = size/img.shape[0]
    new_shape = (int(img.shape[1] * scale), int(size))
    resized = cv2.resize(img, new_shape)
    return resized, scale

def resize_width(img, size, interpolation=cv2.INTER_LINEAR):
    scale = size/img.shape[1]
    new_shape = (int(size), int(img.shape[0] * scale))
    resized = cv2.resize(img, new_shape)
    return resized, scale

def build_one_hot(values, n_values):
    new_shape = list(values.shape)
    new_shape.append(n_values)
    return np.eye(n_values)[values.reshape([-1])].reshape(new_shape)

def point_on_line(a, b, p):
    ap = p - a
    ab = b - a
    result = a + np.dot(ap, ab) / np.dot(ab, ab) * ab
    return result

def dist2poly(poly, point):
    # shapely polygon and shapely point
    if isinstance(point, Point):
        return poly.exterior.distance(point)
    else:
        return poly.exterior.distance(Point(point))

def dist_poly2poly(poly0, poly1):
    pts0 = [np.array((x,y)) for x, y in zip(*poly0.exterior.xy)]
    pts1 = [np.array((x,y)) for x, y in zip(*poly1.exterior.xy)]
    distance = min([dist2poly(poly0, p) for p in pts1]+[dist2poly(poly1, p) for p in pts0])
    return distance

def dist2lineprojection(a, b, p):
    return np.abs(np.cross(a-b, a-p)) / np.linalg.norm(b-a)

def dist2line(a, b, p):
    ap = p - a
    ab = b - a
    bp = p - b
    ba = a - b
    cos_a = np.sum(ap * ab) / max(np.linalg.norm(ap) * np.linalg.norm(ab), 1e-8)
    cos_b = np.sum(bp * ba) / max(np.linalg.norm(bp) * np.linalg.norm(ba), 1e-8)
    if cos_a>=0 and cos_b>=0:
        return dist2lineprojection(a, b, p)
    else:
        return min(np.linalg.norm(ap), np.linalg.norm(bp))
    return

def is_intersect(line1, line2):
    xdiff = (line1[0][0] - line1[1][0], line2[0][0] - line2[1][0])
    ydiff = (line1[0][1] - line1[1][1], line2[0][1] - line2[1][1])

    def det(a, b):
        return a[0] * b[1] - a[1] * b[0]

    div = det(xdiff, ydiff)
    if div == 0:
        return False

    d = (det(*line1), det(*line2))
    x = det(d, xdiff) / div
    y = det(d, ydiff) / div
    return x, y


def is_intersect_legacy(line0, line1, epsilon=1e-6):
    # 平行即便重叠也是 False (端点重合除外)
    (X1, Y1), (X2, Y2) = line0
    (X3, Y3), (X4, Y4) = line1
    pt0a, pt0b = np.array([X1, Y1]), np.array([X2, Y2])
    pt1a, pt1b = np.array([X3, Y3]), np.array([X4, Y4])
    length0 = np.linalg.norm(pt0a-pt0b)
    length1 = np.linalg.norm(pt1a-pt1b)
    
    # 投影距离小于阈值
    if np.linalg.norm(point_on_line(pt1a, pt1b, pt0a) - pt0a)<=epsilon and np.linalg.norm(pt0a-pt1a)<length1 and np.linalg.norm(pt0a-pt1b)<length1:
        return (X1, Y1)
    if np.linalg.norm(point_on_line(pt1a, pt1b, pt0b) - pt0b)<=epsilon and np.linalg.norm(pt0b-pt1a)<length1 and np.linalg.norm(pt0b-pt1b)<length1:
        return (X2, Y2)
    if np.linalg.norm(point_on_line(pt0a, pt0b, pt1a) - pt1a)<=epsilon and np.linalg.norm(pt1a-pt0a)<length1 and np.linalg.norm(pt1a-pt0b)<length0:
        return (X3, Y3)
    if np.linalg.norm(point_on_line(pt0a, pt0b, pt1b) - pt1b)<=epsilon and np.linalg.norm(pt1b-pt0a)<length1 and np.linalg.norm(pt1b-pt0b)<length0:
        return (X4, Y4)
    
    # 端点重合
    if (abs(X1-X3)<=epsilon and abs(Y1-Y3)<=epsilon) or (abs(X1-X4)<=epsilon and abs(Y1-Y4)<=epsilon):
        return (X1, Y1)
    if (abs(X2-X3)<=epsilon and abs(Y2-Y3)<=epsilon) or (abs(X2-X4)<=epsilon and abs(Y2-Y4)<=epsilon):
        return (X2, Y2)
    
    # 超出范围
    if max(X1,X2) < min(X3,X4) or min(X1,X2) > max(X3,X4) or max(Y1,Y2) < min(Y3,Y4) or min(Y1,Y2) > max(Y3,Y4):
        return None
    
    if abs(X1-X2)<=epsilon and abs(X3-X4)<=epsilon:
        return None # Parallel
    
    if abs(X1-X2)<=epsilon:
        percent = (X1-X3) / (X4-X3)
        Ya = Y3 + (Y4-Y3) * percent
        if Ya>max(Y1,Y2) or Ya<min(Y1,Y2):
            return None
        else:
            return (X1, Ya)
    elif abs(X3-X4)<=epsilon:
        percent = (X3-X1) / (X2-X1)
        Ya = Y1 + (Y2-Y1) * percent
        if Ya>max(Y3,Y4) or Ya<min(Y3,Y4):
            return None
        else:
            return (X3, Ya)
    
    A1 = (Y1-Y2)/(X1-X2)
    A2 = (Y3-Y4)/(X3-X4)
    b1 = (Y1-A1*X1) * 0.5 + (Y2-A1*X2) * 0.5 # 左侧右侧单独用其实都是对的
    b2 = (Y3-A2*X3) * 0.5 + (Y4-A2*X4) * 0.5 # 左侧右侧单独用其实都是对的
    
    if abs(A1-A2)<=epsilon:
        return None # Parallel
    
    Xa = (b2 - b1) / (A1 - A2)
    Ya = (A1 * Xa + b1) * 0.5 + (A2 * Xa + b2) * 0.5 # 左侧右侧单独用其实都是对的
    
    if (Xa < max( min(X1,X2), min(X3,X4) )) or (Xa > min( max(X1,X2), max(X3,X4) )):
        return None  # intersection is out of bound
    return (Xa, Ya)

def junc_suppress(juncs_final, juncs_score, dist_T=3):
    juncs = sorted([(x,y) for x,y in zip(juncs_final, juncs_score)], key=lambda x: x[1], reverse=True)
    res = []
    for j, s in juncs:
        if len(res)==0:
            res.append((j,s))
        else:
            min_dist = min([pt_dist(np.array(j), np.array(r[0])) for r in res])
            if min_dist>=dist_T:
                res.append((j,s))
    return np.array([x[0] for x in res]), np.array([x[1] for x in res])

def line_suppress(lines, T_seg_sup_dist=5, T_seg_sup_angle=30):
    tmp_batch_lines_hs = np.copy(lines)
    tmp_batch_lines_hs = sorted([
        (i,x) for i,x in enumerate(tmp_batch_lines_hs)
    ], key=lambda x: np.linalg.norm([x[1][2]-x[1][0],x[1][3]-x[1][1]]))
    batch_lines_hs = []
    indices = []
    for i_line, line in tmp_batch_lines_hs:
        is_valid = True
        for registered_line in batch_lines_hs:
            orders = [
                [(0,1), (2,3), (0,1), (2,3)],
                [(0,1), (2,3), (2,3), (0,1)],
                [(2,3), (0,1), (0,1), (2,3)],
                [(2,3), (0,1), (2,3), (0,1)],
            ]
            dists = [np.linalg.norm([
                line[o[0][0]]-registered_line[o[2][0]],
                line[o[0][1]]-registered_line[o[2][1]]
            ]) for o in orders]

            i_order = np.argmin(dists)
            o = orders[i_order]
            min_dist = dists[i_order]

            if min_dist<T_seg_sup_dist:
                vec0 = np.array([
                    line[o[1][0]]-line[o[0][0]],
                    line[o[1][1]]-line[o[0][1]],
                ])
                vec1 = np.array([
                    registered_line[o[3][0]]-registered_line[o[2][0]],
                    registered_line[o[3][1]]-registered_line[o[2][1]],
                ])
                _cos = np.clip(vec0.dot(vec1) / np.linalg.norm(vec0) / np.linalg.norm(vec1), -1, 1)
                angle = np.round(np.arccos(_cos) / np.pi * 180, 4)
                if angle<T_seg_sup_angle:
                    is_valid = False
                    break
        if is_valid:
            batch_lines_hs.append(line)
            indices.append(i_line)
    batch_lines_hs = np.array(batch_lines_hs, np.float32)

    return batch_lines_hs, indices
