import numpy as np
import json
import matplotlib.pyplot as plt
import math
from shapely.geometry import Point, Polygon, MultiPolygon, LinearRing
import cv2
import os
import sys
import random as rd
import copy

#WIDTH = ic.img_width
#HEIGHT = ic.img_height
COLS_LEFT = 100
#ROWS_DOWN = HEIGHT - 100
MAPPING_RATIO = 100   #cyclops 100

def equal_or_not(a,b):
    if(math.fabs(a-b)<0.0005):
        return True
    else:
        return False

#判断是否为同一个点
def equal_check(a,b):
    if(math.fabs(a[0]-b[0])<0.00005 and math.fabs(a[1]-b[1])<0.00005):
        return True
    else:
        return False

#计算两个点之间的距离
def distance_points(p1, p2):
    accuracy = 4
    x1 = round(p1[0],accuracy)
    x2 = round(p2[0],accuracy)
    y1 = round(p1[1],accuracy)
    y2 = round(p2[1],accuracy)
    res = math.sqrt(math.pow((x1-x2),2)+math.pow((y1-y2),2))
    return res

#删除src除了index为temp的点
def delete_one_point(temp, src):
    res = []
    for i in range(len(src)):
        if(i==temp):
            continue
        else:
            res.append(src[i])
    return res

#删除i+1到j,前闭后闭的点

def delete_some_points(i, j, src):
    res = []
    add = []
    for m in range(len(src)):
        if(m==i):
            res.append(src[m])
            add.append(res[m])
        elif(m>=(i+1) and m<=j):
            add.append(src[m])
        else:
            res.append(src[m])
    return res, add

def isRayIntersectsPolygon(poi,s_poi,e_poi):
    # 输入：判断点，边起点，边终点
    if s_poi[1] == e_poi[1]:  # 排除与射线平行、重合，线段首尾端点重合的情况
        return False
    if s_poi[1] > poi[1] and e_poi[1] > poi[1]:  # 线段在射线上边
        return False
    if s_poi[1] < poi[1] and e_poi[1] < poi[1]:  # 线段在射线下边
        return False
    if s_poi[1] == poi[1] and e_poi[1] > poi[1]:  # 交点为下端点，对应spoint
        return False
    if e_poi[1] == poi[1] and s_poi[1] > poi[1]:  # 交点为下端点，对应epoint
        return False
    if s_poi[0] < poi[0] and e_poi[1] < poi[1]:  # 线段在射线左边
        return False

    xseg = e_poi[0] - (e_poi[0] - s_poi[0]) * (e_poi[1] - poi[1]) / (e_poi[1] - s_poi[1])  # 求交

    if xseg < poi[0]:  # 交点在射线起点的左侧
        return False
    return True  # 排除上述情况之后

#计算单间的线段总长度
def length_sum(input):
    length = 0
    for i in range(len(input)):
        temp = i
        pre = i - 1
        if(pre == -1):
            pre = len(input) - 1
        next = i + 1
        if(next == len(input)):
            next = 0
        length += distance_points(input[temp],input[next])

    return length

#边界上的点不计入
def within(point,polygon):
    intersect=0 #交点个数

    for i in range(len(polygon)-1): #[0,len-1]
        s_poi=polygon[i]
        e_poi=polygon[i+1]
        if isRayIntersectsPolygon(point,s_poi,e_poi):
            intersect += 1 #有交点就加1

    #print("交点个数：{}".format(intersect))

    return True if intersect%2==1 else  False

def manhanttan_check(p2,p,p1):
    print("{},{},{}".format(p2,p,p1))
    if(equal_or_not(p[0],p2[0]) and equal_or_not(p[0],p1[0]) and math.fabs(p[1]-p1[1])<math.fabs(p2[1]-p1[1])):
        print("jjj")
        return True
    elif(equal_or_not(p[1],p2[1]) and equal_or_not(p[1],p1[1]) and math.fabs(p[0]-p1[0])<math.fabs(p2[0]-p1[1])):
        print("hhh")
        return True
    else:
        return False

def compute_angle(v1, v2):
    Lv1 = np.sqrt(v1.dot(v1))
    Lv2 = np.sqrt(v2.dot(v2))
    if(equal_check(Lv1,0) or equal_check(Lv2,0)):
        return 0
    cos_angle = v1.dot(v2) / (Lv1 * Lv2)
    angle = np.arccos(cos_angle)
    angle = angle * 180 / np.pi
    return angle


def compute_dis_point_to_line(p1, p2, p3):
    v1 = p2 - p3
    v2 = p3 - p1
    Lv1 = np.sqrt(v1.dot(v1))
    c = (v1.dot(v2) / (Lv1 * Lv1)) * v1
    e = v2 - c
    dis = np.sqrt(e.dot(e))
    return dis


def compute_projection_point(p1, p2, p3, p4):
    # 计算p1投影到向量p2--p3，p2--p4的法向量方向的点坐标
    v1 = p1 - p3
    v2 = p2 - p3
    Lv2 = np.sqrt(v2.dot(v2))
    v3 = ((v1.dot(v2)) / (Lv2 * Lv2)) * v2
    corner1 = p1 - v3
    v4 = p1 - p4
    v5 = ((v4.dot(v2)) / (Lv2 * Lv2)) * v2
    corner2 = p1 - v5
    return corner1, corner2


def compute_projection_vector(v1, p1, p2, p3):
    # 计算p1投影到v1的法向量方向的点坐标
    v2 = p1 - p2
    v3 = p1 - p3
    Lv1 = np.sqrt(v1.dot(v1))
    v4 = ((v2.dot(v1)) / (Lv1 * Lv1)) * v1
    corner1 = p1 - v4
    v5 = ((v3.dot(v1)) / (Lv1 * Lv1)) * v1
    corner2 = p1 - v5
    return corner1, corner2


def sharp_angle_remove_near(inter, dis_diff, angle_diff):
    #print(inter)
    res = []
    wrongIndex = []
    i = 0
    while i < len(inter) - 1:
        if (i == len(inter) - 2):
            end = 1
        else:
            end = i + 2
        # compute angle i--i+1--i+2
        v1 = inter[i] - inter[i + 1]
        v2 = inter[end] - inter[i + 1]
        angle = compute_angle(v1, v2)

        if (angle < 45):  # angle < 45
            # outer1 i的前一个点
            if (i == 0):
                outer1 = len(inter) - 2
            else:
                outer1 = i - 1

            # dis 点i+2 到直线 outer1--i 的距离
            dis = compute_dis_point_to_line(inter[end], inter[outer1], inter[i])
            # angle1 角outer1 -- i -- i+1
            v3 = inter[outer1] - inter[i]
            v4 = inter[i + 1] - inter[i]
            angle1 = compute_angle(v3, v4)
            diff = abs(angle1 - 90)

            # outer2 i+2的后一个点
            if (i == (len(inter) - 3)):
                outer2 = 1
            else:
                outer2 = end + 1

            # dis1 点i 到直线 outer2--i+2 的距离
            dis1 = compute_dis_point_to_line(inter[i], inter[outer2], inter[end])
            # anlge2 角outer2--i+2--i+1
            v3 = inter[outer2] - inter[end]
            v5 = inter[i + 1] - inter[end]
            angle2 = compute_angle(v3, v5)
            diff1 = abs(angle2 - 90)
            # 符合条件，进行点投影
            if (dis < dis_diff and diff < angle_diff):
                corner1, corner2 = compute_projection_point(inter[i + 1], inter[outer1], inter[i], inter[end])
                res.append(inter[i])
                #res.append(corner1)
                #res.append(corner2)
                wrongIndex.append(i+1)
                if (i == (len(inter) - 2)):
                    res[0] = corner2
                i = i + 1               #i = i + 2
            elif (dis1 < dis_diff and diff1 < angle_diff):
                corner2, corner1 = compute_projection_point(inter[i + 1], inter[outer2], inter[end], inter[i])
                res.append(inter[i])
                #res.append(corner1)
                #res.append(corner2)
                wrongIndex.append(i+1)
                if (i == (len(inter) - 2)):
                    res[0] = corner2
                i = i + 1
            else:
                res.append(inter[i])
                i = i + 1
        else:
            res.append(inter[i])
            i = i + 1
    res1 = []
    for index in range(len(res)):
        if index not in wrongIndex:
            res1.append(res[index])
    res1.append(res1[0])
    return res1


def sharp_angle_remove_far(inter):
    length_threshold = 1
    #print("之前inter是{}".format(inter))
    #print("inter长度为：{}".format(len(inter)))
    wrongIndex = []
    res = []
    i = 0
    while i < len(inter) - 1:
        #print("inter是：{}".format(inter))
        if (i == len(inter) - 2):
            end = 1
        else:
            end = i + 2
        v1 = inter[i] - inter[i + 1]
        v2 = inter[end] - inter[i + 1]
        angle = compute_angle(v1, v2)
        if (angle < 45):  # angle < 30
            if (i == 0):
                outer1 = len(inter) - 1       #outer1 = len(inter) - 2
            else:
                outer1 = i - 1
            if (i == (len(inter) - 3)):
                outer2 = 1
            else:
                outer2 = end + 1
            v3 = inter[i] - inter[outer1]
            v4 = inter[outer2] - inter[end]
            angle = compute_angle(v3, v4)
            #print("{},{},{},{},{}".format(outer1,i,i+1,end,outer2))
            #print("angle:{}".format(angle))
            if (angle < 20):  # angle < 10
                Lv3 = np.sqrt(v3.dot(v3))
                Lv4 = np.sqrt(v4.dot(v4))
                v3 = v3 / Lv3
                v4 = v4 / Lv4
                va3 = v3 + v4
                corner1, corner2 = compute_projection_vector(va3, inter[i + 1], inter[i], inter[end])
                #print(inter[i])
                if(distance_points(inter[i],inter[i+1])>length_threshold):
                    res.append(inter[i])
                res.append(inter[i])
                wrongIndex.append(i+1)
                # res.append(corner1)
                # res.append(corner2)
                if (i == (len(inter) - 2)):
                    res[0] = corner2
                i = i + 1
            elif (angle > 160):  # angle > 170
                Lv3 = np.sqrt(v3.dot(v3))
                Lv4 = np.sqrt(v4.dot(v4))
                v3 = -(v3 / Lv3)
                v4 = v4 / Lv4
                va3 = v3 + v4
                corner1, corner2 = compute_projection_vector(va3, inter[i + 1], inter[i], inter[end])
                #print("距离为{}".format(distance_points(inter[i], inter[i + 1])))
                if (distance_points(inter[i], inter[i + 1]) > length_threshold):
                    res.append(inter[i])
                res.append(inter[i])
                wrongIndex.append(i+1)
                # res.append(corner1)
                # res.append(corner2)
                if (i == (len(inter) - 2)):
                    res[0] = corner2
                i = i + 1
            else:
                res.append(inter[i])
                i = i + 1
        else:
            res.append(inter[i])
            i = i + 1
    #print("i是{}".format(i))
    #print("之后res是{}".format(res))
    res1 = []
    for index in range(len(res)):
        if index not in wrongIndex:
            res1.append(res[index])
    res1.append(res1[0])
    return res1


def line_point_reduce(inter, res):
    # 当有连续两条线段之间夹角小于10°时，将其变为一条线段。
    delete = []
    i = 0
    while i < len(inter) - 1:
        if (i == len(inter) - 2):
            end = 1
        else:
            end = i + 2
        v1 = inter[i + 1] - inter[i]
        v2 = inter[end] - inter[i + 1]
        Lv1 = np.sqrt(v1.dot(v1))
        Lv2 = np.sqrt(v2.dot(v2))
        angle = compute_angle(v1, v2)
        if (angle < 20):  # angle < 10
            va1 = v1 / Lv1
            va2 = v2 / Lv2
            va3 = va1 + va2
            Lva3 = np.sqrt(va3.dot(va3))
            c1 = ((v2.dot(va3)) / (Lva3 * Lva3)) * va3
            newstart = c1 + inter[i + 1]
            res[end] = newstart
            c1 = ((-v1.dot(va3)) / (Lva3 * Lva3)) * va3
            newstart = c1 + inter[i + 1]
            res[i] = newstart
            delete.append(i + 1)
            i = i + 1
        else:
            i = i + 1

    temp = 0
    res1 = []
    for index in range(len(res)):
        if index not in delete:
            res1.append(res[index])
    #for j in delete:
    #    if (j == (len(inter) - 1)):
    #        res.pop(j - temp)
    #        res[0] = res[len(res) - 1]
    #    else:
    #        res.pop(j - temp)
    #        temp += 1
    return res1


def manhattan_approach(inter, res):
    # 调整接近横平竖直的线段使其完全平行于x或y轴
    i = 0
    # x_p y_p x y正反向
    x_p = np.array([1, 0])
    y_p = np.array([0, 1])

    while i < len(inter) - 1:
        v1 = inter[i + 1] - inter[i]
        angleX = compute_angle(v1, x_p)
        angleY = compute_angle(v1, y_p)

        if (angleX < 10 or angleX > 170):  # angleX < 30
            midpoint = (inter[i + 1] + inter[i]) / 2
            va1 = inter[i + 1] - midpoint
            va2 = inter[i] - midpoint
            c1 = va1.dot(x_p) * x_p
            newpoint1 = c1 + midpoint
            c2 = va2.dot(x_p) * x_p
            newpoint2 = c2 + midpoint
            res[i + 1] = newpoint1
            res[i] = newpoint2
        if (angleY < 10 or angleY > 170):  # angleY < 30
            midpoint = (inter[i + 1] + inter[i]) / 2
            va1 = inter[i + 1] - midpoint
            va2 = inter[i] - midpoint
            c1 = va1.dot(y_p) * y_p
            newpoint1 = c1 + midpoint
            c2 = va2.dot(y_p) * y_p
            newpoint2 = c2 + midpoint
            res[i + 1] = newpoint1
            res[i] = newpoint2
        i = i + 1

    res[0] = res[len(res) - 1]
    return res

def remove_same_points(inter):
    #print("删除相同点之前:{}".format(inter))
    cycle = inter
    i = 0
    while(i < len(cycle)):
        temp = i
        next = i + 1
        if(next == len(inter)):
            next = 0
        if(equal_check(inter[temp],inter[next])):
            cycle = delete_one_point(i, cycle)
            continue
        else:
            i += 1
    #print("删除相同点之后:{}".format(cycle))
    #print("删除了{}个点".format(len(inter)-len(cycle)))
    #print("共有{}个点".format(len(cycle)))
    return cycle

def outer_rectangle_area(inter_polygon):
    minX, minY, maxX, maxY = 100, 100, -100, -100
    for i in range(len(inter_polygon)):
        if (inter_polygon[i][0] < minX):
            minX = inter_polygon[i][0]
        if (inter_polygon[i][0] > maxX):
            maxX = inter_polygon[i][0]
        if (inter_polygon[i][1] < minY):
            minY = inter_polygon[i][1]
        if (inter_polygon[i][1] > maxY):
            maxY = inter_polygon[i][1]

    rectangle_area = (maxX - minX) * (maxY - minY)
    return rectangle_area

# 移除近距离的点
def remove_near_points(inter):
    # ratio_max = 1.03
    # ratio_min = 2 - ratio_max
    area_threshold = 0.5
    distance_threshold = 0.3
    i = 0
    cycle = inter
    _, _, original_area = area_proportion(cycle)
    while(i<len(cycle)):
        isFlag = False
        for j in range(i+1, len(cycle)):
            if(distance_points(inter[i],inter[j]) < distance_threshold and abs(j-i)<=3):
                #print("删除之前是{}".format(cycle))
                res, add = delete_some_points(i, j, cycle)
                if(abs(j-i)==1):
                    cycle = res
                    # print("删除之后是{}".format(cycle))
                    isFlag = True
                    break
                if(abs(j-i)>=2 and math.fabs(length_sum(cycle)-length_sum(res)) < 0.5):
                    cycle = res
                    #print("删除之后是{}".format(cycle))
                    isFlag = True
                    break
        if(isFlag):
            continue
        else:
            i += 1
    return cycle

#移除近距离的点
# def remove_near_points(inter):
#     return

def clone_inter(inter):
    print("inter:{}".format(inter))
    res_clone = []
    for i in range(len(inter)):
        res_clone.append(inter[i])
    res_clone.append(inter[0])
    print("res_clone:{}".format(res_clone))

    return res_clone

def Vlen(v): #求取向量长度
    return math.sqrt(math.pow(v[0],2) + math.pow(v[1],2))

def projection_check(i,p_temp, inter):
    next = i + 1
    if(next == len(inter)):
        next = 0
    l1 = Vlen(inter[i]-p_temp)
    l2 = Vlen(inter[next]-p_temp)
    l = Vlen(inter[next] - inter[i])

    if(equal_check(l1+l2, l)):
        return True
    else:
        return False

def manhattan_approach_improvement(inter):  # inter is not a LinearRing
    #inter_clone = clone_inter(inter) #使其变成一个LinearRing
    res = []
    # 处理尖角、外墙不平整和外墙在模型内的问题
    #i = 0
    x_p = np.array([1, 0])
    y_p = np.array([0, 1])

    distance_threshold = 4.5

    #while i < len(inter) - 1:
    for i in range(len(inter)):
        # print(i)
        res.append(inter[i])    #####投影的点只需要加入投影后的点
        next = i + 1
        if(next == len(inter)):
            next = 0
        v1 = inter[next] - inter[i]
        angleX = compute_angle(v1, x_p)
        angleY = compute_angle(v1, y_p)
        distance_two_points = np.sqrt(
            np.square(inter[next][1] - inter[i][1]) + np.square(inter[next][0] - inter[i][0]))

        if (abs(angleX) < 1.0e-16 or abs(angleX - 180) < 1.0e-16
                or abs(angleY) < 1.0e-16 or abs(angleY - 180) < 1.0e-16):
            continue
        elif (distance_two_points <= distance_threshold):
            angleX = angleX / 180 * np.pi  ### np.tan()弧度制
            angleY = angleY / 180 * np.pi
            #print("angleX:{}---tanX:{}".format(angleX, np.tan(angleX)))
            if (abs(np.tan(angleX)) <= 1):
                v_temp = v1.dot(x_p) * x_p
                p_temp = inter[i] + v_temp
                p_mid = (inter[i] + p_temp) / 2
                #print("111{},{}---{}".format(inter[i],inter[next],p_temp))
                #print("inter:{}".format(inter))
                #if (Point(p_temp).within(LinearRing(inter))):  # if the point_temp is inside of the polygon
                if(within(p_temp,inter) or within(p_mid, inter) or projection_check(next, p_temp, inter)):  # manhanttan_check(inter[i],p_temp,inter[i-1])
                    p_temp = inter[next] - v_temp
                #res.append(inter[next])
                res.append(p_temp)
            elif (abs(np.tan(angleX)) > 1):
                v_temp = v1.dot(y_p) * y_p
                p_temp = inter[i] + v_temp
                p_mid = (inter[i] + p_temp) / 2
                #print("222{},{}---{}".format(inter[i],inter[next],p_temp))
                if (within(p_temp, inter) or within(p_mid, inter) or projection_check(next, p_temp, inter)):
                    p_temp = inter[next] - v_temp
                #res.append(inter[next])
                res.append(p_temp)
        elif (distance_two_points > distance_threshold):
            if (angleX < 20 or angleX > 160):  # angleX < 30
                v_temp = v1.dot(x_p) * x_p
                p_temp = inter[i] + v_temp
                # print(p_temp)
                if (within(p_temp, inter)):
                    p_temp = inter[next] - v_temp
                #res.append(inter[next])
                res.append(p_temp)
            elif (angleY < 20 or angleY > 160):  # angleY < 30
                v_temp = v1.dot(y_p) * y_p
                p_temp = inter[i] + v_temp
                # print(p_temp)
                if (within(p_temp, inter)):
                    p_temp = inter[next] - v_temp
                #res.append(inter[next])
                res.append(p_temp)
        #i = i + 1
    return res

def outer_rectangle(inter_polygon):
    minX, minY, maxX, maxY = 100, 100, -100, -100
    for i in range(len(inter_polygon)):
        if (inter_polygon[i][0] < minX):
            minX = inter_polygon[i][0]
        if (inter_polygon[i][0] > maxX):
            maxX = inter_polygon[i][0]
        if (inter_polygon[i][1] < minY):
            minY = inter_polygon[i][1]
        if (inter_polygon[i][1] > maxY):
            maxY = inter_polygon[i][1]

    return minX, minY, maxX, maxY

def concave_shape_remove(inter):
    print("inter是{}".format(inter))
    distance_threhold = 0.45
    isFlag = False
    minX, minY, maxX, maxY = outer_rectangle(inter)
    i = 0
    cycle = inter
    index = 0
    while(i<len(cycle)):
        index += 1
        if(index > len(cycle) + 5):
            break
        print("cycle长度：{}".format(len(cycle)))
        print("第{}个点是{}".format(i+1, cycle[i]))
        temp = i
        next = temp + 1
        if(next == len(cycle)):
            next = 0
        next2 = next + 1
        if(next2 == len(cycle)):
            next2 = 0
        next3 = next2 + 1
        if(next3 == len(cycle)):
            next3 = 0
        if(status_check(cycle[temp],cycle[next])==1 and status_check(cycle[next2],cycle[next3])==1):
            print("...!111")
            if(near_check(cycle[next][1],cycle[next2][1]) and math.fabs(cycle[next][0]-cycle[next2][0]) < distance_threhold and
            cycle[next][0]>minX and cycle[next][0]<maxX and cycle[next2][0]>minX and cycle[next2][0]<maxX):
                print("OK!111")
                isFlag = True
                res = delete_some_points(temp, next2, cycle)
                cycle =res
                print("cycle 是{}".format(cycle))
                print("cycle长度 是{}".format(len(cycle)))
                i = i + 1
                continue
            else:
                i = i + 1
                continue
        elif(status_check(cycle[temp],cycle[next])==2 and status_check(cycle[next2],cycle[next3])==2):
            print("...!222")
            if (near_check(cycle[next][0], cycle[next2][0]) and math.fabs(cycle[next][1] - cycle[next2][1]) < distance_threhold and
            cycle[next][1] > minY and cycle[next][1] < maxY and cycle[next2][1] > minY and cycle[next2][1] < maxY):
                print("OK!222")
                isFlag = True
                res = delete_some_points(temp, next2, cycle)
                cycle = res
                print("cycle 是{}".format(cycle))
                print("cycle长度 是{}".format(len(cycle)))
                i = i + 1
                continue
            else:
                i = i + 1
                continue
        else:
            i = i+1
            print("i是{}".format(i))

    return cycle, isFlag

class MyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(MyEncoder, self).default(obj)

#是否为同一点
def same_point(a,i,j):
    if(a[i][0] == a[j][0] and a[i][2] == a[j][2]):
        return False
    else:
        return True

#将图片的像素坐标转为实际空间坐标值
def mapping(px, py, minX, minZ, ROWS_DOWN):
    res = []
    x = (float)(ROWS_DOWN - px) / MAPPING_RATIO + minZ
    z = (float)(py - COLS_LEFT) / MAPPING_RATIO + minX
    res.append(round(x,4))
    res.append(round(z,4))
    return np.array(res)

def mapping_new(px,py,minX,minZ,mapping_ratio,img_size):
    res = []
    x = (float)(img_size - py) * mapping_ratio + minZ
    z = (float)(px * mapping_ratio) + minX
    res.append(round(x,4))
    res.append(round(z,4))
    return np.array(res)

#求面片中心的xz坐标
def centorid_xz(p0, p1, p2):
    xc = (p0[0]+p1[0]+p2[0])/3
    #yc = (p0[1]+p1[1]+p2[1])/3
    zc = (p0[2]+p1[2]+p2[2])/3
    #return np.array([zc, xc])
    return np.array([zc, xc])

#求面片中心的xyz坐标
def centorid_xyz(p0, p1, p2):
    xc = (p0[0]+p1[0]+p2[0])/3
    yc = (p0[1]+p1[1]+p2[1])/3
    zc = (p0[2]+p1[2]+p2[2])/3
    return np.array([zc, yc, xc])

def mkdir(path):
    path = path.strip()
    path = path.rstrip("\\")
    isExists = os.path.exists(path)

    if not isExists:
        os.makedirs(path)
        #print(path + ' 创建成功')
        return True
    else:
        return False

def file_name(file_dir):
    res = []
    index = 0
    for root, dirs, files in os.walk(file_dir):
        if(index > 0):
            break
        res = dirs
        #print('root_dir:', root)  # 当前目录路径
        #print('sub_dirs:', dirs)  # 当前路径下所有子目录
        #print('files:', files)  # 当前路径下所有非目录子文件
        index += 1

    return res

#判断面片的中心是否在bbox内
def within(p, json, mask_status):
    if(mask_status == 0):
        if (p[0] >= json[0] and p[0] <= json[2] and p[1] >= json[1] and p[1] <= json[3]):
            # print("In it!!!")
            return True
        else:
            return False
    else:
        if(cyclops_simplify_tools.within(p,json)):
            return True
        else:
            return False

def Vlen(v): #求取向量长度
    return math.sqrt(math.pow(v[0],2) + math.pow(v[1],2))

def nearest_distance(p, fpb, fpe):
    AP = p - fpb
    BP = p - fpe
    AB = fpe - fpb

    if(equal_check((AB[0]*AB[0] + AB[1]*AB[1]),0)):
        return 100

    r = AP.dot(AB) / (AB[0]*AB[0] + AB[1]*AB[1])

    if(r<=0):
        return Vlen(AP)
    elif(r>=1):
        return Vlen(BP)
    else:
        len_AC = r*Vlen(AB)
        return math.sqrt(math.pow(Vlen(AP),2) - len_AC * len_AC)

def mesh_distance(p, json, mask_status):
    min_distance = 100
    if (mask_status == 0):
        if (p[0] >= json[0] and p[0] <= json[2] and p[1] >= json[1] and p[1] <= json[3]):
            # print("In it!!!")
            return 0
        else:
            rectangle = []
            rectangle.append(np.array([json[0], json[1]]))
            rectangle.append(np.array([json[0], json[3]]))
            rectangle.append(np.array([json[2], json[3]]))
            rectangle.append(np.array([json[2], json[1]]))
            for i in range(4):
                temp = i
                next = temp + 1
                if(next == 4):
                    next = 0
                if (nearest_distance(p, rectangle[temp], rectangle[next]) < min_distance):
                    min_distance = nearest_distance(p, rectangle[temp], rectangle[next])
            return min_distance

    else:
        if (cyclops_simplify_tools.within(p, json)):
            return 0
        else:
            for i in range(len(json)):
                temp = i
                next = temp + 1
                if(next == len(json)):
                    next = 0
                if(nearest_distance(p, json[temp], json[next])<min_distance):
                    min_distance = nearest_distance(p, json[temp], json[next])
            return min_distance

#判断面片的三个点是否在mask外边过于远
def limit_range(p0, p1, p2, json, rect, mask_status):
    p = centorid_xz(p0, p1, p2)
    if (mask_status == 1):
        if (cyclops_simplify_tools.within(p, json)):
            return True
        else:
            return False
    else:
        p0_2D = np.array([p0[2], p0[0]])
        p1_2D = np.array([p1[2], p1[0]])
        p2_2D = np.array([p2[2], p2[0]])
        threshold = 0.05
        if (p[0] >= json[0] and p[0] <= json[2] and p[1] >= json[1] and p[1] <= json[3] and
            mesh_distance(p0_2D, json, mask_status) < threshold and
            mesh_distance(p1_2D, json, mask_status) < threshold and
            mesh_distance(p2_2D, json, mask_status) < threshold):
            return True
        else:
            return False

def add_mask(mask, minX, minZ, mapping_ratio, img_size):
    res = []
    for m in range(len(mask)):
        a = mapping(mask[m][0][1], mask[m][0][0], minX, minZ, mapping_ratio, img_size)
        res.append(a)

    #print(res)

    return res

def FindLargestContour(contours_t):
    max_index = 0
    max_area = 0
    for i in range(len(contours_t)):
        if(cv2.contourArea(contours_t[i])>max_area):
            max_index = i
            max_area = cv2.contourArea(contours_t[i])

    second_area = 0
    second_index = 0
    for j in range(len(contours_t)):
        if(j is not max_index):
            if(cv2.contourArea(contours_t[j])>second_area):
                second_index = j
                second_area = cv2.contourArea(contours_t[j])

    return max_area, max_index, second_area, second_index

def obtain_mask_out_rectangle(contours):
    res = []
    iX, iZ, aX, aZ = 10000, 10000, -10000, -10000
    for i in range(len(contours)):
        for j in range(len(contours[i])):
            a = contours[i][j][0]
            print("a[0]:{}".format(a[0]))
            print("a[1]:{}".format(a[1]))
            if (a[0] < iX):
                iX = a[0]
            if (a[0] > aX):
                aX = a[0]
            if (a[1] < iZ):
                iZ = a[1]
            if (a[1] > aZ):
                aZ = a[1]
    res.append(np.array([iX, iZ]))
    res.append(np.array([iX, aZ]))
    res.append(np.array([aX, aZ]))
    res.append(np.array([aX, iZ]))

    return res

# p is 2d point
def inOrout(p, json):
    if(within(p,json)):
        #print("in it!!!")
        return True
    else:
        #print("outside it!!!")
        return False

#计算两个点之间的距离
def distance_points(p1, p2):
    accuracy = 4
    x1 = round(p1[0],accuracy)
    x2 = round(p2[0],accuracy)
    y1 = round(p1[1],accuracy)
    y2 = round(p2[1],accuracy)
    res = math.sqrt(math.pow((x1-x2),2)+math.pow((y1-y2),2))
    return res

#计算两个向量的夹角
def compute_angle(v1, v2):
    Lv1 = np.sqrt(v1.dot(v1))
    Lv2 = np.sqrt(v2.dot(v2))
    cos_angle = v1.dot(v2) / (Lv1 * Lv2)
    angle = np.arccos(cos_angle)
    angle = angle * 180 / np.pi
    return angle

#判断是否相等
def equal_check(a,b):
    if(math.fabs(a-b)<0.0005):
        return True
    else:
        return False

#判断是否足够接近
def near_check(a,b):
    if (math.fabs(a - b) < 0.004):
        return True
    else:
        return False

#把 temp+1 之后的点加入 res 里
def supplement(temp,res,src):
    result = res
    for i in range(temp+1, len(src)):
        #print("第{}轮：".format(i))
        result.append(src[i])
        #print(result)
    return result

#计算单间的线段总长度
def length_sum(input):
    length = 0
    for i in range(len(input)):
        temp = i
        pre = i - 1
        if(pre == -1):
            pre = len(input) - 1
        next = i + 1
        if(next == len(input)):
            next = 0
        # next_next = next + 1
        # if(next_next == len(input)):
        #     next_next = 0
        # if(near_check_two_points(input[pre],input[next]) or near_check_two_points(input[temp],input[next_next])):
        #     continue
        length += distance_points(input[temp],input[next])

    return length

#计算单个房间的面积比，面积以及最小外接矩形
def area_proportion(inter_polygon):
    minX, minY, maxX, maxY = 100, 100, -100, -100
    outer_reactangle = []
    for i in range(len(inter_polygon)):
        if(inter_polygon[i][0]<minX):
            minX = inter_polygon[i][0]
        if(inter_polygon[i][0]>maxX):
            maxX = inter_polygon[i][0]
        if(inter_polygon[i][1]<minY):
            minY = inter_polygon[i][1]
        if(inter_polygon[i][1]>maxY):
            maxY = inter_polygon[i][1]

    #外接矩形面积
    outer_reactangle.append(np.array([minX, minY]))
    outer_reactangle.append(np.array([minX, maxY]))
    outer_reactangle.append(np.array([maxX, maxY]))
    outer_reactangle.append(np.array([maxX, minY]))
    rectangle_area = (maxX - minX) * (maxY - minY)

    #grid为网格大小
    grid = 0.015
    i=0
    count_inner = 0
    count_outer = 0
    while((minX+i*grid)<=maxX):
        j = 0
        while((minY+j*grid)<=maxY):
            count_outer += 1
            #if(Point((minX+i*grid),(minY+j*grid)).within(LinearRing(inter_polygon))):
            if(cyclops_simplify_tools.within(np.array([(minX+i*grid),(minY+j*grid)]),inter_polygon)):
                #print("在里边")
                count_inner += 1
            #print("Points分别为：{},{},({},{})".format(i,j,(minX+i*grid),(minY+j*grid)))
            j += 1
        i += 1

    # print("inter_polygon为{}：".format(inter_polygon))
    # print("i和j为：{}和{}".format(i,j))
    # print("外接矩形的大小为({},{},{},{})".format(minX,minY,maxX,maxY))
    # print("count_inner为{},count_outer为{}".format(count_inner,count_outer))
    ratio = count_inner/count_outer

    return ratio, outer_reactangle, rectangle_area*ratio

#删除src除了index为temp的点
def delete_one_point(temp, src):
    res = []
    for i in range(len(src)):
        if(i==temp):
            continue
        else:
            res.append(src[i])
    return res

#删除i+1到j,前闭后闭区间的点
def delete_some_points(i, j, src):
    res = []
    add = []
    for m in range(len(src)):
        if(m==i):
            res.append(src[m])
            add.append(res[m])
        elif(m>=(i+1) and m<=j):
            add.append(src[m])
        else:
            res.append(src[m])
    return res, add

#检查单间是否都是直线，而无斜线
def straight_line_check(room):
    flag = True
    for i in range(len(room)):
        temp = i
        next = i + 1
        if(next == len(room)):
            next = 0
        if(line_status_check(room[temp], room[next])):
            continue
        else:
            flag = False
            break

    return flag

#删除i和j,前闭后闭区间的点，并补i点
def delete_and_supple_points(temp, i, j, next3, src):
    # print(src)
    src_clone = copy.deepcopy(src)
    # src_clone.append(src_clone[0])
    res = []
    point1 = np.array([src[temp][0], src[next3][1]])
    point2 = np.array([src[temp][1], src[next3][0]])
    mid = (src[i] + src[j]) / 2
    for m in range(len(src)):
        if (m == i):
            if(line_length_cal(point1, mid) < line_length_cal(point2, mid)):
                res.append(point1)
            else:
                res.append(point2)
        elif (m == j):
            continue
        else:
            res.append(src[m])

    if(straight_line_check(res)):
        return res
    else:
        return src_clone

#用面积法过滤杂点
def points_filter_WithArea(inter_polygon):
    #print("过滤之前：inter_polygon有{}个".format(len(inter_polygon)))
    ratio_max = 1.02
    ratio_min = 2 - ratio_max
    _, _, original_area = area_proportion(inter_polygon)
    i = 0
    cycle = inter_polygon
    while(len(cycle)>=4 and i<len(cycle)):
        res = delete_one_point(i, cycle)
        _, _, current_area = area_proportion(res)
        print("原面积和现面积分别为:{},{}".format(original_area,current_area))
        if((current_area/original_area) >= ratio_min and (current_area/original_area) <= ratio_max):
            cycle = res
            i = 0
            continue
        else:
            i += 1
    #print("过滤之后：inter_polygon有{}个".format(len(cycle)))
    return cycle

#滤除多余的点
def points_filter(final):
    output_result_first = []
    for i in range(len(final)):
        base_result = length_sum(final[i])
        difference = 0.01
        cycle = final[i]
        j = 0
        while (j < len(cycle)):
            after_delete_one_point = delete_one_point(j, cycle)
            temp_result = length_sum(after_delete_one_point)
            if (math.fabs(temp_result - base_result) < difference):
                cycle = after_delete_one_point
                j = 0
            else:
                j += 1
        output_result_first.append(cycle)

    return output_result_first

#将单间用最小外接矩形替换
def rectangle_replace(input):
    threshold = 0.9
    ratio, outer_rectangle, area = area_proportion(input)
    if (ratio >= threshold):
        temp = outer_rectangle
    else:
        temp = input

    return temp

#选择排序
def selectionSort(nums):
    for i in range(len(nums) - 1):  # 遍历 len(nums)-1 次
        minIndex = i
        for j in range(i + 1, len(nums)):
            if nums[j] < nums[minIndex]:  # 更新最小值索引
                minIndex = j
        nums[i], nums[minIndex] = nums[minIndex], nums[i] # 把最小数交换到前面
    return nums

#将面积从小到大排列
def area_array(input):
    dict = {}
    array = []
    for i in range(len(input)):
        _, _, area = area_proportion(input[i])
        array.append(area)
        dict[str(area)] = i

    sort_result = selectionSort(array)

    room_sort_index = []
    for i in range(len(sort_result)):
        room_sort_index.append(dict[str(sort_result[i])])

    return room_sort_index

#bbox可视化
def visiualization(points):
    result = np.array([[points[0],points[1]],
                      [points[0],points[3]],
                      [points[2],points[1]],
                      [points[2],points[3]]])
    f1 = plt.figure(1)
    plt.subplot(211)
    plt.scatter(result[:,0],result[:,1],c = 'r',marker = 'o')
    plt.show()

#户型图点坐标可视化
def visiualization_result(result):
    result = np.array(result)
    #print(result)
    f1 = plt.figure(2)
    plt.subplot(212)
    plt.scatter(result[:,0],result[:,1],c = 'b',marker = 'o')
    plt.show()

#户型图点线图可视化
def visiualization_result_lines(final):
    plt.figure()
    for i in range(len(final)):
        #print(result)
        final[i].append([final[i][0][0],final[i][0][1]])
        x = np.array(final[i])[:,0]
        y = np.array(final[i])[:,1]
        plt.plot(x,y,c = 'r')
        plt.scatter(x,y,c = 'b',marker = 'o')
        plt.plot(final[i][len(final[i])-1][0], final[i][0][1], c='r')
    plt.show()

#判断三点的连线是否为直线
def NinetyDegree_Line_Check(p0,p1,p2):
    if(equal_check(p0[0],p2[0])):
        if(equal_check(p0[0],p1[0]) and equal_check(p2[0],p1[0])):
            return False
        else:
            return True
    elif(equal_check(p0[1],p2[1])):
        if(equal_check(p0[1],p1[1]) and equal_check(p2[1],p1[1])):
            return False
        else:
            return True

#计算矩形的面积
def area_rectangle(corner):
    return (corner[2]-corner[0])*(corner[3]-corner[1])

#将bbox和mask按照面积从小到大排列
def sort_contours_area(contours, EZ_contour, EZ_contour_area):
    for i in range(len(contours) - 1):  # 遍历 len(nums)-1 次
        minIndex = i
        for j in range(i + 1, len(contours)):
            if area_rectangle(contours[j]) < area_rectangle(contours[minIndex]):  # 更新最小值索引
                minIndex = j
        contours[i], contours[minIndex] = contours[minIndex], contours[i]  # 把最小数交换到前面
        EZ_contour[i], EZ_contour[minIndex] = EZ_contour[minIndex], EZ_contour[i]
        EZ_contour_area[i], EZ_contour_area[minIndex] = EZ_contour_area[minIndex], EZ_contour_area[i]
    return contours, EZ_contour, EZ_contour_area

#将distance按照从小到大排列
def sort_distance(contours, EZ_contour, EZ_contour_area):
    for i in range(len(contours) - 1):  # 遍历 len(nums)-1 次
        minIndex = i
        for j in range(i + 1, len(contours)):
            if contours[j] < contours[minIndex]:  # 更新最小值索引
                minIndex = j
        contours[i], contours[minIndex] = contours[minIndex], contours[i]  # 把最小数交换到前面
        EZ_contour[i], EZ_contour[minIndex] = EZ_contour[minIndex], EZ_contour[i]
        EZ_contour_area[i], EZ_contour_area[minIndex] = EZ_contour_area[minIndex], EZ_contour_area[i]
    return contours, EZ_contour, EZ_contour_area

# 判断两个矩形是否相交
def mat_inter(box1, box2):
    #print("box1:{}".format(box1))
    #print("box2:{}".format(box2))
    # box=(xA,yA,xB,yB)
    x01, y01, x02, y02 = box1[0],box1[1],box1[2],box1[3]
    x11, y11, x12, y12 = box2[0],box2[1],box2[2],box2[3]

    lx = abs((x01 + x02) / 2 - (x11 + x12) / 2)
    ly = abs((y01 + y02) / 2 - (y11 + y12) / 2)
    sax = abs(x01 - x02)
    sbx = abs(x11 - x12)
    say = abs(y01 - y02)
    sby = abs(y11 - y12)
    # print("lx:{}---(sax + sbx) / 2:{}".format(lx, (sax + sbx) / 2))
    if (lx < (sax + sbx) / 2 or equal_check(lx, (sax + sbx) / 2)) and (ly < (say + sby) / 2 or equal_check(ly, (say + sby) / 2)):
        return True
    else:
        return False

#计算两个矩形的交面积和交面积比
def solve_coincide(box1, box2):
    # box=(xA,yA,xB,yB)
    # 计算两个矩形框的重合度
    if mat_inter(box1, box2) == True:
        x01, y01, x02, y02 = box1[0],box1[1],box1[2],box1[3]
        x11, y11, x12, y12 = box2[0],box2[1],box2[2],box2[3]
        col = min(x02, x12) - max(x01, x11)
        row = min(y02, y12) - max(y01, y11)
        intersection = col * row
        area1 = (x02 - x01) * (y02 - y01)
        area2 = (x12 - x11) * (y12 - y11)
        coincide = intersection / (area1 + area2 - intersection)
        return intersection, coincide
    else:
        return -1, -1

#计算bbox是否相交及相交面积
def rectangle_inter_check(box1, box2):
    intersection, coincide = solve_coincide(box1, box2)
    #print("intersection is:{}".format(intersection))
    if(intersection>0.25):
        return True
    else:
        return False

#线段调整，去除多余的点
def line_adjustment(final):
    result = []
    for i in range(len(final)):
        flag = False
        j = 0
        cycle = final[i]
        res = []
        while(j<len(cycle)):
            if(j==0):
                res = []
            #当前点
            temp = j
            #前一个点
            pre = j-1
            if(pre == -1):
                pre = len(cycle)-1
            #下一个点
            next = j+1
            if(next == len(cycle)):
                next = 0

            if(near_check(cycle[temp][0],cycle[pre][0]) and near_check(cycle[temp][0],cycle[next][0])):
                flag = True
                res = supplement(temp, res, cycle)
                cycle = res
                j = 0
                continue
            elif(near_check(cycle[temp][1],cycle[pre][1]) and near_check(cycle[temp][1],cycle[next][1])):
                flag = True
                res = supplement(temp, res, cycle)
                cycle = res
                j = 0
                continue
            elif(NinetyDegree_Line_Check(cycle[pre],cycle[temp],cycle[next])):
                flag = True
                res = supplement(temp, res, cycle)
                cycle = res
                j = 0
                continue
            else:
                res.append(cycle[temp])
            j += 1
        # UnboundLocalError: local variable 'res' referenced before assignment  ID:cyclops-dPAokq21Q2gaJwWO
        if(len(res) != 0):
            result.append(res)
    return result

#polygon的外接矩形
def outer_rectangle(inter_polygon):
    minX, minY, maxX, maxY = 100, 100, -100, -100
    #print("长度{}".format(len(inter_polygon)))
    for i in range(len(inter_polygon)):
        if (inter_polygon[i][0] < minX):
            minX = inter_polygon[i][0]
        if (inter_polygon[i][0] > maxX):
            maxX = inter_polygon[i][0]
        if (inter_polygon[i][1] < minY):
            minY = inter_polygon[i][1]
        if (inter_polygon[i][1] > maxY):
            maxY = inter_polygon[i][1]

    return minX, minY, maxX, maxY

#检查角落区域是否符合特定的形状
def check_corner_area(temp, next, next2):
    length_threshold1 = 1.25
    length_threshold2 = 2.6
    th1 = 0.95
    th2 = 1 / th1
    dis1 = distance_points(temp,next)
    dis2 = distance_points(next,next2)
    if((dis1 + dis2)<length_threshold1):
        return True
    elif((dis1 + dis2)<length_threshold2):
        if((dis1/dis2)<th1 or (dis1/dis2)>th2):
            return True
        else:
            return False
    else:
        return False

#补充的角点是否与其他矩形相交
def corner_check(x, y, final):
    p = np.array([x, y])
    for i in range(len(final)):
        final_debug = copy.deepcopy(final[i])
        final_debug.append(final_debug[0])
        if(cyclops_simplify_tools.within(p, final_debug)):
            # print("Wrong:{}".format(i))
            # print(final_debug)
            return False

    return True

#是否位于最大值和最小值之间
def InsideMinAndMax(a, minU, maxU):
    if(a>minU and a<maxU):
        return True
    else:
        return False

#移除角落的顶柜，管道的点
def remove_corner_area(final):
    for i in range(len(final)):
        if(len(final[i])>=6):
            minX, minY, maxX, maxY = outer_rectangle(final[i])
            j = 0
            while (j < len(final[i])):
                temp = j
                next = temp + 1
                if (next == len(final[i])):
                    next = 0
                next2 = next + 1
                if (next2 == len(final[i])):
                    next2 = 0
                # print("{}--{},{},{}".format(j,final[i][temp],final[i][next],final[i][next2]))
                if (near_check(final[i][temp][0], minX) and near_check(final[i][next2][1], maxY) and
                InsideMinAndMax(final[i][temp][1], minY, maxY) and InsideMinAndMax(final[i][next2][0], minX, maxX) and
                        check_corner_area(final[i][temp], final[i][next], final[i][next2]) and
                        corner_check(minX, maxY, final)):
                    final[i][next][0] = minX
                    final[i][next][1] = maxY
                    j += 3
                    continue
                elif (near_check(final[i][temp][1], maxY) and near_check(final[i][next2][0], maxX) and
                InsideMinAndMax(final[i][temp][0], minX, maxX) and InsideMinAndMax(final[i][next2][1], minY, maxY) and
                      check_corner_area(final[i][temp], final[i][next], final[i][next2]) and
                      corner_check(maxX, maxY, final)):
                    final[i][next][0] = maxX
                    final[i][next][1] = maxY
                    j += 3
                    continue
                elif (near_check(final[i][temp][0], maxX) and near_check(final[i][next2][1], minY) and
                InsideMinAndMax(final[i][temp][1], minY, maxY) and InsideMinAndMax(final[i][next2][0], minX, maxX) and
                      check_corner_area(final[i][temp], final[i][next], final[i][next2]) and
                      corner_check(maxX, minY, final)):
                    final[i][next][0] = maxX
                    final[i][next][1] = minY
                    j += 3
                    continue
                elif (near_check(final[i][temp][1], minY) and near_check(final[i][next2][0], minX) and
                InsideMinAndMax(final[i][temp][0], minX, maxX) and InsideMinAndMax(final[i][next2][1], minY, maxY) and
                      check_corner_area(final[i][temp], final[i][next], final[i][next2]) and
                      corner_check(minX, minY, final)):
                    final[i][next][0] = minX
                    final[i][next][1] = minY
                    j += 3
                    continue
                else:
                    j += 1
                # print("j is {}".format(j))
    return final

def line_length_cal(pb, pe):
    res = []
    a = pb[0] - pe[0]
    res.append(a)
    b = pb[1] - pe[1]
    res.append(b)

    return Vlen(res)

#去除凹型和凸型线段
def remove_lousy_lines(final):
    gap_threshold_12 = 0.85 #next and next2的距离
    gap_threshold_03 = 1.5  #temp and next3的距离
    for i in range(len(final)):
        if(len(final[i])>=7):
            #minX, minY, maxX, maxY = outer_rectangle(final[i])
            # print(len(final[i]))
            j = 0
            while(j<len(final[i])):
                temp = j
                next = temp + 1
                if (next >= len(final[i])):
                    next = 0
                next2 = next + 1
                if (next2 >= len(final[i])):
                    next2 = 0
                next3 = next2 + 1
                if (next3 >= len(final[i])):
                    next3 = 0
                # print("{}--{},{},{}".format(j, next, next2, next3))
                if(status_check(final[i][temp], final[i][next]) == status_check(final[i][next2], final[i][next3])):
                    if(line_length_cal(final[i][next], final[i][next2]) < gap_threshold_12
                            and line_length_cal(final[i][temp], final[i][next3]) < gap_threshold_03
                    and max_intersection_ratio(final[i][temp], final[i][next], final[i][next2], final[i][next3], status_check(final[i][temp], final[i][next])) > 0.2):
                        final[i]= delete_and_supple_points(temp, next, next2, next3, final[i])
                j += 1

    result = line_adjustment(final)
    return remove_corner_area(result)

def remove_concave_area(final):
    for i in range(len(final)):
        print("inter是{}".format(final[i]))
        distance_threhold = 0.45
        isFlag = False
        minX, minY, maxX, maxY = outer_rectangle(final[i])
        i = 0
        cycle = final[i]
        index = 0
        while (i < len(cycle)):
            index += 1
            if (index > len(cycle) + 5):
                break
            print("cycle长度：{}".format(len(cycle)))
            print("第{}个点是{}".format(i + 1, cycle[i]))
            temp = i
            next = temp + 1
            if (next == len(cycle)):
                next = 0
            next2 = next + 1
            if (next2 == len(cycle)):
                next2 = 0
            next3 = next2 + 1
            if (next3 == len(cycle)):
                next3 = 0
            if (status_check(cycle[temp], cycle[next]) == 1 and status_check(cycle[next2], cycle[next3]) == 1):
                print("...!111")
                if (near_check(cycle[next][1], cycle[next2][1]) and cycle[next][0] > minX and cycle[next][0] < maxX and cycle[next2][0] > minX and cycle[next2][0] < maxX):
                    print("OK!111")
                    isFlag = True
                    res = delete_some_points(temp, next2, cycle)
                    cycle = res
                    print("cycle 是{}".format(cycle))
                    print("cycle长度 是{}".format(len(cycle)))
                    i = i + 1
                    continue
                else:
                    i = i + 1
                    continue
            elif (status_check(cycle[temp], cycle[next]) == 2 and status_check(cycle[next2], cycle[next3]) == 2):
                print("...!222")
                if (near_check(cycle[next][0], cycle[next2][0]) and cycle[next][1] > minY and cycle[next][1] < maxY and cycle[next2][1] > minY and cycle[next2][1] < maxY):
                    print("OK!222")
                    isFlag = True
                    res = delete_some_points(temp, next2, cycle)
                    cycle = res
                    print("cycle 是{}".format(cycle))
                    print("cycle长度 是{}".format(len(cycle)))
                    i = i + 1
                    continue
                else:
                    i = i + 1
                    continue
            else:
                i = i + 1
                print("i是{}".format(i))
    return

#判断两条线断的是否交叉及交叉比
def max_intersection_ratio(p1,p2,p3,p4,status_value):
    max_ratio = 0
    if(status_value == 1): #竖直
        max1 = max(p1[1], p2[1])
        min1 = min(p1[1], p2[1])
        max2 = max(p3[1], p4[1])
        min2 = min(p3[1], p4[1])
        if(max1 <= min2 or max2 <= min1):
            intersection_length = min(max1 - min2, max2 - min1)
            ratio1 = intersection_length / (max1 - min1)
            ratio2 = intersection_length / (max2 - min2)
            max_ratio = max(ratio1,ratio2)
        elif((min1 >= min2 and max1 <= max2)or(min2 >= min1 and max2 <= max1)):
            max_ratio = 1
        else:
            intersection_length = min(math.fabs(max1 - min2), math.fabs(max2 - min1))
            ratio1 = intersection_length / (max1 - min1)
            ratio2 = intersection_length / (max2 - min2)
            max_ratio = max(ratio1,ratio2)
    elif(status_value == 2): #水平
        max1 = max(p1[0], p2[0])
        min1 = min(p1[0], p2[0])
        max2 = max(p3[0], p4[0])
        min2 = min(p3[0], p4[0])
        if (max1 <= min2 or max2 <= min1):
            intersection_length = min(max1 - min2, max2 - min1)
            ratio1 = intersection_length / (max1 - min1)
            ratio2 = intersection_length / (max2 - min2)
            max_ratio = max(ratio1, ratio2)
        elif ((min1 >= min2 and max1 <= max2) or (min2 >= min1 and max2 <= max1)):
            max_ratio = 1
        else:
            intersection_length = min(math.fabs(max1 - min2), math.fabs(max2 - min1))
            ratio1 = intersection_length / (max1 - min1)
            ratio2 = intersection_length / (max2 - min2)
            max_ratio = max(ratio1, ratio2)
    return max_ratio

#连通性检查用
def max_intersection_ratio_connect(p1,p2,p3,p4,status_value):
    max_ratio = 0
    if(status_value == 1): #竖直
        max1 = max(p1[1], p2[1])
        min1 = min(p1[1], p2[1])
        max2 = max(p3[1], p4[1])
        min2 = min(p3[1], p4[1])
        if(max1 <= min2 or max2 <= min1):
            max_ratio = 0
        elif((min1 >= min2 and max1 <= max2)or(min2 >= min1 and max2 <= max1)):
            max_ratio = 1
        else:
            intersection_length = min(math.fabs(max1 - min2), math.fabs(max2 - min1))
            ratio1 = intersection_length / (max1 - min1)
            max_ratio = ratio1
    elif(status_value == 2): #水平
        max1 = max(p1[0], p2[0])
        min1 = min(p1[0], p2[0])
        max2 = max(p3[0], p4[0])
        min2 = min(p3[0], p4[0])
        if (max1 <= min2 or max2 <= min1):
            max_ratio = 0
        elif ((min1 >= min2 and max1 <= max2) or (min2 >= min1 and max2 <= max1)):
            max_ratio = 1
        else:
            intersection_length = min(math.fabs(max1 - min2), math.fabs(max2 - min1))
            ratio1 = intersection_length / (max1 - min1)
            max_ratio = ratio1
    return max_ratio

#检查两条线段在x,y方向是否交叉
def intersectioin_check(p1,p2,p3,p4,status,line1,line2,lines_room):
    #print("检查是否交叉")
    distance_threshold_DifferentRoom = 0.7  #是否交叉的阈值
    level1 = 0.7
    level2 = 0.43
    level3 = 0.3
    distance_threshold_SameRoom = 0.05  # cyclops 0.05  laser 0.075
    level4 = 0.25                       # cyclops 0.12  laser 0.2
    length_distance = 0.2
    if(status==1): #竖直
        max1 = max(p1[1],p2[1])
        min1 = min(p1[1],p2[1])
        max2 = max(p3[1],p4[1])
        min2 = min(p3[1],p4[1])
        if(lines_room[line1] != lines_room[line2]):
            if(max1<(min2-distance_threshold_DifferentRoom) or max2<(min1-distance_threshold_DifferentRoom)):
                return False
            else:
                if(max_intersection_ratio(p1,p2,p3,p4,1)<=0 and math.fabs(p3[0]-p1[0])<=level3):
                    return True
                elif(max_intersection_ratio(p1,p2,p3,p4,1)>0 and max_intersection_ratio(p1,p2,p3,p4,1)<0.2 and math.fabs(p3[0]-p1[0])<=level3):
                    return True
                elif(max_intersection_ratio(p1,p2,p3,p4,1)>= 0.2 and max_intersection_ratio(p1,p2,p3,p4,1)<0.4 and math.fabs(p3[0]-p1[0])<=level2):
                    return True
                elif(max_intersection_ratio(p1,p2,p3,p4,1)>=0.4 and math.fabs(p3[0]-p1[0])<=level1):
                    return True
                else:
                    return False
        else:
            if((min2-max1)>=(-1.0*distance_threshold_SameRoom) and (min2-max1)<=distance_threshold_SameRoom and math.fabs(p3[0]-p1[0])<=level4):
                return True
            elif((min1-max2)>=(-1.0*distance_threshold_SameRoom) and (min1-max2)<=distance_threshold_SameRoom and math.fabs(p3[0]-p1[0])<=level4):
                return True
            else:
                return False

    if(status==2): #水平
        max1 = max(p1[0], p2[0])
        min1 = min(p1[0], p2[0])
        max2 = max(p3[0], p4[0])
        min2 = min(p3[0], p4[0])
        if (lines_room[line1] != lines_room[line2]):
            if (max1 < (min2-distance_threshold_DifferentRoom) or max2 < (min1-distance_threshold_DifferentRoom)):
                return False
            else:
                # print("交叉比：{}".format(max_intersection_ratio(p1,p2,p3,p4,2)))
                # print("距离:{}".format(math.fabs(p3[1]-p1[1])))
                # print("Y:{},{}".format(p3[1], p1[1]))
                if(max_intersection_ratio(p1,p2,p3,p4,2)<=0 and math.fabs(p3[1]-p1[1])<=level3):
                    return True
                elif(max_intersection_ratio(p1,p2,p3,p4,2)>0 and max_intersection_ratio(p1,p2,p3,p4,2)<0.2 and math.fabs(p3[1]-p1[1])<=level3):
                    return True
                elif(max_intersection_ratio(p1,p2,p3,p4,2)>=0.2 and max_intersection_ratio(p1,p2,p3,p4,2)<0.4 and math.fabs(p3[1]-p1[1])<=level2):
                    return True
                elif(max_intersection_ratio(p1,p2,p3,p4,2)>=0.4 and math.fabs(p3[1]-p1[1])<=level1):
                    return True
                else:
                    return False
        else:
            if ((min2 - max1) >= -1.0 * distance_threshold_SameRoom and (min2 - max1) <= distance_threshold_SameRoom and math.fabs(p3[1]-p1[1])<=level4):
                return True
            elif ((min1 - max2) >= -1.0 * distance_threshold_SameRoom and (min1 - max2) <= distance_threshold_SameRoom and math.fabs(p3[1]-p1[1])<=level4):
                return True
            else:
                return False

#检查p1,p2矢量是平行于x轴还是y轴
def status_check(p1,p2):
    constant = 0.01
    if(math.fabs(p1[0]-p2[0])<constant):
        return 1
    elif(math.fabs(p1[1]-p2[1])<constant):
        return 2

#检查p1,p2矢量是横平竖直的线还是斜线
def line_status_check(p1,p2):
    constant = 0.0002
    if(math.fabs(p1[0]-p2[0])<constant):
        return True
    elif(math.fabs(p1[1]-p2[1])<constant):
        return True
    else:
        return False

#调整户型图
def floorplan(input):
    isFlag = False
    threshold_min = 0.03
    threshold_max = 0.5
    final = input
    for i in range(len(final)):
        index_test = 0
        for m in range(len(final[i])):
            index_test = index_test + 1
            current = m
            next = m+1
            if(next == len(final[i])):
                next = 0
            #print("主点为：({},{})和({},{})".format(final[i][current][0],final[i][current][1],final[i][next][0],final[i][next][1]))
            if(status_check(final[i][current],final[i][next])==1):
                for j in range(i+1,len(final)):
                    for n in range(len(final[j])):
                        o_current = n
                        o_next = n+1
                        if(o_next == len(final[j])):
                            o_next = 0
                        if(status_check(final[j][o_current],final[j][o_next])==1):
                            if(math.fabs(final[i][current][0]-final[j][o_current][0])<threshold_max and
                               math.fabs(final[i][current][0] - final[j][o_current][0]) > threshold_min and
                                    intersectioin_check(final[i][current],final[i][next],final[j][o_current],final[j][o_next],1)==True):
                                #print("!!Success!!")
                                isFlag = True
                                average1 = (final[i][current][0] + final[j][o_current][0])/2
                                final[i][current][0] = average1
                                final[i][next][0] = average1
                                final[j][o_current][0] = average1
                                final[j][o_next][0] = average1
                        else:
                            continue
            if(status_check(final[i][current],final[i][next])==2):
                for j in range(i+1,len(final)):
                    for n in range(len(final[j])):
                        o_current = n
                        o_next = n+1
                        if(o_next == len(final[j])):
                            o_next = 0
                        if(status_check(final[j][o_current],final[j][o_next])==2):
                            if(math.fabs(final[i][current][1]-final[j][o_current][1])<threshold_max and
                                math.fabs(final[i][current][1] - final[j][o_current][1]) > threshold_min and
                                    intersectioin_check(final[i][current],final[i][next],final[j][o_current],final[j][o_next],2)):
                                isFlag = True
                                average0 = (final[i][current][1] + final[j][o_current][1])/2
                                final[i][current][1] = average0
                                final[i][next][1] = average0
                                final[j][o_current][1] = average0
                                final[j][o_next][1] = average0
                        else:
                            continue
    return final,isFlag

#创建所有点的列表和线点对应关系的字典
def lines_points_dict(input_data):
    points_list = []
    for i in range(len(input_data)):
        for j in range(len(input_data[i])):
            points_list.append(input_data[i][j])
    # print(points_list)
    lines_room = {} #line属于哪个房间
    lines_dict = {} #line的编号
    index_line = 0
    for i in range(len(input_data)):
        for j in range(len(input_data[i])):
            line_id = 'line_' + str(index_line)
            if (j == (len(input_data[i]) - 1)):
                points_id = str(index_line) + "_" + str(index_line + 1 - len(input_data[i]))
            else:
                points_id = str(index_line) + "_" + str(index_line + 1)
            lines_dict[line_id] = points_id
            lines_room[line_id] = "room_" + str(i)
            index_line += 1

    return points_list, lines_dict, lines_room

def get_segment_end_index(points_index):
    positioin = points_index.find('_')
    point_begin = points_index[0:positioin]
    point_end = points_index[positioin+1:]

    return int(point_end), int(point_begin)


#获得线段的状态，1或者2,返回线段首尾端点坐标
def get_lines_status(points_list, points_index):
    positioin = points_index.find('_')
    point_begin = points_index[0:positioin]
    point_end = points_index[positioin+1:]

    return status_check(points_list[int(point_begin)],points_list[int(point_end)]), points_list[int(point_begin)], points_list[int(point_end)]

#获得线段的横纵坐标的均值
def get_lines_position(points_list, points_index):
    positioin = points_index.find('_')
    point_begin = points_index[0:positioin]
    point_end = points_index[positioin+1:]

    x_avg = (points_list[int(point_begin)][0]+points_list[int(point_end)][0])/2
    y_avg = (points_list[int(point_begin)][1]+points_list[int(point_end)][1])/2

    return x_avg, y_avg

#status_list 存放的是线的序号
#线段从左到右，从下到上排序
def sort_line_status(status_list,points_list,lines_dict,status_value):
    if(status_value==1): #竖直线段
        for i in range(len(status_list) - 1):  # 遍历 len()-1 次
            minIndex = i
            for j in range(i + 1, len(status_list)):
                x_j, _ = get_lines_position(points_list, lines_dict[status_list[j]])
                x_minIndex, _ = get_lines_position(points_list, lines_dict[status_list[minIndex]])
                if x_j < x_minIndex:  # 更新最小值索引
                    minIndex = j
            status_list[i], status_list[minIndex] = status_list[minIndex], status_list[i]
        sort_coordinateX = []
        for i in range(len(status_list)):
            x_j, _ = get_lines_position(points_list, lines_dict[status_list[i]])
            sort_coordinateX.append(x_j)
        # print("sort_coordinateX为：{}".format(sort_coordinateX))
        return status_list, sort_coordinateX
    elif (status_value == 2): #水平线段
        for i in range(len(status_list) - 1):  # 遍历 len()-1 次
            minIndex = i
            for j in range(i + 1, len(status_list)):
                _, y_j = get_lines_position(points_list, lines_dict[status_list[j]])
                _, y_minIndex = get_lines_position(points_list, lines_dict[status_list[minIndex]])
                if y_j < y_minIndex:  # 更新最小值索引
                    minIndex = j
            status_list[i], status_list[minIndex] = status_list[minIndex], status_list[i]
        sort_coordinateY = []
        for i in range(len(status_list)):
            _, y_j = get_lines_position(points_list, lines_dict[status_list[i]])
            sort_coordinateY.append(y_j)
        # print("sort_coordinateY为：{}".format(sort_coordinateY))
        return status_list, sort_coordinateY


#线段按距离分组
def group_lines_WithDistance(status_list, sort_coordinateC):
    #print("按距离分组前：{}".format(status_list))
    distance_threshold = 0.25   #original 0.55
    result = []
    result_temp = []
    result_temp.append(status_list[0])
    for i in range(1, len(sort_coordinateC)):
        temp = i
        pre = i - 1
        if ((sort_coordinateC[temp] - sort_coordinateC[pre]) < distance_threshold):
            result_temp.append(status_list[temp])
            continue
        else:
            result.append(result_temp)
            result_temp = []
            result_temp.append(status_list[temp])

    result.append(result_temp)
    #print("按距离分组后：{}".format(result))

    return result

def compare_status_list(status_list_i, j, k, coresponding_dict):
    a = int(coresponding_dict[status_list_i[j]])
    b = int(coresponding_dict[status_list_i[k]])
    if(a>b):
        return k, j, b, a
    else:
        return j, k, a, b

#得到每个分组列表里两两之间的交叉关系矩阵
def matrix_jxj(status, lines_dict, points_list, lines_room, status_value):
    matrix_status = np.zeros((len(status),len(status)))
    for i in range(len(status) - 1):
        for j in range(i+1, len(status)):
            _, point_begin_i, point_end_i = get_lines_status(points_list, lines_dict[status[i]])
            _, point_begin_j, point_end_j = get_lines_status(points_list, lines_dict[status[j]])
            if(intersectioin_check(point_begin_i,point_end_i,point_begin_j,point_end_j,status_value,status[i],status[j],lines_room)):
                matrix_status[i][j] = 1
                matrix_status[j][i] = 1
            else:
                matrix_status[i][j] = -1
                matrix_status[j][i] = -1
    #print("matrix_status是{}".format(matrix_status))
    return matrix_status

#合并两个分组
def fusion_one_line_status(result_middle, index1, index2):
    #print("result_middle,index1,index2分别为:{},{},{}".format(result_middle,index1,index2))
    res = []
    for i in range(len(result_middle)):
        if(i==index1):
            for j in range(len(result_middle[index2])):
                result_middle[index1].append(result_middle[index2][j])
            res.append(result_middle[index1])
        elif(i==index2):
            continue
        else:
            res.append(result_middle[i])

    return  res

#按是否相交来分组
def group_lines_WithIntersection(status_list, lines_dict, points_list, lines_room, status_value):
    #0, -1, 1
    result_large = []
    for i in range(len(status_list)):
        #print("status_list是{}".format(status_list))
        matrix_status = matrix_jxj(status_list[i], lines_dict, points_list, lines_room, status_value)
        group_status = np.zeros((matrix_status.shape[0],1))
        result_middle = []
        coresponding_dict = {}
        #print("status_list为：".format(status_list[i]))
        for j in range(matrix_status.shape[0] - 1):
            for k in range(j+1, matrix_status.shape[0]):
                if(group_status[j]==0): #没被归纳
                    #互相交叉，且另一条也没被归纳
                    if(matrix_status[j][k] == 1 and group_status[k] == 0):
                        result_little = []
                        result_little.append(status_list[i][j])
                        result_little.append(status_list[i][k])
                        index = len(result_middle)
                        coresponding_dict[status_list[i][j]] = str(index)
                        coresponding_dict[status_list[i][k]] = str(index)
                        group_status[j] = 1
                        group_status[k] = 1
                        result_middle.append(result_little)
                    #互相交叉，但另一条已被归纳
                    elif(matrix_status[j][k] == 1 and group_status[k] == 1):
                        result_middle[int(coresponding_dict[status_list[i][k]])].append(status_list[i][j])
                        group_status[j] = 1
                        coresponding_dict[status_list[i][j]] = coresponding_dict[status_list[i][k]]
                    #互不交叉
                    elif(matrix_status[j][k] == -1):
                        result_little = []
                        result_little.append(status_list[i][j])
                        index = len(result_middle)
                        coresponding_dict[status_list[i][j]] = str(index)
                        group_status[j] = 1
                        result_middle.append(result_little)
                elif(group_status[j]==1): #已被归纳
                    #互相交叉，但另一条未被归纳
                    if (matrix_status[j][k] == 1 and group_status[k] == 0):
                        result_middle[int(coresponding_dict[status_list[i][j]])].append(status_list[i][k])
                        group_status[k] = 1
                        coresponding_dict[status_list[i][k]] = coresponding_dict[status_list[i][j]]
                    #互相交叉，且另一条也被归纳
                    elif(matrix_status[j][k] == 1 and group_status[k] == 1):
                        if(coresponding_dict[status_list[i][j]] == coresponding_dict[status_list[i][k]]):
                            continue
                        else:
                            #print("status_list为：{}".format(status_list[i]))
                            #print("corespondint_dict为：{}".format(coresponding_dict))
                            #print("j和k值为：{},{}".format(j,k))
                            min_index, max_index, min_dict_index, max_dict_index = compare_status_list(status_list[i],j,k,coresponding_dict)
                            #print("四个值分别为：{},{},{},{}".format(min_index,max_index,min_dict_index,max_dict_index))
                            for m in range(len(result_middle[max_dict_index])):
                                coresponding_dict[result_middle[max_dict_index][m]] \
                                    = coresponding_dict[result_middle[min_dict_index][0]]
                            result_middle = fusion_one_line_status(result_middle, min_dict_index, max_dict_index)

                    #互不交叉
                    elif(matrix_status[j][k] == -1):
                        continue
        #考虑分组里的最后一条线段是否被分类
        last = len(status_list[i])-1
        if(group_status[last]==0):
            result_little = []
            result_little.append(status_list[i][last])
            index = len(result_middle)
            coresponding_dict[status_list[i][last]] = str(index)
            result_middle.append(result_little)
        result_large.append(result_middle)

    return result_large

#进行线段融合
def lines_fusion(lines_status, lines_dict, points_list, status_value):
    #print("lines_status是{}".format(lines_status))
    if(status_value == 1 and len(lines_status) > 1):
        max_length = 0
        for i in range(len(lines_status)):
            points_index = lines_dict[lines_status[i]]
            positioin = points_index.find('_')
            point_begin_index = points_index[0:positioin]
            point_end_index = points_index[positioin + 1:]
            point_begin = points_list[int(point_begin_index)]
            point_end = points_list[int(point_end_index)]
            #print("({},{})和({},{})".format(point_begin[0],point_begin[1],point_end[0],point_end[1]))
            if(distance_points(point_begin,point_end)>max_length):
                max_length = distance_points(point_begin,point_end)
        #print("max_length为：{}".format(max_length))
        coordinate_sumX = 0
        minY = 100
        minY_Index = 0
        maxY = -100
        maxY_Index = 0
        weight_sum = 0
        for i in range(len(lines_status)):
            points_index = lines_dict[lines_status[i]]
            positioin = points_index.find('_')
            point_begin_index = points_index[0:positioin]
            point_end_index = points_index[positioin + 1:]
            point_begin = points_list[int(point_begin_index)]
            point_end = points_list[int(point_end_index)]
            # 每条线段根据自己的长度有不同的权重，权重为其长度/最长线段长度的平方
            if (max_length != 0):
                weight = math.pow((distance_points(point_begin, point_end)/max_length),2)
            else:
                weight = 1
                #print("max_lenght是0，状态1")
            weight_sum += weight
            coordinate_sumX += point_begin[0] * weight
            coordinate_sumX += point_end[0] * weight
            if(point_begin[1]<minY):
                minY = point_begin[1]
                minY_Index = point_begin_index
            if(point_begin[1]>maxY):
                maxY = point_begin[1]
                maxY_Index = point_begin_index
            if(point_end[1]<minY):
                minY = point_end[1]
                minY_Index = point_end_index
            if(point_end[1]>maxY):
                maxY = point_end[1]
                maxY_Index = point_end_index
        coordinate_avgX = coordinate_sumX / (2*weight_sum)
        for i in range(len(lines_status)):
            points_index = lines_dict[lines_status[i]]
            positioin = points_index.find('_')
            point_begin_index = points_index[0:positioin]
            point_end_index = points_index[positioin + 1:]
            points_list[int(point_begin_index)][0] = coordinate_avgX
            points_list[int(point_end_index)][0] = coordinate_avgX
            if(i>0):
                lines_dict.pop(lines_status[i])
        fusion_index = str(minY_Index)+ '_'+ str(maxY_Index)
        lines_dict[lines_status[0]] = fusion_index

    if(status_value==2 and len(lines_status) > 1):
        max_length = 0
        for i in range(len(lines_status)):
            points_index = lines_dict[lines_status[i]]
            positioin = points_index.find('_')
            point_begin_index = points_index[0:positioin]
            point_end_index = points_index[positioin + 1:]
            point_begin = points_list[int(point_begin_index)]
            point_end = points_list[int(point_end_index)]
            #print("({},{})和({},{})".format(point_begin[0], point_begin[1], point_end[0], point_end[1]))
            if (distance_points(point_begin, point_end) > max_length):
                max_length = distance_points(point_begin, point_end)
        #print("max_length为：{}".format(max_length))

        coordinate_sumY = 0
        minX = 100
        minX_Index = 0
        maxX = -100
        maxX_Index = 0
        weight_sum = 0
        for i in range(len(lines_status)):
            points_index = lines_dict[lines_status[i]]
            positioin = points_index.find('_')
            point_begin_index = points_index[0:positioin]
            point_end_index = points_index[positioin + 1:]
            point_begin = points_list[int(point_begin_index)]
            point_end = points_list[int(point_end_index)]
            # 每条线段根据自己的长度有不同的权重，权重为其长度/最长线段长度的平方
            if(max_length != 0):
                weight = math.pow(distance_points(point_begin, point_end) / max_length,2)
            else :
                weight = 1
                print("max_lenght是0，状态1")
            weight_sum += weight
            coordinate_sumY += point_begin[1] * weight
            coordinate_sumY += point_end[1] * weight
            if (point_begin[0] < minX):
                minX = point_begin[0]
                minX_Index = point_begin_index
            if (point_begin[0] > maxX):
                maxX = point_begin[0]
                maxX_Index = point_begin_index
            if (point_end[0] < minX):
                minX = point_end[0]
                minX_Index = point_end_index
            if (point_end[0] > maxX):
                maxX = point_end[0]
                maxX_Index = point_end_index
        coordinate_avgY = coordinate_sumY / (2*weight_sum)
        for i in range(len(lines_status)):
            points_index = lines_dict[lines_status[i]]
            positioin = points_index.find('_')
            point_begin_index = points_index[0:positioin]
            point_end_index = points_index[positioin + 1:]
            points_list[int(point_begin_index)][1] = coordinate_avgY
            points_list[int(point_end_index)][1] = coordinate_avgY
            if(i > 0):
                lines_dict.pop(lines_status[i])
        fusion_index = str(minX_Index)+ '_'+ str(maxX_Index)
        lines_dict[lines_status[0]] = fusion_index

    return lines_dict, points_list

#将lines分组
def group_lines(points_list, lines_dict, lines_room, output_path, minX, minZ, img_w, img_h, mapping_ratio, DEBUG_MODE):
    lines_status1 = []  #x坐标接近，竖直线段
    lines_status2 = []  #y坐标接近，水平线段
    for line_key in lines_dict:
        points_index = lines_dict[line_key]
        line_status_value, _, _ = get_lines_status(points_list,points_index)
        if(line_status_value == 1):
            lines_status1.append(line_key)
        elif(line_status_value==2):
            lines_status2.append(line_key)

    #Bug
    if (len(lines_status1) == 0 or len(lines_status2) == 0):
        print("竖直点云数据残缺，房屋模型欠佳")
        sys.exit(-3)

    #线段从左到右，从下到上排序
    lines_status1, sort_coordinateX = sort_line_status(lines_status1,points_list,lines_dict,1)
    lines_status2, sort_coordinateY = sort_line_status(lines_status2,points_list,lines_dict,2)

    #线段按距离分组
    lines_distance1 = group_lines_WithDistance(lines_status1,sort_coordinateX)
    lines_distance2 = group_lines_WithDistance(lines_status2,sort_coordinateY)

    #线段按交叉临近关系分组
    lines_intersection1 = group_lines_WithIntersection(lines_distance1,lines_dict,points_list,lines_room,1)
    lines_intersection2 = group_lines_WithIntersection(lines_distance2,lines_dict,points_list,lines_room,2)

    if DEBUG_MODE:
        output_fusion_image(lines_intersection1, lines_intersection2, points_list, lines_dict, output_path, minX, minZ, img_w, img_h, mapping_ratio)

    for i in range(len(lines_intersection1)):
        for j in range(len(lines_intersection1[i])):
            lines_dict, points_list = lines_fusion(lines_intersection1[i][j],lines_dict,points_list,1)

    for i in range(len(lines_intersection2)):
        for j in range(len(lines_intersection2[i])):
            lines_dict, points_list = lines_fusion(lines_intersection2[i][j],lines_dict,points_list,2)

    return points_list, lines_dict

#将lines分组
# def group_lines(points_list, lines_dict, lines_room, output_path, minX, minZ, img_size, mapping_ratio, DEBUG_MODE):
#     lines_status1 = []  #x坐标接近，竖直线段
#     lines_status2 = []  #y坐标接近，水平线段
#     for line_key in lines_dict:
#         points_index = lines_dict[line_key]
#         line_status_value, _, _ = get_lines_status(points_list,points_index)
#         if(line_status_value == 1):
#             lines_status1.append(line_key)
#         elif(line_status_value==2):
#             lines_status2.append(line_key)

#     #Bug
#     if (len(lines_status1) == 0 or len(lines_status2) == 0):
#         print("竖直点云数据残缺，房屋模型欠佳")
#         sys.exit(-3)

#     #线段从左到右，从下到上排序
#     lines_status1, sort_coordinateX = sort_line_status(lines_status1,points_list,lines_dict,1)
#     lines_status2, sort_coordinateY = sort_line_status(lines_status2,points_list,lines_dict,2)

#     #线段按距离分组
#     lines_distance1 = group_lines_WithDistance(lines_status1,sort_coordinateX)
#     lines_distance2 = group_lines_WithDistance(lines_status2,sort_coordinateY)

#     #线段按交叉临近关系分组
#     lines_intersection1 = group_lines_WithIntersection(lines_distance1,lines_dict,points_list,lines_room,1)
#     lines_intersection2 = group_lines_WithIntersection(lines_distance2,lines_dict,points_list,lines_room,2)

#     if DEBUG_MODE:
#         output_fusion_image(lines_intersection1, lines_intersection2, points_list, lines_dict, output_path, minX, minZ, img_size, mapping_ratio)

#     for i in range(len(lines_intersection1)):
#         for j in range(len(lines_intersection1[i])):
#             lines_dict, points_list = lines_fusion(lines_intersection1[i][j],lines_dict,points_list,1)

#     for i in range(len(lines_intersection2)):
#         for j in range(len(lines_intersection2[i])):
#             lines_dict, points_list = lines_fusion(lines_intersection2[i][j],lines_dict,points_list,2)

#     return points_list, lines_dict

#按前端展示的需求格式输出json文件
def output_json(final_data,output_path,pid):
    print("输出json文件")
    dict = {}
    #to be confirmed
    scale = 5.3
    anchor_x = -5.1
    anchor_y = -1.2
    #the data of the points
    index_point = 0
    points = []
    for i in range(len(final_data)):
        for j in range(len(final_data[i])):
            point = {}
            point['x'] = final_data[i][j][0]
            point['y'] = final_data[i][j][1]
            point['id'] = 'point_' + str(index_point)
            index_point += 1
            points += [point]

    #the data of the lines
    index_line = 0
    lines = []
    for i in range(len(final_data)):
        for j in range(len(final_data[i])):
            line = {}
            line['id'] = 'line_' + str(index_line)
            relation = []
            if(j==(len(final_data[i])-1)):
                relation.append('point_' + str(index_line))
                relation.append('point_' + str(index_line + 1 - len(final_data[i])))
            else:
                relation.append('point_' + str(index_line))
                relation.append('point_' + str(index_line + 1))
            line['points'] = relation
            index_line += 1
            lines += [line]

    dict["id"] = pid
    dict["scale"] = scale
    dict["anchor_x"] = anchor_x
    dict["anchor_y"] = anchor_y
    dict["points"] = points
    dict["lines"] = lines
    dict["lineItems"] = []
    dict["areas"] = []

    with open(output_path,'w') as dump_w:
        json.dump(dict, dump_w, cls=MyEncoder)

    print("json文件输出完成")
    return 0

def output_floorplan_image(points_list, lines_dict, file_path, minX, minZ, img_size, mapping_ratio):
    output_img = np.zeros((img_size, img_size, 3), dtype=np.uint8)

    for key in lines_dict:
        points_index = lines_dict[key]
        _, point_begin, point_end = get_lines_status(points_list, points_index)
        point1_x = int(round((point_begin[0] - minX),3) / mapping_ratio)
        point1_y = img_size - int(round((point_begin[1] - minZ),3) / mapping_ratio)
        point2_x = int(round((point_end[0] - minX), 3) / mapping_ratio)
        point2_y = img_size - int(round((point_end[1] - minZ), 3) / mapping_ratio)
        #cv2.circle(output_img, (point1_x, point1_y), 6, (0, 255, 0), -1)
        #cv2.circle(output_img, (point2_x, point2_y), 6, (0, 255, 0), -1)
        cv2.line(output_img, (point1_x, point1_y), (point2_x, point2_y), (50, 50, 204), 3)

    cv2.imwrite(file_path, output_img)
    #cv2.imshow("Result",output_img)
    #cv2.waitKey(0)

    return output_img

def output_floorplan_image(points_list, lines_dict, file_path, minX, minZ, img_w, img_h, mapping_ratio):
    output_img = np.zeros((img_w, img_h, 3), dtype=np.uint8)

    for key in lines_dict:
        points_index = lines_dict[key]
        _, point_begin, point_end = get_lines_status(points_list, points_index)
        point1_x = int(round((point_begin[0] - minX),3) / mapping_ratio)
        point1_y = img_h - int(round((point_begin[1] - minZ),3) / mapping_ratio)
        point2_x = int(round((point_end[0] - minX), 3) / mapping_ratio)
        point2_y = img_h - int(round((point_end[1] - minZ), 3) / mapping_ratio)
        #cv2.circle(output_img, (point1_x, point1_y), 6, (0, 255, 0), -1)
        #cv2.circle(output_img, (point2_x, point2_y), 6, (0, 255, 0), -1)
        cv2.line(output_img, (point1_x, point1_y), (point2_x, point2_y), (50, 50, 204), 3)

    cv2.imwrite(file_path, output_img)
    #cv2.imshow("Result",output_img)
    #cv2.waitKey(0)

    return output_img

#输出户型图和实际点云的比较图
def output_floorplan_image_compare(points_list, lines_dict, input_path, file_path, txt_path, txt_path1, minX, minZ, img_size, mapping_ratio, DEBUG_MODE):
    output_img = cv2.imread(input_path)
    floordata = []
    floordata_idx = []
    index = 0
    key_index_dict = {}
    for key in lines_dict:
        temp = []
        temp_idx = []
        points_index = lines_dict[key]
        key_index_dict[index] = key
        index += 1
        _, point_begin, point_end = get_lines_status(points_list, points_index)
        idx1, idx2 = get_segment_end_index(points_index)
        temp_idx.append(idx1)
        temp_idx.append(idx2)
        temp.append(point_begin)
        temp.append(point_end)
        floordata.append(temp)
        floordata_idx.append(temp_idx)
        if DEBUG_MODE:
            point1_x = int(round((point_begin[0] - minX),3) / mapping_ratio)
            point1_y = img_size - int(round((point_begin[1] - minZ),3) / mapping_ratio)
            point2_x = int(round((point_end[0] - minX), 3) / mapping_ratio)
            point2_y = img_size - int(round((point_end[1] - minZ), 3) / mapping_ratio)
            cv2.line(output_img, (point1_x, point1_y), (point2_x, point2_y), (0, 100, 255), 5)

    with open(txt_path,'w') as s:
        for i in range(len(floordata)):
            s.write(str(floordata[i][0][0]) + " " + str(floordata[i][0][1]) + "\n")
            s.write(str(floordata[i][1][0]) + " " + str(floordata[i][1][1]) + "\n")
        s.close()

    with open(txt_path1,'w') as s:
        s.write(str(len(points_list)) + "\n")
        for i in range(len(points_list)):
            s.write(str(points_list[i][0]) + " " + str(points_list[i][1]) + "\n")
        s.write(str(len(floordata)) + "\n")
        for i in range(len(floordata)):
            s.write(str(floordata_idx[i][0]) + " " + str(floordata_idx[i][1]) + "\n")
        s.close()

    if DEBUG_MODE:
        cv2.imwrite(file_path, output_img)

    return floordata, key_index_dict

def output_floorplan_image_compare(points_list, lines_dict, input_path, file_path, txt_path, txt_path1, minX, minZ, img_w, img_h, mapping_ratio, DEBUG_MODE):
    # output_img = cv2.imread(input_path)
    output_img = np.zeros((img_w, img_h, 3), dtype=np.uint8)
    floordata = []
    floordata_idx = []
    index = 0
    key_index_dict = {}
    for key in lines_dict:
        temp = []
        temp_idx = []
        points_index = lines_dict[key]
        key_index_dict[index] = key
        index += 1
        _, point_begin, point_end = get_lines_status(points_list, points_index)
        idx1, idx2 = get_segment_end_index(points_index)
        temp_idx.append(idx1)
        temp_idx.append(idx2)
        temp.append(point_begin)
        temp.append(point_end)
        floordata.append(temp)
        floordata_idx.append(temp_idx)
        if DEBUG_MODE:
            point1_x = int(round((point_begin[0] - minX),3) / mapping_ratio)
            point1_y = img_h - int(round((point_begin[1] - minZ),3) / mapping_ratio)
            point2_x = int(round((point_end[0] - minX), 3) / mapping_ratio)
            point2_y = img_h - int(round((point_end[1] - minZ), 3) / mapping_ratio)
            cv2.line(output_img, (point1_x, point1_y), (point2_x, point2_y), (0, 100, 255), 5)

    with open(txt_path,'w') as s:
        for i in range(len(floordata)):
            s.write(str(floordata[i][0][0]) + " " + str(floordata[i][0][1]) + "\n")
            s.write(str(floordata[i][1][0]) + " " + str(floordata[i][1][1]) + "\n")
        s.close()

    with open(txt_path1,'w') as s:
        s.write(str(len(points_list)) + "\n")
        for i in range(len(points_list)):
            s.write(str(points_list[i][0]) + " " + str(points_list[i][1]) + "\n")
        s.write(str(len(floordata)) + "\n")
        for i in range(len(floordata)):
            s.write(str(floordata_idx[i][0]) + " " + str(floordata_idx[i][1]) + "\n")
        s.close()

    if DEBUG_MODE:
        cv2.imwrite(file_path, output_img)

    return floordata, key_index_dict

def output_win_door_positon_status(winPos, doorPos, doorPos_status, doorPos_score, enter_door, path):
    with open(path,'w') as s:
        s.write(str(len(doorPos)) + "\n")
        for i in range(len(doorPos)):
            s.write(str(round(doorPos[i][0][0], 8)) + " " + str(round(doorPos[i][0][1], 8)) + \
                " " + str(round(doorPos[i][1][0], 8)) + " " + str(round(doorPos[i][1][1], 8)) + \
                    " " + str(int(doorPos_status[i])) + " " + str(float(doorPos_score[i])) + "\n")
        s.write(str(len(winPos)) + "\n")
        for i in range(len(winPos)):
            s.write(str(round(winPos[i][0][0], 8)) + " " + str(round(winPos[i][0][1], 8)) + \
                " " + str(round(winPos[i][1][0], 8)) + " " + str(round(winPos[i][1][1], 8)) + "\n")

        if len(enter_door) == 2:
            s.write("1\n")
            s.write(str(round(enter_door[0][0], 8)) + " " + str(round(enter_door[0][1], 8)) + \
            " " + str(round(enter_door[1][0], 8)) + " " + str(round(enter_door[1][1], 8)) + "\n")
        else:
            s.write("0\n")
        s.close()

def output_door_positon_status(doorPos, doorPos_score, path):
    with open(path,'w') as s:
        s.write(str(len(doorPos)) + "\n")
        for i in range(len(doorPos)):
            s.write(str(round(doorPos[i][0][0], 8)) + " " + str(round(doorPos[i][0][1], 8)) + \
                " " + str(round(doorPos[i][1][0], 8)) + " " + str(round(doorPos[i][1][1], 8)) + \
                " " + str(float(doorPos_score[i])) + "\n")
        s.close()

def output_doordata_positon_status(doorPos, path):
    with open(path,'w') as s:
        s.write(str(len(doorPos)) + "\n")
        for i in range(len(doorPos)):
            s.write(str(round(doorPos[i][0][0], 8)) + " " + str(round(doorPos[i][0][1], 8)) + \
                " " + str(round(doorPos[i][1][0], 8)) + " " + str(round(doorPos[i][1][1], 8)) + "\n")
        s.close()

# 由拓扑点线数据转换成线段数据
def floorplan_dict_to_floordata(points_list, lines_dict, txt_path, txt_path1):
# def floorplan_dict_to_floordata(points_list, lines_dict, input_path, file_path, txt_path, txt_path1, minX, minZ, ROWS_DOWN):
    # output_img = cv2.imread(input_path)
    floordata = []
    floordata_idx = []
    index = 0
    key_index_dict = {}
    for key in lines_dict:
        temp = []
        temp_idx = []
        points_index = lines_dict[key]
        key_index_dict[index] = key
        index += 1
        _, point_begin, point_end = get_lines_status(points_list, points_index)
        idx1, idx2 = get_segment_end_index(points_index)
        temp_idx.append(idx1)
        temp_idx.append(idx2)
        temp.append(point_begin)
        temp.append(point_end)
        floordata.append(temp)
        floordata_idx.append(temp_idx)
        # point1_x = COLS_LEFT + int(round((point_begin[0] - minX),3) * MAPPING_RATIO)
        # point1_y = ROWS_DOWN - int(round((point_begin[1] - minZ),3) * MAPPING_RATIO)
        # point2_x = COLS_LEFT + int(round((point_end[0] - minX), 3) * MAPPING_RATIO)
        # point2_y = ROWS_DOWN - int(round((point_end[1] - minZ), 3) * MAPPING_RATIO)

        # cv2.line(output_img, (point1_x, point1_y), (point2_x, point2_y), (0, 100, 255), 5)
    with open(txt_path,'w') as s:
        for i in range(len(floordata)):
            s.write(str(floordata[i][0][0]) + " " + str(floordata[i][0][1]) + "\n")
            s.write(str(floordata[i][1][0]) + " " + str(floordata[i][1][1]) + "\n")
        s.close()

    with open(txt_path1,'w') as s:
        s.write(str(len(points_list)) + "\n")
        for i in range(len(points_list)):
            s.write(str(points_list[i][0]) + " " + str(points_list[i][1]) + "\n")
        s.write(str(len(floordata)) + "\n")
        for i in range(len(floordata)):
            s.write(str(floordata_idx[i][0]) + " " + str(floordata_idx[i][1]) + "\n")
        s.close()

    return floordata, key_index_dict

#测试单房间输出
def single_room_img(points_list, file_path, minX, minZ, img_size, mapping_ratio):
    output_img = np.zeros((img_size, img_size, 3), dtype=np.uint8)
    for j in range(len(points_list)):
        for i in range(len(points_list[j])):
            temp = i
            next = i + 1
            if(next == len(points_list[j])):
                next = 0
            point1_x = int(round((points_list[j][temp][0] - minX), 3) / mapping_ratio)
            point1_y = img_size - int(round((points_list[j][temp][1] - minZ), 3) / mapping_ratio)
            point2_x = int(round((points_list[j][next][0] - minX), 3) / mapping_ratio)
            point2_y = img_size - int(round((points_list[j][next][1] - minZ), 3) / mapping_ratio)
            cv2.circle(output_img, (point1_x, point1_y), 5, (0, 255, 0), -1)
            cv2.circle(output_img, (point2_x, point2_y), 5, (0, 255, 0), -1)
            cv2.line(output_img, (point1_x, point1_y), (point2_x, point2_y), (255, 255, 255), 2)
        _, out_rectangle, _ = area_proportion(points_list[j])
        for i in range(len(out_rectangle)):
            temp = i
            next = i+1
            if(next == len(out_rectangle)):
                next = 0
            point1_x = int(round((out_rectangle[temp][0] - minX), 3) / mapping_ratio)
            point1_y = img_size - int(round((out_rectangle[temp][1] - minZ), 3) / mapping_ratio)
            point2_x = int(round((out_rectangle[next][0] - minX), 3) / mapping_ratio)
            point2_y = img_size - int(round((out_rectangle[next][1] - minZ), 3) / mapping_ratio)
            cv2.circle(output_img, (point1_x, point1_y), 5, (255, 0, 0), -1)
            cv2.circle(output_img, (point2_x, point2_y), 5, (255, 0, 0), -1)

    cv2.imwrite(file_path, output_img)

def single_room_img(points_list, file_path, minX, minZ, img_w, img_h, mapping_ratio):
    output_img = np.zeros((img_w, img_h, 3), dtype=np.uint8)
    for j in range(len(points_list)):
        for i in range(len(points_list[j])):
            temp = i
            next = i + 1
            if(next == len(points_list[j])):
                next = 0
            point1_x = int(round((points_list[j][temp][0] - minX), 3) / mapping_ratio)
            point1_y = img_h - int(round((points_list[j][temp][1] - minZ), 3) / mapping_ratio)
            point2_x = int(round((points_list[j][next][0] - minX), 3) / mapping_ratio)
            point2_y = img_h - int(round((points_list[j][next][1] - minZ), 3) / mapping_ratio)
            cv2.circle(output_img, (point1_x, point1_y), 5, (0, 255, 0), -1)
            cv2.circle(output_img, (point2_x, point2_y), 5, (0, 255, 0), -1)
            cv2.line(output_img, (point1_x, point1_y), (point2_x, point2_y), (255, 255, 255), 2)
        _, out_rectangle, _ = area_proportion(points_list[j])
        for i in range(len(out_rectangle)):
            temp = i
            next = i+1
            if(next == len(out_rectangle)):
                next = 0
            point1_x = int(round((out_rectangle[temp][0] - minX), 3) / mapping_ratio)
            point1_y = img_h - int(round((out_rectangle[temp][1] - minZ), 3) / mapping_ratio)
            point2_x = int(round((out_rectangle[next][0] - minX), 3) / mapping_ratio)
            point2_y = img_h - int(round((out_rectangle[next][1] - minZ), 3) / mapping_ratio)
            cv2.circle(output_img, (point1_x, point1_y), 5, (255, 0, 0), -1)
            cv2.circle(output_img, (point2_x, point2_y), 5, (255, 0, 0), -1)

    cv2.imwrite(file_path, output_img)


#测试mesh投影不正确
def single_room_img2(points_list, file_path):
    img_rows = 1000
    img_cols = 1000
    amplify_ratio = 50
    output_img = np.zeros((img_rows, img_cols, 3), dtype=np.uint8)
    minX, minY, maxX, maxY = 100, 100, -100, -100
    for i in range(len(points_list)):
        if (points_list[i][0] < minX):
            minX = points_list[i][0]
        if (points_list[i][0] > maxX):
            maxX = points_list[i][0]
        if (points_list[i][1] < minY):
            minY = points_list[i][1]
        if (points_list[i][1] > maxY):
            maxY = points_list[i][1]

    origin_x = (maxX + minX) / 2
    origin_y = (maxY + minY) / 2

    for i in range(len(points_list)):
        temp = i
        next = i + 1
        if (next == len(points_list)):
            next = 0
        point1_x = img_cols // 2 + int(round((points_list[temp][0] - origin_x), 3) * amplify_ratio)
        point1_y = img_rows // 2 - int(round((points_list[temp][1] - origin_y), 3) * amplify_ratio)
        point2_x = img_cols // 2 + int(round((points_list[next][0] - origin_x), 3) * amplify_ratio)
        point2_y = img_rows // 2 - int(round((points_list[next][1] - origin_y), 3) * amplify_ratio)
        cv2.circle(output_img, (point1_x, point1_y), 5, (0, 255, 0), -1)
        cv2.circle(output_img, (point2_x, point2_y), 5, (0, 255, 0), -1)
        cv2.line(output_img, (point1_x, point1_y), (point2_x, point2_y), (255, 255, 255), 2)
    _, out_rectangle, _ = area_proportion(points_list)
    for i in range(len(out_rectangle)):
        temp = i
        next = i + 1
        if (next == len(out_rectangle)):
            next = 0
        point1_x = img_cols // 2 + int(round((out_rectangle[temp][0] - origin_x), 3) * amplify_ratio)
        point1_y = img_rows // 2 - int(round((out_rectangle[temp][1] - origin_y), 3) * amplify_ratio)
        point2_x = img_cols // 2 + int(round((out_rectangle[next][0] - origin_x), 3) * amplify_ratio)
        point2_y = img_rows // 2 - int(round((out_rectangle[next][1] - origin_y), 3) * amplify_ratio)
        cv2.circle(output_img, (point1_x, point1_y), 5, (255, 0, 0), -1)
        cv2.circle(output_img, (point2_x, point2_y), 5, (255, 0, 0), -1)

    cv2.imwrite(file_path, output_img)

#获得各个单间的质心
def get_centroid_single_room(output_img, single_room, minX, minZ, img_size, mapping_ratio):
    for i in range(len(single_room)):
        point1_x = int(round((single_room[i][0] - minX), 3) / mapping_ratio)  # 列
        point1_y = img_size - int(round((single_room[i][1] - minZ), 3) / mapping_ratio)  # 行
        single_room[i][0] = point1_x
        single_room[i][1] = point1_y

    contour = np.zeros((len(single_room), 1, 2), dtype=np.uint)
    for i in range(len(single_room)):
        contour[i][0] = single_room[i]

    #print("contour:{}".format(contour))
    mu = cv2.moments(contour, False)

    centroid_x = int(mu["m10"] / mu["m00"])  # 列
    centroid_y = int(mu["m01"] / mu["m00"])  # 行

    cv2.circle(output_img, (centroid_x, centroid_y), 6, (0, 255, 0), -1)
    #cv2.imwrite("./test" + str(index) + ".jpg", output_img)

    res = mapping(centroid_y, centroid_x, minX, minZ, img_size)

    return res

def output_json_NewVersion(points_list, lines_dict, winPos, doorPos, enter_door, doorPos_status, single_rooms_clone, relation_dict, minX, minZ, img_size, output_path, pid, business_scenarios):
    dict = {}
    #to be confirmed
    scale = 5.3
    anchor_x = -5.1
    anchor_y = -1.2

    #the data of the points
    points = []
    for i in range(len(points_list)):
        point = {}
        point['x'] = points_list[i][0]
        point['y'] = points_list[i][1]
        point['id'] = 'point_' + str(i)
        points += [point]

    # points = []
    # lines = []
    # areas = []
    # point_index = 0
    # for i in range(len(line_adjustment)):
    #     area = {}
    #     area['id'] = 'area_' + str(i)
    #     for j in range(len(line_adjustment[i])):
    #         point = {}
    #         line = {}
    #
    #         line['id'] = 'line_' + str(point_index)
    #         point['x'] = line_adjustment[i][j][0]
    #         point['y'] = line_adjustment[i][j][1]
    #         point['id'] = 'point_' + str(point_index)
    #
    #         relation = []
    #         if(j == (len(line_adjustment[i]) - 1)):
    #             relation.append('point_' + str(point_index))
    #             relation.append('point_' + str(point_index - len(line_adjustment[i]) + 1))
    #         else:
    #             relation.append('point_' + str(point_index))
    #             relation.append('point_' + str(point_index + 1))
    #         line['points'] = relation
    #
    #         lines += [line]
    #         point_index += 1
    #         points += [point]

    #len1 = point_index
    len1 = len(points_list)
    lineItems = []
    lineItems_index = 100
    for i in range(len(winPos)):
        lineItem = {}
        line_id = "line_" + str(lineItems_index)
        lineItems_index += 1
        lineItem['id'] = line_id

        point_start = {}
        point_start['x'] = winPos[i][0][0]
        point_start['y'] = winPos[i][0][1]
        point_start['id'] = 'point_' + str(len1 + 2 * i)
        lineItem['startPointAt'] = point_start

        point_end = {}
        point_end['x'] = winPos[i][1][0]
        point_end['y'] = winPos[i][1][1]
        point_end['id'] = 'point_' + str(len1 + 2 * i + 1)
        lineItem['endPointAt'] = point_end

        lineItem['type'] = '5'  #普通窗
        lineItems += [lineItem]

    len2 = 2 * len(winPos)
    for i in range(len(doorPos)):
        lineItem = {}
        line_id = "line_" + str(lineItems_index)
        lineItems_index += 1
        lineItem['id'] = line_id

        point_start = {}
        point_start['x'] = doorPos[i][0][0]
        point_start['y'] = doorPos[i][0][1]
        point_start['id'] = 'point_' + str(len1 + len2 + 2 * i)
        lineItem['startPointAt'] = point_start

        point_end = {}
        point_end['x'] = doorPos[i][1][0]
        point_end['y'] = doorPos[i][1][1]
        point_end['id'] = 'point_' + str(len1 + len2 + 2 * i + 1)
        lineItem['endPointAt'] = point_end

        if(doorPos_status[i]):
            lineItem['type'] = '0'   #单开门
        else:
            lineItem['type'] = '16'  #垭口
        lineItems += [lineItem]

    if(len(enter_door)):
        # 加入入户门
        lineItem = {}
        lineItems_index += 1
        line_id = "line_" + str(lineItems_index)
        lineItem['id'] = line_id

        point_start = {}
        point_start['x'] = enter_door[0][0]
        point_start['y'] = enter_door[0][1]
        point_start['id'] = 'point_' + str(len1 + len2 + 2 * len(doorPos))
        lineItem['startPointAt'] = point_start

        point_end = {}
        point_end['x'] = enter_door[1][0]
        point_end['y'] = enter_door[1][1]
        point_end['id'] = 'point_' + str(len1 + len2 + 2 * len(doorPos) + 1)
        lineItem['endPointAt'] = point_end

        lineItem['type'] = '0'
        if business_scenarios in ['realsee-vr', 'nofilter']: #如视VR加入入户门字段
            print("添加入户门")
            lineItem['entrance'] = '1' # 单开门 deleted on 2020/04/20
        lineItems += [lineItem]

    #the data of the lines
    lines = []
    for line_id in lines_dict:
        line = {}
        line['id'] = line_id
        relation = []
        points_index = lines_dict[line_id]
        positioin = points_index.find('_')
        point_begin = points_index[0:positioin]
        point_end = points_index[positioin + 1:]
        relation.append('point_' + str(point_begin))
        relation.append('point_' + str(point_end))
        line['points'] = relation
        lines += [line]

    areas = []
    output_img = np.zeros((img_size, img_size, 3), dtype=np.uint8)
    for i in range(len(single_rooms_clone)):
        area = {}
        area['id'] = 'area_' + str(i)
        #centroid = get_centroid_single_room(output_img, single_rooms_clone[i], minX, minZ, img_size - 100)
        centroid = Polygon(single_rooms_clone[i]).centroid
        area['center_x'] = centroid.coords[0][0]
        area['center_y'] = centroid.coords[0][1]     #x 和 y 坐标不要搞反了
        # if(relation_dict[str(i)] == "Passway"):
        #     relation_dict[str(i)] = "other"
        # if (relation_dict[str(i)] == "Room"):
        #     relation_dict[str(i)] = "other"
        # if (relation_dict[str(i)] == "Passway"):
        #     relation_dict[str(i)] = "other"
        # if (relation_dict[str(i)] == "Dining"):
        #     relation_dict[str(i)] = "other"
        # if (relation_dict[str(i)] == "Laundry"):
        #     relation_dict[str(i)] = "other"
        area['roomName'] = relation_dict[str(i)]
        areas += [area]

    dict["id"] = pid
    dict["scale"] = scale
    dict["anchor_x"] = anchor_x
    dict["anchor_y"] = anchor_y
    dict["points"] = points
    dict["lines"] = lines
    dict["lineItems"] = lineItems
    dict["areas"] = areas

    with open(output_path,'w') as dump_w:
        json.dump(dict, dump_w, cls=MyEncoder)

    # if realsee_vr: #如视VR pipeline
    #     fp_dir = os.path.dirname(output_path)
    #     #new_fp_path = update_fp_fcuntionroom(output_path)
    #     gan_fp_path = os.path.join(fp_dir, 'gan_floorplan.json')
    #     output_path = os.path.join(fp_dir, 'display.json')
    #     new_fp_json = utils_new.update_function_name(output_path, gan_fp_path)
    #
    #     new_fp_path = os.path.join(fp_dir, 'floorplan.json')
    #     with open(new_fp_path,'w') as dump_w:
    #         json.dump(new_fp_json, dump_w, cls=MyEncoder)

    return 0

def output_json_NewVersion(points_list, lines_dict, winPos, doorPos, enter_door, doorPos_status, single_rooms_clone, relation_dict, minX, minZ, img_w, img_h, output_path, pid, business_scenarios):
    dict = {}
    #to be confirmed
    scale = 5.3
    anchor_x = -5.1
    anchor_y = -1.2

    #the data of the points
    points = []
    for i in range(len(points_list)):
        point = {}
        point['x'] = points_list[i][0]
        point['y'] = points_list[i][1]
        point['id'] = 'point_' + str(i)
        points += [point]

    # points = []
    # lines = []
    # areas = []
    # point_index = 0
    # for i in range(len(line_adjustment)):
    #     area = {}
    #     area['id'] = 'area_' + str(i)
    #     for j in range(len(line_adjustment[i])):
    #         point = {}
    #         line = {}
    #
    #         line['id'] = 'line_' + str(point_index)
    #         point['x'] = line_adjustment[i][j][0]
    #         point['y'] = line_adjustment[i][j][1]
    #         point['id'] = 'point_' + str(point_index)
    #
    #         relation = []
    #         if(j == (len(line_adjustment[i]) - 1)):
    #             relation.append('point_' + str(point_index))
    #             relation.append('point_' + str(point_index - len(line_adjustment[i]) + 1))
    #         else:
    #             relation.append('point_' + str(point_index))
    #             relation.append('point_' + str(point_index + 1))
    #         line['points'] = relation
    #
    #         lines += [line]
    #         point_index += 1
    #         points += [point]

    #len1 = point_index
    len1 = len(points_list)
    lineItems = []
    lineItems_index = 100
    for i in range(len(winPos)):
        lineItem = {}
        line_id = "line_" + str(lineItems_index)
        lineItems_index += 1
        lineItem['id'] = line_id

        point_start = {}
        point_start['x'] = winPos[i][0][0]
        point_start['y'] = winPos[i][0][1]
        point_start['id'] = 'point_' + str(len1 + 2 * i)
        lineItem['startPointAt'] = point_start

        point_end = {}
        point_end['x'] = winPos[i][1][0]
        point_end['y'] = winPos[i][1][1]
        point_end['id'] = 'point_' + str(len1 + 2 * i + 1)
        lineItem['endPointAt'] = point_end

        lineItem['type'] = '5'  #普通窗
        lineItems += [lineItem]

    len2 = 2 * len(winPos)
    for i in range(len(doorPos)):
        lineItem = {}
        line_id = "line_" + str(lineItems_index)
        lineItems_index += 1
        lineItem['id'] = line_id

        point_start = {}
        point_start['x'] = doorPos[i][0][0]
        point_start['y'] = doorPos[i][0][1]
        point_start['id'] = 'point_' + str(len1 + len2 + 2 * i)
        lineItem['startPointAt'] = point_start

        point_end = {}
        point_end['x'] = doorPos[i][1][0]
        point_end['y'] = doorPos[i][1][1]
        point_end['id'] = 'point_' + str(len1 + len2 + 2 * i + 1)
        lineItem['endPointAt'] = point_end

        if(doorPos_status[i]):
            lineItem['type'] = '0'   #单开门
        else:
            lineItem['type'] = '16'  #垭口
        lineItems += [lineItem]

    if(len(enter_door)):
        # 加入入户门
        lineItem = {}
        lineItems_index += 1
        line_id = "line_" + str(lineItems_index)
        lineItem['id'] = line_id

        point_start = {}
        point_start['x'] = enter_door[0][0]
        point_start['y'] = enter_door[0][1]
        point_start['id'] = 'point_' + str(len1 + len2 + 2 * len(doorPos))
        lineItem['startPointAt'] = point_start

        point_end = {}
        point_end['x'] = enter_door[1][0]
        point_end['y'] = enter_door[1][1]
        point_end['id'] = 'point_' + str(len1 + len2 + 2 * len(doorPos) + 1)
        lineItem['endPointAt'] = point_end

        lineItem['type'] = '0'
        if business_scenarios in ['realsee-vr', 'nofilter']: #如视VR加入入户门字段
            print("添加入户门")
            lineItem['entrance'] = '1' # 单开门 deleted on 2020/04/20
        lineItems += [lineItem]

    #the data of the lines
    lines = []
    for line_id in lines_dict:
        line = {}
        line['id'] = line_id
        relation = []
        points_index = lines_dict[line_id]
        positioin = points_index.find('_')
        point_begin = points_index[0:positioin]
        point_end = points_index[positioin + 1:]
        relation.append('point_' + str(point_begin))
        relation.append('point_' + str(point_end))
        line['points'] = relation
        lines += [line]

    areas = []
    output_img = np.zeros((img_w, img_h, 3), dtype=np.uint8)
    for i in range(len(single_rooms_clone)):
        area = {}
        area['id'] = 'area_' + str(i)
        #centroid = get_centroid_single_room(output_img, single_rooms_clone[i], minX, minZ, img_size - 100)
        centroid = Polygon(single_rooms_clone[i]).centroid
        area['center_x'] = centroid.coords[0][0]
        area['center_y'] = centroid.coords[0][1]     #x 和 y 坐标不要搞反了
        # if(relation_dict[str(i)] == "Passway"):
        #     relation_dict[str(i)] = "other"
        if (relation_dict[str(i)] == "Room"):
            relation_dict[str(i)] = "other"
        if (relation_dict[str(i)] == "Passway"):
            relation_dict[str(i)] = "other"
        if (relation_dict[str(i)] == "Dining"):
            relation_dict[str(i)] = "other"
        if (relation_dict[str(i)] == "Laundry"):
            relation_dict[str(i)] = "other"
        area['roomName'] = relation_dict[str(i)]
        areas += [area]

    dict["id"] = pid
    dict["scale"] = scale
    dict["anchor_x"] = anchor_x
    dict["anchor_y"] = anchor_y
    dict["points"] = points
    dict["lines"] = lines
    dict["lineItems"] = lineItems
    dict["areas"] = areas

    with open(output_path,'w') as dump_w:
        json.dump(dict, dump_w, cls=MyEncoder)

    # if realsee_vr: #如视VR pipeline
    #     fp_dir = os.path.dirname(output_path)
    #     #new_fp_path = update_fp_fcuntionroom(output_path)
    #     gan_fp_path = os.path.join(fp_dir, 'gan_floorplan.json')
    #     output_path = os.path.join(fp_dir, 'display.json')
    #     new_fp_json = utils_new.update_function_name(output_path, gan_fp_path)
    #
    #     new_fp_path = os.path.join(fp_dir, 'floorplan.json')
    #     with open(new_fp_path,'w') as dump_w:
    #         json.dump(new_fp_json, dump_w, cls=MyEncoder)

    return 0

#更新功能间名称等信息
def update_fp_fcuntionroom(ori_fp_path):
    fp_dir = os.path.dirname(ori_fp_path)
    fp_path = os.path.join(fp_dir, 'display.json')
    output_path = os.path.join(fp_dir, 'gan_floorplan.json')
    command = "./transfer/bin/vr_floorplan_transfer " + ori_fp_path + " " + fp_path
    os.system(command)

    #os.system('conda activate recomm')
    os.chdir('./generate_functionroom')
    command = 'python ' + 'generate_functionroom.py -s ' + fp_path + ' -d ' + output_path
    print(command)
    os.system(command)
    os.chdir('..')
    return output_path

#fp.json里加入层高等信息
def add_floorplan_info(fp_path, **kwargs):
    standard_height = 2700
    fp_height = kwargs.get('height', standard_height)
    if fp_height < 2000:
        fp_height = standard_height
    info_dict = {
                "subheight":0,
                "height":fp_height,
                "description":"",
                "ruleType":0,
                "tags":[]
                }
    with open(fp_path, 'r') as f:
        fp_json = json.load(f)
    fp_json['floorplans'][0]['info'] = info_dict
    with open(fp_path, 'w') as dump_w:
        json.dump(fp_json, dump_w, cls=MyEncoder)

#输出户型图线段分组情况可视化
def output_fusion_image(lines_status1, lines_status2, points_list, lines_dict, file_path, minX, minZ, img_size, mapping_ratio):
    output_img = np.zeros((img_size, img_size, 3), dtype=np.uint8)

    for i in range(len(lines_status1)):
        for j in range(len(lines_status1[i])):
            a = rd.randint(0, 255)
            b = rd.randint(0, 255)
            c = rd.randint(0, 255)
            for k in range(len(lines_status1[i][j])):
                points_index = lines_dict[lines_status1[i][j][k]]
                _, point_begin, point_end = get_lines_status(points_list, points_index)
                point1_x = int(round((point_begin[0] - minX), 3) / mapping_ratio)
                point1_y = img_size - int(round((point_begin[1] - minZ), 3) / mapping_ratio)
                point2_x = int(round((point_end[0] - minX), 3) / mapping_ratio)
                point2_y = img_size - int(round((point_end[1] - minZ), 3) / mapping_ratio)
                cv2.circle(output_img, (point1_x, point1_y), 5, (0, 255, 0), -1)
                cv2.circle(output_img, (point2_x, point2_y), 5, (0, 255, 0), -1)
                cv2.line(output_img, (point1_x, point1_y), (point2_x, point2_y), (a, b, c), 2)

    for i in range(len(lines_status2)):
        for j in range(len(lines_status2[i])):
            a = rd.randint(0, 255)
            b = rd.randint(0, 255)
            c = rd.randint(0, 255)
            for k in range(len(lines_status2[i][j])):
                points_index = lines_dict[lines_status2[i][j][k]]
                _, point_begin, point_end = get_lines_status(points_list, points_index)
                point1_x = int(round((point_begin[0] - minX), 3) / mapping_ratio)
                point1_y = img_size - int(round((point_begin[1] - minZ), 3) / mapping_ratio)
                point2_x = int(round((point_end[0] - minX), 3) / mapping_ratio)
                point2_y = img_size - int(round((point_end[1] - minZ), 3) / mapping_ratio)
                cv2.circle(output_img, (point1_x, point1_y), 5, (0, 255, 0), -1)
                cv2.circle(output_img, (point2_x, point2_y), 5, (0, 255, 0), -1)
                cv2.line(output_img, (point1_x, point1_y), (point2_x, point2_y), (a, b, c), 2)

    cv2.imwrite(file_path, output_img)


def output_fusion_image(lines_status1, lines_status2, points_list, lines_dict, file_path, minX, minZ, img_w, img_h, mapping_ratio):
    output_img = np.zeros((img_w, img_h, 3), dtype=np.uint8)

    for i in range(len(lines_status1)):
        for j in range(len(lines_status1[i])):
            a = rd.randint(0, 255)
            b = rd.randint(0, 255)
            c = rd.randint(0, 255)
            for k in range(len(lines_status1[i][j])):
                points_index = lines_dict[lines_status1[i][j][k]]
                _, point_begin, point_end = get_lines_status(points_list, points_index)
                point1_x = int(round((point_begin[0] - minX), 3) / mapping_ratio)
                point1_y = img_h - int(round((point_begin[1] - minZ), 3) / mapping_ratio)
                point2_x = int(round((point_end[0] - minX), 3) / mapping_ratio)
                point2_y = img_h - int(round((point_end[1] - minZ), 3) / mapping_ratio)
                cv2.circle(output_img, (point1_x, point1_y), 5, (0, 255, 0), -1)
                cv2.circle(output_img, (point2_x, point2_y), 5, (0, 255, 0), -1)
                cv2.line(output_img, (point1_x, point1_y), (point2_x, point2_y), (a, b, c), 2)

    for i in range(len(lines_status2)):
        for j in range(len(lines_status2[i])):
            a = rd.randint(0, 255)
            b = rd.randint(0, 255)
            c = rd.randint(0, 255)
            for k in range(len(lines_status2[i][j])):
                points_index = lines_dict[lines_status2[i][j][k]]
                _, point_begin, point_end = get_lines_status(points_list, points_index)
                point1_x = int(round((point_begin[0] - minX), 3) / mapping_ratio)
                point1_y = img_h - int(round((point_begin[1] - minZ), 3) / mapping_ratio)
                point2_x = int(round((point_end[0] - minX), 3) / mapping_ratio)
                point2_y = img_h - int(round((point_end[1] - minZ), 3) / mapping_ratio)
                cv2.circle(output_img, (point1_x, point1_y), 5, (0, 255, 0), -1)
                cv2.circle(output_img, (point2_x, point2_y), 5, (0, 255, 0), -1)
                cv2.line(output_img, (point1_x, point1_y), (point2_x, point2_y), (a, b, c), 2)

    cv2.imwrite(file_path, output_img)

#单间户型中是否有斜线的检查
def slope_check(inter):
    for i in range(len(inter)):
        temp = i
        next = temp + 1
        if(next == len(inter)):
            next = 0
        if(math.fabs(inter[next][0]-inter[temp][0])>0.25 or math.fabs(inter[next][1]-inter[temp][1])>0.25):
            return False

    return True


def output_singleroom(output_file, line_adjustment):
    # with open(output_file,'w') as f:
    #     for i in range(len(line_adjustment)):
    #         for j in range(len(line_adjustment[i])):
    #             f.write(str(line_adjustment[i][j][0])+" "+str(line_adjustment[i][j][1]) + " ")
    #         f.write('\n')
    #     f.close()

    # single_rooms = []
    # for i in range(len(line_adjustment)):
    #     temp = []
    #     for j in range(len(line_adjustment[i])):
    #         temp.append(line_adjustment[i][j])
    #     single_rooms.append(temp)
    #
    # single_rooms_clone = []
    # for i in range(len(line_adjustment)):
    #     temp = []
    #     for j in range(len(line_adjustment[i])):
    #         temp.append(line_adjustment[i][j])
    #     single_rooms_clone.append(temp)

    single_rooms = copy.deepcopy(line_adjustment)
    single_rooms_clone = copy.deepcopy(line_adjustment)

    return single_rooms, single_rooms_clone

#分析业务场景
def analysis_argv(argv):
    if len(argv) == 5:
        if argv[4] in ['-realseevr', '--realseevr']:
            print("如视VR业务场景")
            return 'realsee-vr'
        elif argv[4] in ['-nofilter', '--nofilter']:
            print('特殊业务场景, stage1不过滤')
            return 'nofilter'
        else:
            return 'other'
    elif len(argv) == 6:
        if argv[5] in ['-nofilter', '--nofilter'] or argv[4] in ['-nofilter', '--nofilter']:
            print('特殊业务场景, stage1不过滤')
            return 'nofilter'
        elif argv[5] in ['-realseevr', '--realseevr']:
            print("如视VR业务场景")
            return 'realsee-vr'
        else:
            return 'other'
    else:
        print("线上场景")
        return 'other'

#处理中间产物
def operate_mid_products(data_root_dir):
    # 将所有文件夹拷贝到FloorPlan_Results中
    all_input_data = os.listdir(data_root_dir)
    mid_products_dir = os.path.join(data_root_dir, 'FloorPlan_Results')
    os.makedirs(mid_products_dir, exist_ok=True)
    all_input_data_dir = os.path.join(mid_products_dir, 'all_input_data')
    os.makedirs(all_input_data_dir, exist_ok=True)
    for sub_dir in all_input_data:
        cur_dir = os.path.join(data_root_dir, sub_dir)
        os.system('cp -r ' + cur_dir + ' ' + all_input_data_dir + '/')
    print('中间产物拷贝完毕！')

    # output_dir = os.path.join(data_root_dir, 'FloorPlan_Results', 'Depth_Detection')
    # os.makedirs(output_dir, exist_ok=True)
    # print("Depth_Detection 文件夹建立")
    #
    # derived_dir = os.path.join(data_root_dir, 'derived') #derived_dir
    # all_subdirs = list(filter(lambda x: os.path.isdir(os.path.join(derived_dir, x)), os.listdir(derived_dir)))
    #
    # print(f"有{len(all_subdirs)}个子文件夹")
    # for subdir in all_subdirs:
    #     for depth_name in ['depth.png', 'depth_filter.png', 'depth_image_origin.png', 'plane_label_map.png']:
    #         os.makedirs(os.path.join(output_dir, subdir), exist_ok=True)
    #         depth_file_path = os.path.join(derived_dir, subdir, depth_name)
    #         if os.path.exists(depth_file_path):
    #             os.system('cp ' + depth_file_path + ' ' + os.path.join(output_dir, subdir, depth_name))
    #
    # os.system('cp ' + os.path.join(derived_dir, 'detection*.png') + ' ' + output_dir + '/')
    # print("深度图、物品检测图处理完毕")
    # pass
