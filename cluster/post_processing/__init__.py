# -*- coding: utf-8 -*-
import os
import cv2
import torch
import time
import subprocess
import numpy as np
from utils.util import show_img, draw_bbox
import matplotlib.pyplot as plt
from util import canvas as cav
from util.misc import fill_hole

BASE_DIR = os.path.dirname(os.path.realpath(__file__))

if subprocess.call(['make', '-C', BASE_DIR]) != 0:  # return value
    raise RuntimeError('Cannot compile pse: {}'.format(BASE_DIR))


def decode(tr_mask,tcl_mask,kernel_mask,border_mask,preds):
    """
    在输出上使用sigmoid 将值转换为置信度，并使用阈值来进行文字和背景的区分
    :param preds: 网络输出
    :param scale: 网络的scale
    :param threshold: sigmoid的阈值
    :return: 最后的输出图和文本框
    """
    from .pse import pse_cpp, get_points, get_num
    from cluster.post_processing.pypse import pse_py
    preds = preds.detach().cpu().numpy()
    similarity_vectors = preds[7:11].transpose((1, 2, 0))


    # cv2.imshow("tr_mask", cav.heatmap(np.array(tr_mask * 255 / np.max(tr_mask), dtype=np.uint8)))
    # cv2.imshow("tcl_mask",
    #            cav.heatmap(np.array(tcl_mask * 255 / np.max(tcl_mask), dtype=np.uint8)))
    # cv2.imshow("kernel_mask", cav.heatmap(np.array(kernel_mask * 255 / np.max(kernel_mask), dtype=np.uint8)))
    # cv2.imshow("border_mask", cav.heatmap(np.array(border_mask * 255 / np.max(border_mask), dtype=np.uint8)))


    kernel_num, kernel_label = cv2.connectedComponents(kernel_mask.astype(np.uint8), connectivity=8)

    # n=20
    # for i in range(1,kernel_num+1):
    #     pts=np.where(kernel_label==i)
    #     if len(pts[0])<n:
    #         kernel_label[pts]=0
    # kernel_label=kernel_label>0
    # kernel_num, kernel_label = cv2.connectedComponents(kernel_label.astype(np.uint8), connectivity=8)
    # print("kernel", kernel_num)


    border_num, border_label = cv2.connectedComponents(border_mask.astype(np.uint8), connectivity=8)
    #print("border",border_num)


    print("kernel_num","border_num",kernel_num,border_num)
    #0.05
    new_border=pse_cpp(border_label, similarity_vectors, kernel_label, kernel_num,border_num, 0.05)
    new_border=new_border.reshape(border_label.shape)
    #
    # cv2.imshow("new_border", cav.heatmap(np.array(new_border * 255 / np.max(new_border), dtype=np.uint8)))
    # cv2.imshow("new_kernel", cav.heatmap(np.array(kernel_label * 255 / np.max(kernel_label), dtype=np.uint8)))
    # cv2.waitKey(0)

    value_list=[]
    w,h=new_border.shape[0],new_border.shape[1]
    for i in range(w):
        for j in range(h):
            if new_border[i][j]!=0:
                value_list.append(new_border[i][j])
    value_list=list(set(value_list))
    print("value",value_list)

    return new_border,kernel_label



