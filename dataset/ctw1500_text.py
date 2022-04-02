#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import numpy as np
from dataset.data_util import pil_load_img
from dataset.dataload import TextDataset, TextInstance
from util.io import read_lines
import cv2
import mmcv

def get_ann(img, gt_path):
    h, w = img.shape[0:2]
    lines = mmcv.list_from_file(gt_path)
    bboxes = []
    words = []
    for line in lines:
        line = line.replace('\xef\xbb\xbf', '')
        gt = line.split(',')

        x1 = np.int(gt[0])
        y1 = np.int(gt[1])

        bbox = [np.int(gt[i]) for i in range(4, 32)]
        bbox = np.asarray(bbox) + ([x1 * 1.0, y1 * 1.0] * 14)
        bbox = np.asarray(bbox) / ([w * 1.0, h * 1.0] * 14)

        bboxes.append(bbox)
        words.append('???')
    return bboxes, words


class Ctw1500Text(TextDataset):

    def __init__(self, data_root, is_training=True, transform=None, ignore_list=None):
        super().__init__(transform, is_training)
        self.data_root = data_root
        self.is_training = is_training

        self.image_root = os.path.join(data_root, 'train' if is_training else 'test', "text_image")
        self.annotation_root = os.path.join(data_root, 'train' if is_training else 'test', "text_label_circum")
        self.image_list = os.listdir(self.image_root)
        self.annotation_list = ['{}'.format(img_name.replace('.jpg', '')) for img_name in self.image_list]

    @staticmethod
    def parse_carve_txt(gt_path):
        """
        .mat file parser
        :param gt_path: (str), mat file path
        :return: (list), TextInstance
        """
        lines = read_lines(gt_path + ".txt")
        polygons = []
        for line in lines:
            # line = strs.remove_all(line.strip('\ufeff'), '\xef\xbb\xbf')
            gt = list(map(int, line.split(',')))
            pts = np.stack([gt[4::2], gt[5::2]]).T.astype(np.int32)

            pts[:, 0] = pts[:, 0] + gt[0]
            pts[:, 1] = pts[:, 1] + gt[1]
            polygons.append(TextInstance(pts, 'c', "**"))

        return polygons

    def __getitem__(self, item):

        image_id = self.image_list[item]
        image_path = os.path.join(self.image_root, image_id)
        #print("image_path",image_path)

        # Read image data
        image = pil_load_img(image_path)
        try:
            h, w, c = image.shape
            assert(c == 3)
        except:
            image = cv2.imread(image_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = np.array(image)

        # Read annotation
        annotation_id = self.annotation_list[item]
        annotation_path = os.path.join(self.annotation_root, annotation_id)
        polygons = self.parse_carve_txt(annotation_path)


        return self.get_training_data(image, polygons, image_id, image_path)

    def __len__(self):
        return len(self.image_list)


if __name__ == '__main__':
    from util.augmentation import Augmentation
    from util.misc import regularize_sin_cos
    from nmslib import lanms
    from util.pbox import bbox_transfor_inv, minConnectPath
    from util import canvas as cav
    import time

    means = (0.485, 0.456, 0.406)
    stds = (0.229, 0.224, 0.225)

    transform = Augmentation(
        size=640, mean=means, std=stds
    )

    trainset = Ctw1500Text(
        data_root='/home/sy/ocr/datasets/ctw1500',
        is_training=True,
        transform=transform
    )

    for idx in range(0, len(trainset)):
        print(idx)

        t0 = time.time()
        img, train_mask, tr_mask, tcl_mask, radius_map, sin_map, cos_map,kernel, border = trainset[idx]
        img, train_mask, tr_mask, tcl_mask, radius_map, sin_map, cos_map ,kernel,border\
            = map(lambda x: x.cpu().numpy(), (img, train_mask, tr_mask, tcl_mask, radius_map, sin_map, cos_map,kernel,border))

        print("--------------------------")
        print(tr_mask.max().item())
        print(tcl_mask[:, :, 0].max().item())
        print(kernel.max().item())
        print(border.max().item())
        print("-----------------------")
        assert tcl_mask[:, :, 0].max().item() == kernel.max().item() ==border.max().item() , "label not match!"

        img = img.transpose(1, 2, 0)
        img = ((img * stds + means) * 255).astype(np.uint8)
        top_map = radius_map[:, :, 0]       #h1
        bot_map = radius_map[:, :, 1]       #h2
        sin_map, cos_map = regularize_sin_cos(sin_map, cos_map)

        tcl=np.clip(tcl_mask[:, :, 0],0,1)

        # new_area=tcl_mask[:, :, 0] * kernel
        # new_area1 = tcl_mask[:, :, 0] * border

        new_area=tcl * kernel
        new_area1 = tcl * border


        pure_center_area=np.zeros(img.shape[:2], np.int32)
        pure_border_area = np.zeros(img.shape[:2], np.int32)
        for i in range(1,tcl_mask[:, :, 0].max().item()+1):     # traverse every instance
            mask=kernel==i
            pure_center_area=mask * new_area       # final tcl_mask
            boxes1 = bbox_transfor_inv(radius_map, sin_map, cos_map, pure_center_area, wclip=(2, 8))

            mask = border == i
            pure_border_area = mask * new_area1  # final tcl_mask
            boxes2 = bbox_transfor_inv(radius_map, sin_map, cos_map, pure_border_area, wclip=(2, 8))

            # nms
            boxes1 = lanms.merge_quadrangle_n9(boxes1.astype('float32'), 0.25)  # kernel box
            boxes2 = lanms.merge_quadrangle_n9(boxes2.astype('float32'), 0.7)   # border box

            if boxes1.shape[0] and boxes2.shape[0]>0:
                boxes1 = boxes1[:, :8].reshape((-1, 4, 2)).astype(np.int32)
                boxes2 = boxes2[:, :8].reshape((-1, 4, 2)).astype(np.int32)
                boxes=np.concatenate((boxes1,boxes2))

                if boxes.shape[0] > 1:
                    center = np.mean(boxes, axis=1).astype(np.int32).tolist()
                    paths, routes_path = minConnectPath(center)
                    boxes = boxes[routes_path]
                    top = np.mean(boxes[:, 0:2, :], axis=1).astype(np.int32).tolist()
                    bot = np.mean(boxes[:, 2:4, :], axis=1).astype(np.int32).tolist()
                    boundary_point = top + bot[::-1]
                    # for index in routes:
                    cv2.drawContours(img, [np.array(boundary_point)], -1, (0, 255, 255), 1)
        cv2.imwrite('/home/sy/ocr/TextBorder/gt_vis/'+str(idx)+".jpg",img )


        cv2.imshow('imgs', img)
        #cv2.imshow("train_mask",
         #          cav.heatmap(np.array(train_mask * 255 / np.max(tcl_mask[:, :, 0]), dtype=np.uint8)))

        cv2.imshow("tr_mask", cav.heatmap(np.array(tr_mask * 255 / np.max(tr_mask), dtype=np.uint8)))

        cv2.imshow("tcl_mask",
                  cav.heatmap(np.array(tcl_mask[:, :, 0] * 255 / np.max(tcl_mask[:, :, 0]), dtype=np.uint8)))
        #cv2.imshow("real_tcl_mask",
                   #cav.heatmap(np.array(tcl_mask[:, :, 0] * 255 / np.max(tcl_mask[:, :, 0]), dtype=np.uint8)))

        #cv2.imshow("top_map", cav.heatmap(np.array(top_map * 255 / np.max(top_map), dtype=np.uint8)))
        #cv2.imshow("bot_map", cav.heatmap(np.array(bot_map * 255 / np.max(bot_map), dtype=np.uint8)))
        cv2.imshow("kernel", cav.heatmap(np.array(kernel * 255 / np.max(kernel), dtype=np.uint8)))
        cv2.imshow("border", cav.heatmap(np.array(border * 255 / np.max(border), dtype=np.uint8)))
        cv2.imshow("overlapping", cav.heatmap(np.array(new_area1 * 255 / np.max(new_area1), dtype=np.uint8)))
        cv2.waitKey(0)

