import os
import numpy as np
from dataset.data_util import pil_load_img
from dataset.dataload import TextDataset, TextInstance


class SynthText(TextDataset):

    def __init__(self, data_root, is_training=True, transform=None, ignore_list=None):
        super().__init__(transform, is_training)
        self.data_root = data_root
        self.is_training = is_training
        self.image_root = data_root
        self.annotation_root = os.path.join(data_root, 'gt')
        print("annoatation_root",self.annotation_root)

        with open(os.path.join(data_root, 'image_list.txt')) as f:
            self.annotation_list = [line.strip() for line in f.readlines()]

    @staticmethod
    def parse_txt(annotation_path):

        with open(annotation_path) as f:
            lines = [line.strip() for line in f.readlines()]
            image_id = lines[0]
            polygons = []
            for line in lines[1:]:
                points = [float(coordinate) for coordinate in line.split(',')]
                points = np.array(points, dtype=int).reshape(4, 2)
                polygon = TextInstance(points, 'c', 'abc')
                polygons.append(polygon)
        return image_id, polygons

    def __getitem__(self, item):

        # Read annotation
        annotation_id = self.annotation_list[item]
        annotation_path = os.path.join(self.annotation_root, annotation_id)

        image_id, polygons = self.parse_txt(annotation_path)

        # Read image data
        image_path = os.path.join(self.image_root, image_id)
        image = pil_load_img(image_path)

        return self.get_training_data(image, polygons, image_id=image_id, image_path=image_path)

    def __len__(self):
        return len(self.annotation_list)


if __name__ == '__main__':
    from util.augmentation import BaseTransform, Augmentation
    from util.augmentation import Augmentation
    from util.misc import regularize_sin_cos
    from nmslib import lanms
    from util.pbox import bbox_transfor_inv, minConnectPath
    from util import canvas as cav
    import time
    import cv2

    means = (0.485, 0.456, 0.406)
    stds = (0.229, 0.224, 0.225)

    transform = Augmentation(
        size=512, mean=means, std=stds
    )

    trainset = SynthText(
        data_root='/home/uircv/桌面/cv/ocr/datasets/SynthText/SynthText',
        is_training=True,
        transform=transform
    )

    # img, train_mask, tr_mask, tcl_mask, radius_map, sin_map, cos_map, meta = trainset[944]

    for idx in range(0, len(trainset)):
        print(idx)

        t0 = time.time()
        img, train_mask, tr_mask, tcl_mask, radius_map, sin_map, cos_map, kernel = trainset[idx]
        img, train_mask, tr_mask, tcl_mask, radius_map, sin_map, cos_map, kernel \
            = map(lambda x: x.cpu().numpy(), (img, train_mask, tr_mask, tcl_mask, radius_map, sin_map, cos_map, kernel))

        print("--------------------------")
        print(tr_mask.max().item())
        print(tcl_mask[:, :, 0].max().item())
        print(kernel.max().item())
        print("-----------------------")
        assert tcl_mask[:, :, 0].max().item() == kernel.max().item(), "label not match!"

        img = img.transpose(1, 2, 0)
        img = ((img * stds + means) * 255).astype(np.uint8)

        top_map = radius_map[:, :, 0]  # h1
        bot_map = radius_map[:, :, 1]  # h2

        sin_map, cos_map = regularize_sin_cos(sin_map, cos_map)
        ret, labels = cv2.connectedComponents(tcl_mask[:, :, 0].astype(np.uint8), connectivity=8)
        # print("sum",np.sum(tcl_mask[:, :, 1]))

        t0 = time.time()

        """
        for bbox_idx in range(1, ret):
            bbox_mask = labels == bbox_idx
            text_map = tcl_mask[:, :, 0] * bbox_mask

            boxes = bbox_transfor_inv(radius_map, sin_map, cos_map, text_map, wclip=(2, 8))
            # nms
            boxes = lanms.merge_quadrangle_n9(boxes.astype('float32'), 0.25)
            boxes = boxes[:, :8].reshape((-1, 4, 2)).astype(np.int32)
            if boxes.shape[0] > 1:
                center = np.mean(boxes, axis=1).astype(np.int32).tolist()
                paths, routes_path = minConnectPath(center)
                boxes = boxes[routes_path]
                top = np.mean(boxes[:, 0:2, :], axis=1).astype(np.int32).tolist()
                bot = np.mean(boxes[:, 2:4, :], axis=1).astype(np.int32).tolist()

                boundary_point = top + bot[::-1]
                # for index in routes:

                for ip, pp in enumerate(top):
                    if ip == 0:
                        color = (0, 255, 255)
                    elif ip == len(top) - 1:
                        color = (255, 255, 0)
                    else:
                        color = (0, 0, 255)
                    cv2.circle(img, (int(pp[0]), int(pp[1])), 2, color, -1)
                for ip, pp in enumerate(bot):
                    if ip == 0:
                        color = (0, 255, 255)
                    elif ip == len(top) - 1:
                        color = (255, 255, 0)
                    else:
                        color = (0, 255, 0)
                    cv2.circle(img, (int(pp[0]), int(pp[1])), 2, color, -1)
                cv2.drawContours(img, [np.array(boundary_point)], -1, (0, 255, 255), 1)

        cv2.imshow('imgs', img)
        # cv2.imshow("train_mask",
        #          cav.heatmap(np.array(train_mask * 255 / np.max(tcl_mask[:, :, 0]), dtype=np.uint8)))

        cv2.imshow("tr_mask", cav.heatmap(np.array(tr_mask * 255 / np.max(tr_mask), dtype=np.uint8)))

        # cv2.imshow("tcl_mask",
        #          cav.heatmap(np.array(tcl_mask[:, :, 1] * 255 / np.max(tcl_mask[:, :, 1]), dtype=np.uint8)))
        cv2.imshow("real_tcl_mask",
                   cav.heatmap(np.array(tcl_mask[:, :, 0] * 255 / np.max(tcl_mask[:, :, 0]), dtype=np.uint8)))

        # cv2.imshow("top_map", cav.heatmap(np.array(top_map * 255 / np.max(top_map), dtype=np.uint8)))
        # cv2.imshow("bot_map", cav.heatmap(np.array(bot_map * 255 / np.max(bot_map), dtype=np.uint8)))
        cv2.imshow("kernel", cav.heatmap(np.array(kernel * 255 / np.max(kernel), dtype=np.uint8)))
        cv2.waitKey(0)
        """