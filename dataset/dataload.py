import copy
import cv2
import torch
import numpy as np
from PIL import Image
from util.config import config as cfg
from layers.proposal_layer import ProposalTarget
from util.misc import find_bottom, find_long_edges, split_edge_seqence, \
    norm2, vector_cos, vector_sin, split_edge_seqence_by_step, point_dist_to_line
import Polygon as plg
import pyclipper


def pil_load_img(path):
    image = Image.open(path)
    image = np.array(image)
    return image

def dist(a, b):
    return np.linalg.norm((a - b), ord=2, axis=0)

def perimeter(bbox):
    peri = 0.0
    for i in range(bbox.shape[0]):
        peri += dist(bbox[i], bbox[(i + 1) % bbox.shape[0]])
    return peri


def shrink(bboxes, rate, max_shr=20):
    rate = rate * rate
    shrinked_bboxes = []
    for bbox in bboxes:
        area = plg.Polygon(bbox).area()
        peri = perimeter(bbox)

        try:
            pco = pyclipper.PyclipperOffset()
            pco.AddPath(bbox, pyclipper.JT_ROUND, pyclipper.ET_CLOSEDPOLYGON)
            offset = min(int(area * (1 - rate) / (peri + 0.001) + 0.5), max_shr)

            shrinked_bbox = pco.Execute(-offset)
            if len(shrinked_bbox) == 0:
                shrinked_bboxes.append(bbox)
                continue

            shrinked_bbox = np.array(shrinked_bbox[0])
            if shrinked_bbox.shape[0] <= 2:
                shrinked_bboxes.append(bbox)
                continue

            shrinked_bboxes.append(shrinked_bbox)
        except Exception as e:
            print(type(shrinked_bbox), shrinked_bbox)
            print('area:', area, 'peri:', peri)
            shrinked_bboxes.append(bbox)

    return shrinked_bboxes



def shrink1(bbox, rate, max_shr=20):
    rate = rate * rate

    area = plg.Polygon(bbox).area()
    peri = perimeter(bbox)

    try:
        pco = pyclipper.PyclipperOffset()
        pco.AddPath(bbox, pyclipper.JT_ROUND, pyclipper.ET_CLOSEDPOLYGON)
        offset = min(int(area * (1 - rate) / (peri + 0.001) + 0.5), max_shr)

        shrinked_bbox = pco.Execute(-offset)
        shrinked_bbox = np.array(shrinked_bbox[0])

    except Exception as e:

        print('area:', area, 'peri:', peri)
        return bbox

    return shrinked_bbox



class TextInstance(object):
    def __init__(self, points, orient, text):
        self.orient = orient
        self.text = text
        self.bottoms = None
        self.e1 = None
        self.e2 = None

        if self.text != "#":
            self.label = 1
        else:
            self.label = -1

        self.points = np.array(points)

        """
        remove_points = []
        if len(points) > 4:
            # remove point if area is almost unchanged after removing it
            ori_area = cv2.contourArea(points)
            for p in range(len(points)):
                # attempt to remove p
                index = list(range(len(points)))
                index.remove(p)
                area = cv2.contourArea(points[index])
                if np.abs(ori_area - area)/ori_area < 0.0017 and len(points) - len(remove_points) > 4:
                    remove_points.append(p)
            self.points = np.array([point for i, point in enumerate(points) if i not in remove_points])
        else:
            self.points = np.array(points)

        """


    def find_bottom_and_sideline(self):
        #input point position
        self.bottoms = find_bottom(self.points)  # find two edges of this Text
        self.e1, self.e2 = find_long_edges(self.points, self.bottoms)  # find two long edge sequence


    def disk_cover(self, n_disk=15):
        """
        cover text region with several disks
        :param n_disk: number of disks
        :return:
        """
        inner_points1 = split_edge_seqence(self.points, self.e1, n_disk)
        inner_points2 = split_edge_seqence(self.points, self.e2, n_disk)
        inner_points2 = inner_points2[::-1]  # innverse one of long edge

        center_points = (inner_points1 + inner_points2) / 2  # disk center
        radii = norm2(inner_points1 - center_points, axis=1)  # disk radius

        return inner_points1, inner_points2, center_points, radii

    def Equal_width_bbox_cover(self, step=16.0):

        inner_points1, inner_points2 = split_edge_seqence_by_step(self.points, self.e1, self.e2, step=step)
        inner_points2 = inner_points2[::-1]  # innverse one of long edge
        center_points = (inner_points1 + inner_points2) / 2  # disk center

        #print("inner points1",inner_points1)
        #print("inner points2",inner_points2)
        #print("center_points",center_points)

        return inner_points1, inner_points2, center_points

    def __repr__(self):
        return str(self.__dict__)

    def __getitem__(self, item):
        return getattr(self, item)


class TextDataset(object):

    def __init__(self, transform, is_training=False):
        super().__init__()
        self.transform = transform
        self.is_training = is_training
        self.proposal = ProposalTarget(k_at_hop1=cfg.k_at_hop1)


    @staticmethod
    def make_text_region(img, polygons):

        tr_mask = np.zeros(img.shape[:2], np.uint8)
        train_mask = np.ones(img.shape[:2], np.uint8)
        if polygons is None:
            return tr_mask, train_mask

        for i,polygon in enumerate(polygons):
            cv2.fillPoly(tr_mask, [polygon.points.astype(np.int32)], i+1)    #make text_region
            if polygon.text == '#':
                cv2.fillPoly(train_mask, [polygon.points.astype(np.int32)], 0)

        return tr_mask, train_mask

    @staticmethod
    def fill_polygon(mask, pts, value):
        """
        fill polygon in the mask with value
        :param mask: input mask
        :param pts: polygon to draw
        :param value: fill value
        """

        cv2.fillPoly(mask, [pts.astype(np.int32)], color=(value,))


    def make_text_center_line(self, sideline1, sideline2,
                              center_line, tcl_msk1, tcl_msk2,
                              radius_map, sin_map, cos_map,value):
        expand = 0.4
        shrink = 0
        width = 1

        mask = np.zeros_like(tcl_msk1)
        # TODO: shrink 1/2 * radius at two line end
        #print("sideline1",sideline1)
        #print("sideline2",sideline2)

        p1 = np.mean(sideline1, axis=0)
        p2 = np.mean(sideline2, axis=0)

        #print("p1",p1)
        #print("p2",p2)

        vpp = vector_sin(p1 - p2)
        if vpp >= 0:
            top_line = sideline2
            bot_line = sideline1
        else:
            top_line = sideline1
            bot_line = sideline2

        if len(center_line) < 5:
            shrink = 0


        for i in range(shrink, len(center_line) - 1 - shrink):

            c1 = center_line[i]
            c2 = center_line[i + 1]
            top1 = top_line[i]
            top2 = top_line[i + 1]
            bottom1 = bot_line[i]
            bottom2 = bot_line[i + 1]

            top = (top1 + top2) / 2
            bottom = (bottom1 + bottom1) / 2

            sin_theta = vector_sin(top - bottom)
            cos_theta = vector_cos(top - bottom)

            p1 = c1 + (top1 - c1) * expand
            p2 = c1 + (bottom1 - c1) * expand
            p3 = c2 + (bottom2 - c2) * expand
            p4 = c2 + (top2 - c2) * expand
            ploy1 = np.stack([p1, p2, p3, p4])

            #print("ploy1",ploy1)

            # imshow ploy1
            """
            img=cv2.imread('/home/uircv/桌面/cv/ocr/datasets/ctw15001/train/text_image/0321.jpg')
            ploy_test=np.array(ploy1,dtype=np.int32)
            cv2.polylines(img,[ploy_test],True,color=(255,0,0),thickness=1)
            cv2.imshow("newi", img)
            cv2.waitKey(0)
            """


            self.fill_polygon(tcl_msk1, ploy1, value=value)     #make text_center line1(expend)
            self.fill_polygon(sin_map, ploy1, value=sin_theta)
            self.fill_polygon(cos_map, ploy1, value=cos_theta)

            deal_mask = mask.copy()
            self.fill_polygon(deal_mask, ploy1, value=1)

            bbox_point_cords = np.argwhere(deal_mask == 1)

            for y, x in bbox_point_cords:
                point = np.array([x, y], dtype=np.float32)
                # top   h1
                radius_map[y, x, 0] = point_dist_to_line((top1, top2), point)  # 计算point到直线的距离
                # down  h2
                radius_map[y, x, 1] = point_dist_to_line((bottom1, bottom2), point)


            pp1 = c1 + (top1 - c1) * width/norm2(top1 - c1)
            pp2 = c1 + (bottom1 - c1) * width/norm2(bottom1 - c1)
            pp3 = c2 + (bottom2 - c2) * width/norm2(top1 - c1)
            pp4 = c2 + (top2 - c2) * width/norm2(bottom2 - c2)
            poly2 = np.stack([pp1, pp2, pp3, pp4])
            #print("poly2", poly2)

            """
            # imshow ploy2
            #img=cv2.imread('/home/uircv/桌面/cv/ocr/datasets/ctw15001/train/text_image/0321.jpg')
            ploy_test=np.array(poly2,dtype=np.int32)
            cv2.polylines(img,[ploy_test],True,color=(255,255,0),thickness=1)
            cv2.imshow("newi", img)
            cv2.waitKey(0)
            """

            self.fill_polygon(tcl_msk2, poly2, value=value)     #make text_center_line2


    def make_shrink_text_center_line(self, sideline1, sideline2,
                              center_line, tcl_msk1,
                               value):
        expand = 0.6
        shrink = 2


        p1 = np.mean(sideline1, axis=0)
        p2 = np.mean(sideline2, axis=0)

        vpp = vector_sin(p1 - p2)
        if vpp >= 0:
            top_line = sideline2
            bot_line = sideline1
        else:
            top_line = sideline1
            bot_line = sideline2


        if len(center_line) < 10:
            shrink = 0

        for i in range(shrink, len(center_line) - 1 - shrink):
            c1 = center_line[i]
            c2 = center_line[i + 1]
            top1 = top_line[i]
            top2 = top_line[i + 1]
            bottom1 = bot_line[i]
            bottom2 = bot_line[i + 1]

            p1 = c1 + (top1 - c1) * expand
            p2 = c1 + (bottom1 - c1) * expand
            p3 = c2 + (bottom2 - c2) * expand
            p4 = c2 + (top2 - c2) * expand
            ploy1 = np.stack([p1, p2, p3, p4])

            self.fill_polygon(tcl_msk1, ploy1, value=value)  # make text_center line1(expend)

    @staticmethod
    def make_text_border(img, polygons):

        border_mask = np.zeros(img.shape[:2], np.uint8)
        if polygons is None:
            return border_mask

        for i,polygon in enumerate(polygons):
            view_point = polygon['points']
            #print("polygons", view_point)
            #print("length", view_point.shape)
            edge=polygon['bottoms']
            #print("edge",edge)
            e1,e2=polygon['e1'],polygon['e2']
            #print("e1,e2",e1,e2)


            pts = []
            for e in edge:
                pts1, pts2 = view_point[e[0]], view_point[e[1]]
                vector1 = view_point[e[0]] - view_point[e[0] - 1]
                # vector2=view_point[e[1]+1]-view_point[e[1]]

                try:
                    vector2 = view_point[e[1]] - view_point[e[1] + 1]
                except:
                    vector2 = view_point[e[1]] - view_point[e[1] -1]


                pts.append((pts1, vector1))
                pts.append((pts2, vector2))


            # calculate two edge length
            length_left = np.linalg.norm(pts[2][0]-pts[3][0])
            length_right = np.linalg.norm(pts[0][0]-pts[1][0])
            #print("length",length_left,length_right)

            # depict border region
            border=[]
            num=0
            for p in pts:
                num+=1
                identity=p[1]/np.linalg.norm(p[1])

                if num<=2:
                    length_value=length_right
                else:
                    length_value=length_left


                right_border=p[0]+0.2*length_value*identity
                left_border=p[0]-0.4*length_value*identity
                border.append(right_border)
                border.append(left_border)
            border=np.array(border).astype(np.int32)
            #print("border",border)

            border1=border[:4]
            border1[[2,3],:]=border1[[3,2],:]
            border2=border[4:]
            border2[[2, 3], :] = border2[[3, 2], :]

            cv2.polylines(img,[border1],isClosed=True,color=(255,0,255))
            cv2.polylines(img,[border2], isClosed=True, color=(255, 0, 255))

            cv2.fillPoly(border_mask, [border1], i + 1)  # make border_region
            cv2.fillPoly(border_mask, [border2], i + 1)  # make border_region


        #cv2.imshow("src1",img)
        #cv2.waitKey(0)

        return border_mask






    def get_training_data(self, image, polygons, image_id, image_path):
        H, W, _ = image.shape

        if self.transform:
            image, polygons = self.transform(image, copy.copy(polygons))


        tcl_mask = np.zeros((image.shape[0], image.shape[1], 2), np.uint8)
        radius_map = np.zeros((image.shape[0], image.shape[1], 2), np.float32)
        sin_map = np.zeros(image.shape[:2], np.float32)
        cos_map = np.zeros(image.shape[:2], np.float32)

        tcl_msk1 = np.zeros((image.shape[0], image.shape[1]), np.uint8)
        tcl_msk2 = np.zeros((image.shape[0], image.shape[1]), np.uint8)
        shrink_tcl_msk = np.zeros((image.shape[0], image.shape[1]), np.uint8)



        if polygons is not None:

            for i, polygon in enumerate(polygons):
                if polygon.text == '#':
                    continue

                polygon.find_bottom_and_sideline()      # find long sequence

                sideline1, sideline2, center_points = polygon.Equal_width_bbox_cover(step=4.0)  #find average points along two edge sequence
                self.make_text_center_line(sideline1, sideline2, center_points,
                                           tcl_msk1, tcl_msk2, radius_map, sin_map, cos_map,i+1)    #make textcenterline and radius sin cos map
                self.make_shrink_text_center_line(sideline1,sideline2,center_points,shrink_tcl_msk,i+1)


        tcl_mask[:, :, 0] = tcl_msk1    #center line which is expanded by 0.3
        tcl_mask[:, :, 1] = tcl_msk2    #real center line


        tr_mask, train_mask = self.make_text_region(image, polygons)    #make t_r map and train_mask
        border_mask = self.make_text_border(image, polygons)


        # clip value (0, 1) #need modify
        #tcl_mask = np.clip(tcl_mask, 0, 1)
        #tr_mask = np.clip(tr_mask, 0, 1)
        train_mask = np.clip(train_mask, 0, 1)
        tcl= np.clip(tcl_mask[:, :, 0], 0, 1)
        # border_mask=tcl * border_mask


        # # to pytorch channel sequence
        image = image.transpose(2, 0, 1)

        if not self.is_training:    #test condition
            points = np.zeros((cfg.max_annotation, cfg.max_points, 2))
            length = np.zeros(cfg.max_annotation, dtype=int)
            if polygons is not None:
                for i, polygon in enumerate(polygons):
                    pts = polygon.points
                    points[i, :pts.shape[0]] = polygon.points
                    length[i] = pts.shape[0]

            meta = {
                'image_id': image_id,
                'image_path': image_path,
                'annotation': points,
                'n_annotation': length,
                'Height': H,
                'Width': W
            }

            return image, train_mask, tr_mask, tcl_mask, radius_map, sin_map, cos_map,shrink_tcl_msk, border_mask, meta


        image = torch.from_numpy(image).float()
        train_mask = torch.from_numpy(train_mask).byte()
        #tr_mask = torch.from_numpy(tr_mask).byte().bool()
        tr_mask = torch.from_numpy(tr_mask).long()

        tcl_mask = torch.from_numpy(tcl_mask).long()
        border_mask=torch.from_numpy(border_mask).long()
        radius_map = torch.from_numpy(radius_map).float()
        sin_map = torch.from_numpy(sin_map).float()
        cos_map = torch.from_numpy(cos_map).float()
        shrink_tcl_msk=torch.from_numpy(shrink_tcl_msk).long()


        return image, train_mask, tr_mask, tcl_mask, radius_map, sin_map, cos_map,shrink_tcl_msk,border_mask


    def get_test_data(self, image, image_id, image_path):
        H, W, _ = image.shape

        if self.transform:
            image, polygons = self.transform(image)

        # to pytorch channel sequence
        image = image.transpose(2, 0, 1)

        meta = {
            'image_id': image_id,
            'image_path': image_path,
            'Height': H,
            'Width': W
        }
        return image, meta

    def __len__(self):
        raise NotImplementedError()
