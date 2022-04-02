import numpy as np
import cv2
from util.config import config as cfg
from util.misc import fill_hole, regularize_sin_cos
from util.misc import norm2,get_index
from nmslib import lanms
from util.pbox import minConnectPath
import torch
#from cluster.post_processing import decode


class TextDetector(object):

    def __init__(self, model):
        self.model = model
        self.tr_thresh = cfg.tr_thresh
        self.tcl_thresh = cfg.tcl_thresh
        self.expend = cfg.expend + 1.0

        # evaluation mode
        model.eval()


    @staticmethod
    def in_contour(cont, point):
        """
        utility function for judging whether `point` is in the `contour`
        :param cont: cv2.findCountour result
        :param point: 2d coordinate (x, y)
        :return:
        """
        x, y = point
        return cv2.pointPolygonTest(cont, (x, y), False) > 0

    def select_edge(self, cont, box):
        cont = np.array(cont)
        box = box.astype(np.int32)
        c1 = np.array(0.5 * (box[0, :] + box[3, :]), dtype=np.int)
        c2 = np.array(0.5 * (box[1, :] + box[2, :]), dtype=np.int)

        if not self.in_contour(cont, c1):
            return [box[0, :].tolist(), box[3, :].tolist()]
        elif not self.in_contour(cont, c2):
            return [box[1, :].tolist(), box[2, :].tolist()]
        else:
            return None

    def bbox_transfor_inv(self, radius_map, sin_map, cos_map, score_map, wclip=(2, 8)):
        xy_text = np.argwhere(score_map > 0)
        # sort the text boxes via the y axis
        xy_text = xy_text[np.argsort(xy_text[:, 0])]
        origin = xy_text
        radius = radius_map[xy_text[:, 0], xy_text[:, 1], :]
        sin = sin_map[xy_text[:, 0], xy_text[:, 1]]
        #print("sin_map",sin_map.shape)
        #print("sin",sin.shape)
        cos = cos_map[xy_text[:, 0], xy_text[:, 1]]
        #print("cos_map",cos_map.shape)
        #print("cos",cos.shape)
        dtx = radius[:, 0] * cos * self.expend
        dty = radius[:, 0] * sin * self.expend
        ddx = radius[:, 1] * cos * self.expend
        ddy = radius[:, 1] * sin * self.expend
        topp = origin + np.stack([dty, dtx], axis=-1)
        botp = origin - np.stack([ddy, ddx], axis=-1)
        width = (radius[:, 0] + radius[:, 1]) // 3
        width = np.clip(width, wclip[0], wclip[1])

        top1 = topp - np.stack([width * cos, -width * sin], axis=-1)
        top2 = topp + np.stack([width * cos, -width * sin], axis=-1)
        bot1 = botp - np.stack([width * cos, -width * sin], axis=-1)
        bot2 = botp + np.stack([width * cos, -width * sin], axis=-1)

        bbox = np.stack([top1, top2, bot2, bot1], axis=1)[:, :, ::-1]
        bboxs = np.zeros((bbox.shape[0], 9), dtype=np.float32)
        bboxs[:, :8] = bbox.reshape((-1, 8))
        bboxs[:, 8] = score_map[xy_text[:, 0], xy_text[:, 1]]
        return bboxs    #p*9 e.p 1414*9


    def detect_contours(self, tr_pred, tcl_pred, sin_pred, cos_pred, radii_pred,test_image):

        # thresholding
        tr_pred_mask = tr_pred > self.tr_thresh
        tcl_pred_mask = tcl_pred > self.tcl_thresh

        # multiply TR and TCL
        tcl_mask = tcl_pred_mask * tr_pred_mask
        #tcl_mask=tcl_pred_mask

        # regularize
        sin_pred, cos_pred = regularize_sin_cos(sin_pred, cos_pred)
        # find disjoint regions
        tcl_mask = fill_hole(tcl_mask)
        tcl_contours, _ = cv2.findContours(tcl_mask.astype(np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        mask = np.zeros_like(tcl_mask)
        bbox_contours = list()
        for cont in tcl_contours:       #experience each text instance
            deal_map = mask.copy()
            cv2.drawContours(deal_map, [cont], -1, 1, -1)
            if deal_map.sum() <= 100:
                continue
            text_map = tr_pred * deal_map
            bboxs = self.bbox_transfor_inv(radii_pred, sin_pred, cos_pred, text_map, wclip=(4, 12))

            # nms
            boxes = lanms.merge_quadrangle_n9(bboxs.astype('float32'), 0.25)
            boxes = boxes[:, :8].reshape((-1, 4, 2)).astype(np.int32)
            #print(boxes.shape)  #n*4*2
            #print("bboxes", boxes)

            """
            # show text components
            test_image = cv2.UMat(test_image).get()
            cv2.imshow("test_image", test_image)

            for b in boxes:
                cv2.polylines(test_image, [b], True, (255, 0, 0), 2)
                cv2.imshow("test_image", test_image)
                cv2.waitKey(0)
            """

            boundary_point = None
            if boxes.shape[0] > 1:
                center = np.mean(boxes, axis=1).astype(np.int32).tolist()
                #print("center",len(center), center)
                paths, routes_path = minConnectPath(center)
                #print("route_path",routes_path)
                boxes = boxes[routes_path]
                #print("new_boxes",boxes)

                """
                test_image = cv2.UMat(test_image).get()
                cv2.imshow("test_image", test_image)
                for b in boxes:
                    cv2.polylines(test_image, [b], True, (255, 0, 0), 2)
                    cv2.imshow("test_image", test_image)
                    cv2.waitKey(0)
                """

                top = np.mean(boxes[:, 0:2, :], axis=1).astype(np.int32).tolist()
                bot = np.mean(boxes[:, 2:4, :], axis=1).astype(np.int32).tolist()
                edge0 = self.select_edge(top + bot[::-1], boxes[0])
                edge1 = self.select_edge(top + bot[::-1], boxes[-1])
                if edge0 is not None:
                    top.insert(0, edge0[0])
                    bot.insert(0, edge0[1])
                if edge1 is not None:
                    top.append(edge1[0])
                    bot.append(edge1[1])
                boundary_point = np.array(top + bot[::-1])

            elif boxes.shape[0] == 1:
                top = boxes[0, 0:2, :].astype(np.int32).tolist()
                bot = boxes[0, 2:4:-1, :].astype(np.int32).tolist()
                boundary_point = np.array(top + bot)

            if boundary_point is None:
                continue
            reconstruct_mask = mask.copy()
            cv2.drawContours(reconstruct_mask, [boundary_point], -1, 1, -1)
            if (reconstruct_mask * tr_pred_mask).sum() < reconstruct_mask.sum() * 0.5:
                print("continue")
                continue
            # if reconstruct_mask.sum() < 200:
            #     continue

            rect = cv2.minAreaRect(boundary_point)
            if min(rect[1][0], rect[1][1]) < 10 or rect[1][0] * rect[1][1] < 300:
                continue

            bbox_contours.append([boundary_point, np.array(np.stack([top, bot], axis=1))])

        return bbox_contours


    def detect_contours1(self, tr_pred, tcl_pred, sin_pred, cos_pred, radii_pred,test_image):

        # thresholding

        tr_pred_mask = tr_pred > self.tr_thresh

        tcl_mask=tcl_pred

        # regularize
        sin_pred, cos_pred = regularize_sin_cos(sin_pred, cos_pred)
        # find disjoint regions
        tcl_mask = fill_hole(tcl_mask)
        tcl_contours, _ = cv2.findContours(tcl_mask.astype(np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        mask = np.zeros_like(tcl_mask)
        bbox_contours = list()
        all_bboxes=[]
        all_tag=[]
        for cont in tcl_contours:       #experience each text instance
            deal_map = mask.copy()
            cv2.drawContours(deal_map, [cont], -1, 1, -1)
            if deal_map.sum() <= 100:
                continue

            text_map = tcl_pred * deal_map
            text_map1=tr_pred * deal_map

            """
            tag=[]
            h, w = text_map.shape
            for i in range(h):
                for j in range(w):
                    if text_map[i][j] != 0:
                        tag.append(text_map[i][j])

            tag = list(set(tag))
            all_tag.append(tag[0])
            """

            all_tag.append(text_map.max())

            bboxs = self.bbox_transfor_inv(radii_pred, sin_pred, cos_pred, text_map1, wclip=(4, 12)) #4,12

            # nms
            boxes = lanms.merge_quadrangle_n9(bboxs.astype('float32'), 0.25)
            boxes = boxes[:, :8].reshape((-1, 4, 2)).astype(np.int32)
            all_bboxes.append(boxes)


        #print("all_tag",all_tag)
        #print(len(all_bboxes))
        assert len(all_bboxes) == len(all_tag), "length not match!"

        #for i in range(len(all_bboxes)):
            #print(all_bboxes[i].shape)

        # combine boxes
        new_boxes=[]
        remove_all_tag=list(set(all_tag))
        for i in range(len(remove_all_tag)):
            result=get_index(all_tag,remove_all_tag[i])
            if len(result) == 1:
                new_boxes.append(all_bboxes[result[0]])
            else:
                temp=[]
                for j in result:
                    temp.append(all_bboxes[j])
                temp=np.array(temp)
                temp=np.concatenate(temp)
                new_boxes.append(temp)


        #print("******")
        for boxes in new_boxes:
            #print(boxes.shape)

            """
            # show text components
            test_image = cv2.UMat(test_image).get()
            cv2.imshow("test_image", test_image)
            for b in boxes:
                cv2.polylines(test_image, [b], True, (255, 0, 0), 2)
                cv2.imshow("test_image", test_image)
                cv2.waitKey(0)
            """



            boundary_point = None
            if boxes.shape[0] > 1:
                center = np.mean(boxes, axis=1).astype(np.int32).tolist()
                paths, routes_path = minConnectPath(center)
                boxes = boxes[routes_path]

                top = np.mean(boxes[:, 0:2, :], axis=1).astype(np.int32).tolist()
                bot = np.mean(boxes[:, 2:4, :], axis=1).astype(np.int32).tolist()
                edge0 = self.select_edge(top + bot[::-1], boxes[0])
                edge1 = self.select_edge(top + bot[::-1], boxes[-1])
                if edge0 is not None:
                    top.insert(0, edge0[0])
                    bot.insert(0, edge0[1])
                if edge1 is not None:
                    top.append(edge1[0])
                    bot.append(edge1[1])
                boundary_point = np.array(top + bot[::-1])

            elif boxes.shape[0] == 1:
                top = boxes[0, 0:2, :].astype(np.int32).tolist()
                bot = boxes[0, 2:4:-1, :].astype(np.int32).tolist()
                boundary_point = np.array(top + bot)

            if boundary_point is None:
                continue
            reconstruct_mask = mask.copy()
            cv2.drawContours(reconstruct_mask, [boundary_point], -1, 1, -1)
            if (reconstruct_mask * tr_pred_mask).sum() < reconstruct_mask.sum() * 0.5:
                continue
            #if reconstruct_mask.sum() < 200:
            #     continue

            rect = cv2.minAreaRect(boundary_point)

            if min(rect[1][0], rect[1][1]) < 5 or rect[1][0] * rect[1][1] < 300:
                continue

            bbox_contours.append([boundary_point, np.array(np.stack([top, bot], axis=1))])

        return bbox_contours


    def detect(self, image):
        # print("image",image.shape)  #1*3*512*512
        test_image = image.squeeze(0)  # 3*512*512
        test_image = test_image.cpu().numpy()
        test_image = test_image.transpose(1, 2, 0)
        means = (0.485, 0.456, 0.406)
        stds = (0.229, 0.224, 0.225)
        test_image = ((test_image * stds + means) * 255).astype(np.uint8)
        # cv2.imshow("test_image",test_image)
        # cv2.waitKey(0)

        # get model output
        with torch.no_grad():
            output = self.model(image)

        image = image[0].data.cpu().numpy()

        tr_pred = torch.sigmoid(output[0, 0, :, :]).data.cpu().numpy()
        tcl_pred = torch.sigmoid(output[0, 1, :, :]).data.cpu().numpy()


        sin_pred = output[0, 2].data.cpu().numpy()  # 512*512
        cos_pred = output[0, 3].data.cpu().numpy()  # 512*512
        radii_pred = output[0, 4:6].permute(1, 2, 0).contiguous().data.cpu().numpy()  # 512*512*2
        kernel_pred= torch.sigmoid(output[0,6, :, :]).data.cpu().numpy()
        # find text contours
        contours = self.detect_contours(tr_pred, tcl_pred, sin_pred, cos_pred, radii_pred, test_image)  # (n_tcl, 3)
        # contours = self.adjust_contours(img_show, contours)

        output = {
            'image': image,
            'tr': tr_pred,
            'tcl': tcl_pred,
            'sin': sin_pred,
            'cos': cos_pred,
            'radii': radii_pred,
            'kernel':kernel_pred
        }


        return contours, output


    def detect1(self, image):

        test_image = image.squeeze(0)  # 3*512*512
        test_image = test_image.cpu().numpy()
        test_image = test_image.transpose(1, 2, 0)
        means = (0.485, 0.456, 0.406)
        stds = (0.229, 0.224, 0.225)
        test_image = ((test_image * stds + means) * 255).astype(np.uint8)

        with torch.no_grad():
            # get model output
            output = self.model(image)


        image = image[0].data.cpu().numpy()
        tr_pred = torch.sigmoid(output[0, 0, :, :]).data.cpu().numpy()
        tcl_pred = torch.sigmoid(output[0, 1, :, :]).data.cpu().numpy()
        sin_pred = output[0, 2].data.cpu().numpy()  # 512*512
        cos_pred = output[0, 3].data.cpu().numpy()  # 512*512
        radii_pred = output[0, 4:6].permute(1, 2, 0).contiguous().data.cpu().numpy()  # 512*512*2
        kernel_pred = torch.sigmoid(output[0, 6, :, :]).data.cpu().numpy()
        border_pred = torch.sigmoid(output[0, 11, :, :]).data.cpu().numpy()

        tr_mask = (tr_pred > cfg.tr).astype(np.uint8)
        tcl_mask = (tcl_pred > cfg.tcl).astype(np.uint8)
        kernel_mask = (kernel_pred > cfg.kernel).astype(np.uint8)
        border_mask= (border_pred > cfg.border).astype(np.uint8)

        tcl_mask = tcl_mask * tr_mask
        kernel_mask = kernel_mask * tr_mask
        border_mask = border_mask * tcl_mask

        new_border, new_kernel = decode(tr_mask,tcl_mask,kernel_mask,border_mask,output[0])
        #print("*********** ",new_border.max(),new_kernel.max())


        # find text contours
        contours = self.detect_contours2(tr_pred, tr_mask, tcl_mask,new_kernel,new_border, sin_pred, cos_pred, radii_pred, test_image)  # (n_tcl, 3)


        output = {
            'image': image,
            'tr': tr_pred,
            'tcl': tcl_pred,
            'sin': sin_pred,
            'cos': cos_pred,
            'radii': radii_pred,
            'kernel': kernel_pred,
            'border':border_pred
        }
        return contours, output


    def detect_contours2(self,tr_pred, tr_mask, tcl_mask, kernel_mask, border_mask, sin_pred, cos_pred, radii_pred,test_image):
        test_image1 = test_image.copy()
        # regularize
        sin_pred, cos_pred = regularize_sin_cos(sin_pred, cos_pred)

        new_area = tcl_mask * kernel_mask
        new_area1 = tcl_mask * border_mask


        bbox_contours = list()
        for i in range(1,kernel_mask.max().item()+1):     # traverse every instance
            mask=kernel_mask==i
            pure_center_area=mask * new_area       # final pure_kernel
            boxes1 = self.bbox_transfor_inv(radii_pred, sin_pred, cos_pred, pure_center_area, wclip=(2, 8))

            mask = border_mask == i
            pure_border_area = mask * new_area1  # final pure_border
            boxes2 = self.bbox_transfor_inv(radii_pred, sin_pred, cos_pred, pure_border_area, wclip=(2, 8))

            # nms
            boxes1 = lanms.merge_quadrangle_n9(boxes1.astype('float32'), 0.25)  # kernel box
            boxes2 = lanms.merge_quadrangle_n9(boxes2.astype('float32'), 0.7)   # border box

            boxes=None
            if boxes1.shape[0]==0:
                print("no_kernel")
                continue

            if boxes1.shape[0]>0:
                boxes1 = boxes1[:, :8].reshape((-1, 4, 2)).astype(np.int32)
                boxes = boxes1

            if boxes2.shape[0]>0:
                boxes2 = boxes2[:, :8].reshape((-1, 4, 2)).astype(np.int32)
                boxes=np.concatenate((boxes1,boxes2))

            cv2.drawContours(test_image1,boxes,-1, (0, 255, 255), 1)
            # cv2.imshow("src",test_image1)
            # cv2.waitKey(0)


            boundary_point = None
            if boxes.shape[0] > 1:
                center = np.mean(boxes, axis=1).astype(np.int32).tolist()
                paths, routes_path = minConnectPath(center)
                boxes = boxes[routes_path]
                top = np.mean(boxes[:, 0:2, :], axis=1).astype(np.int32).tolist()
                bot = np.mean(boxes[:, 2:4, :], axis=1).astype(np.int32).tolist()
                edge0 = self.select_edge(top + bot[::-1], boxes[0])
                edge1 = self.select_edge(top + bot[::-1], boxes[-1])
                if edge0 is not None:
                    top.insert(0, edge0[0])
                    bot.insert(0, edge0[1])
                if edge1 is not None:
                    top.append(edge1[0])
                    bot.append(edge1[1])
                boundary_point = np.array(top + bot[::-1])


            elif boxes.shape[0] == 1:
                top = boxes[0, 0:2, :].astype(np.int32).tolist()
                bot = boxes[0, 2:4:-1, :].astype(np.int32).tolist()
                boundary_point = np.array(top + bot)

            if boundary_point is None:
                continue

            cv2.drawContours(test_image1, [np.array(boundary_point)], -1, (0, 255, 255), 1)


            reconstruct_mask = mask.copy()
            if (reconstruct_mask * tr_mask).sum() < reconstruct_mask.sum() * 0.5:
                print("continue")
                continue

            rect = cv2.minAreaRect(boundary_point)
            if min(rect[1][0], rect[1][1]) < 10 or rect[1][0] * rect[1][1] < 300:
                continue

            bbox_contours.append([boundary_point, np.array(np.stack([top, bot], axis=1))])

        #cv2.imshow("src", test_image1)
        #cv2.waitKey(0)

        return bbox_contours






















