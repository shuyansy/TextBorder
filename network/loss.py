import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import itertools


class TextLoss(nn.Module):

    def __init__(self):
        super().__init__()
        self.ohem_ratio = 3
        self.delta_agg = 0.5
        self.delta_dis = 3 #3

    @staticmethod
    def ohem(predict, target, train_mask, negative_ratio=3.):

        #print(predict.shape,target.shape)

        pos = (target * train_mask).byte().bool()
        neg = ((1 - target) * train_mask).byte().bool()
        n_pos = pos.float().sum()

        #print("pos",predict[pos].shape,target[pos].shape)

        if n_pos.item() > 0:
            loss_pos = F.cross_entropy(predict[pos], target[pos], reduction='sum')
            loss_neg = F.cross_entropy(predict[neg], target[neg], reduction='none')
            n_neg = min(int(neg.float().sum().item()), int(negative_ratio * n_pos.float()))
        else:
            loss_pos = torch.tensor(0.)
            loss_neg = F.cross_entropy(predict[neg], target[neg], reduction='none')
            n_neg = 100

        loss_neg, _ = torch.topk(loss_neg, n_neg)

        return (loss_pos + loss_neg.sum()) / (n_pos + n_neg).float()

    @staticmethod
    def smooth_l1_loss(inputs, target, sigma=9.0):
        try:
            diff = torch.abs(inputs - target)
            less_one = (diff < 1.0 / sigma).float()
            loss = less_one * 0.5 * diff ** 2 * sigma \
                   + torch.abs(torch.tensor(1.0) - less_one) * (diff - 0.5 / sigma)
            loss = torch.mean(loss) if loss.numel() > 0 else torch.tensor(0.0)
        except Exception as e:
            print('RPN_REGR_Loss Exception:', e)
            loss = torch.tensor(0.0)

        return loss


    def ohem_single(self, score, gt_text, training_mask):
        pos_num = (int)(np.sum(gt_text > 0.5)) - (int)(np.sum((gt_text > 0.5) & (training_mask <= 0.5)))

        if pos_num == 0:
            # selected_mask = gt_text.copy() * 0 # may be not good
            selected_mask = training_mask
            selected_mask = selected_mask.reshape(1, selected_mask.shape[0], selected_mask.shape[1]).astype('float32')
            return selected_mask

        neg_num = (int)(np.sum(gt_text <= 0.5))
        neg_num = (int)(min(pos_num * self.ohem_ratio, neg_num))

        if neg_num == 0:
            selected_mask = training_mask
            selected_mask = selected_mask.reshape(1, selected_mask.shape[0], selected_mask.shape[1]).astype('float32')
            return selected_mask

        neg_score = score[gt_text <= 0.5]
        neg_score_sorted = np.sort(-neg_score)
        threshold = -neg_score_sorted[neg_num - 1]
        selected_mask = ((score >= threshold) | (gt_text > 0.5)) & (training_mask > 0.5)
        selected_mask = selected_mask.reshape(1, selected_mask.shape[0], selected_mask.shape[1]).astype('float32')
        return selected_mask

    def ohem_batch(self, scores, gt_texts, training_masks):
        scores = scores.data.cpu().numpy()
        gt_texts = gt_texts.data.cpu().numpy()
        training_masks = training_masks.data.cpu().numpy()

        selected_masks = []
        for i in range(scores.shape[0]):
            selected_masks.append(self.ohem_single(scores[i, :, :], gt_texts[i, :, :], training_masks[i, :, :]))

        selected_masks = np.concatenate(selected_masks, 0)
        selected_masks = torch.from_numpy(selected_masks).float()

        return selected_masks

    def dice_loss(self, input, target, mask):
        input = torch.sigmoid(input)
        target[target <= 0.5] = 0
        target[target > 0.5] = 1

        input = input.contiguous().view(input.size()[0], -1)    #bs*hw
        target = target.contiguous().view(target.size()[0], -1)
        mask = mask.contiguous().view(mask.size()[0], -1)

        input = input * mask
        target = target * mask

        a = torch.sum(input * target, 1)
        b = torch.sum(input * input, 1) + 0.001
        c = torch.sum(target * target, 1) + 0.001
        d = (2 * a) / (b + c)
        return 1 - d



    def agg_dis_loss(self, texts, kernels, gt_texts, gt_kernels, similarity_vectors):
        """
        计算 loss agg
        :param texts: 文本实例的分割结果 batch_size * (w*h)
        :param kernels: 缩小的文本实例的分割结果 batch_size * (w*h)
        :param gt_texts: 文本实例的gt batch_size * (w*h)
        :param gt_kernels: 缩小的文本实例的gt batch_size*(w*h)
        :param similarity_vectors: 相似度向量的分割结果 batch_size * 4 *(w*h)
        :return:
        """
        batch_size = texts.size()[0]
        texts = texts.contiguous().reshape(batch_size, -1)
        kernels = kernels.contiguous().reshape(batch_size, -1)
        gt_texts = gt_texts.contiguous().reshape(batch_size, -1)
        gt_kernels = gt_kernels.contiguous().reshape(batch_size, -1)
        similarity_vectors = similarity_vectors.contiguous().view(batch_size, 8, -1)

        loss_aggs = []
        loss_diss = []
        loss_reg = []
        for text_i, kernel_i, gt_text_i, gt_kernel_i, similarity_vector in zip(texts, kernels, gt_texts, gt_kernels,
                                                                               similarity_vectors):
            text_num = gt_text_i.max().item() + 1
            loss_agg_single_sample = []
            G_kernel_list = []  # 存储计算好的G_Ki,用于计算loss dis
            # 求解每一个文本实例的loss agg
            for text_idx in range(1, int(text_num)):
                # 计算 D_p_Ki
                single_kernel_mask = gt_kernel_i == text_idx

                if single_kernel_mask.sum() == 0 or (gt_text_i == text_idx).sum() == 0:
                    # 这个文本被crop掉了
                    continue

                # G_Ki, shape: 4
                G_kernel = similarity_vector[:, single_kernel_mask].mean(1)  # 4
                G_kernel_list.append(G_kernel)

                # 文本像素的矩阵 F(p) shape: 4* nums (num of text pixel)
                text_similarity_vector = similarity_vector[:, gt_text_i == text_idx]

                # ||F(p) - G(K_i)|| - delta_agg, shape: nums
                text_G_ki = (text_similarity_vector - G_kernel.reshape(8, 1)).norm(2, dim=0) - self.delta_agg

                # D(p,K_i), shape: nums
                D_text_kernel = torch.max(text_G_ki, torch.tensor(0, device=text_G_ki.device, dtype=torch.float)).pow(2)
                # 计算单个文本实例的loss, shape: nums
                loss_agg_single_text = torch.log(D_text_kernel + 1).mean()
                loss_agg_single_sample.append(loss_agg_single_text)

            if len(loss_agg_single_sample) > 0:
                loss_agg_single_sample = torch.stack(loss_agg_single_sample).mean()
            else:
                loss_agg_single_sample = torch.tensor(0, device=texts.device, dtype=torch.float)
            loss_aggs.append(loss_agg_single_sample)

            # 求解每一个文本实例的loss dis
            loss_dis_single_sample = 0
            for G_kernel_i, G_kernel_j in itertools.combinations(G_kernel_list, 2):
                # delta_dis - ||G(K_i) - G(K_j)||
                kernel_ij = self.delta_dis - (G_kernel_i - G_kernel_j).norm(2)
                # D(K_i,K_j)
                D_kernel_ij = torch.max(kernel_ij, torch.tensor(0, device=kernel_ij.device, dtype=torch.float)).pow(2)
                loss_dis_single_sample += torch.log(D_kernel_ij + 1)

            if len(G_kernel_list) > 1:
                loss_dis_single_sample /= (len(G_kernel_list) * (len(G_kernel_list) - 1))
            else:
                loss_dis_single_sample = torch.tensor(0, device=texts.device, dtype=torch.float)
            loss_diss.append(loss_dis_single_sample)

            # regulation
            loss_reg_single_sample = 0
            for G_kernel_a in G_kernel_list:
                kernel_a = G_kernel_a.norm(2)
                A_kernel_ij = torch.max(kernel_a, torch.tensor(0, device=kernel_a.device, dtype=torch.float)).pow(2)
                loss_reg_single_sample += torch.log(A_kernel_ij+1)

            if len(G_kernel_list) > 1:
                loss_reg_single_sample /= len(G_kernel_list)
            else:
                loss_reg_single_sample = torch.tensor(0, device=texts.device, dtype=torch.float)
            loss_reg.append(loss_reg_single_sample)


        return torch.stack(loss_aggs), torch.stack(loss_diss), torch.stack(loss_reg)



    def forward(self, inputs, train_mask, tr_mask, tcl_mask, radii_map, sin_map, cos_map, kernel_mask, border_mask):

        """
        calculate textsnake loss
        :param inputs: (Variable), network predict, (BS, 8, H, W)
        :param gcn_data: (Variable), (gcn_pred ,gtmat_batch)
        :param tr_mask: (Variable), TR target, (BS, H, W)
        :param tcl_mask: (Variable), TCL target, (BS, H, W)
        :param sin_map: (Variable), sin target, (BS, H, W)
        :param cos_map: (Variable), cos target, (BS, H, W)
        :param radii_map: (Variable), radius target, (BS, H, W)
        :param train_mask: (Variable), training mask, (BS, H, W)
        :return: loss_tr, loss_tcl, loss_radii, loss_sin, loss_cos
        """

        #tr_pred = inputs[:, 0].contiguous().view(-1)  # (BSxHxW, )

        tr_pred = inputs[:, 0]
        tcl_pred = inputs[:, 1]
        sin_pred = inputs[:, 2].contiguous().view(-1)  # (BSxHxW,)
        cos_pred = inputs[:, 3].contiguous().view(-1)  # (BSxHxW,)

        # regularize sin and cos: sum to 1
        scale = torch.sqrt(1.0 / (sin_pred ** 2 + cos_pred ** 2 + 0.0001))
        sin_pred = sin_pred * scale
        cos_pred = cos_pred * scale

        top_pred = inputs[:, 4].contiguous().view(-1)  # (BSxHxW,)
        bot_pred = inputs[:, 5].contiguous().view(-1)  # (BSxHxW,)
        kernel_pred=inputs[:,6]  # (BSxHxW,)
        border_pred=inputs[:,15]
        # above -> pred map


        tcl_mask = tcl_mask[:, :, :, 0]
        sin_map = sin_map.contiguous().view(-1)
        cos_map = cos_map.contiguous().view(-1)
        top_map = radii_map[:, :, :, 0].contiguous().view(-1)
        bot_map = radii_map[:, :, :, 1].contiguous().view(-1)

        # assert label corresponding
        #assert tcl_mask.max().item() == kernel_mask.max().item() == border_mask.max().item(), "label not match!"

        #calculating embedding loss
        if border_mask.max().item() == kernel_mask.max().item():
            # embedding and cluster
            similarity_vectors = inputs[:, 7:15]  # 8*4*640*640
            loss_aggs, loss_diss, loss_reg = self.agg_dis_loss(tcl_pred, kernel_pred, border_mask, kernel_mask, similarity_vectors)
            loss_agg = loss_aggs.mean()
            loss_dis = loss_diss.mean()
            loss_re = 0.001* loss_reg.mean()
            #loss_embedding = loss_agg + loss_dis + 0.001 * loss_re
        else:
            #loss_embedding=torch.tensor(0.,requires_grad=True)
            loss_agg = torch.tensor(0.,requires_grad=True)
            loss_dis = torch.tensor(0.,requires_grad=True)
            loss_re = torch.tensor(0.,requires_grad=True)


        # modify tr_loss cross_entropy loss to dice loss
        selected_masks = self.ohem_batch(tr_pred, tr_mask, train_mask)
        selected_masks = selected_masks.cuda()
        loss_tr = self.dice_loss(tr_pred, tr_mask, selected_masks)

        selected_masks1 = self.ohem_batch(border_pred, border_mask, train_mask)
        selected_masks1 = selected_masks1.cuda()
        loss_border = self.dice_loss(border_pred, border_mask, selected_masks1)



        # modify tcl_loss cross_entropy loss to dice loss
        selected_masks = ((tr_mask > 0.5) & (train_mask > 0.5)).float()
        selected_masks = selected_masks.float().cuda()

        loss_tcl = self.dice_loss(tcl_pred,tcl_mask, selected_masks)
        loss_kernel=self.dice_loss(kernel_pred,kernel_mask,selected_masks)


        #dimension transformation
        train_mask = train_mask.contiguous().view(-1)
        tcl_mask = tcl_mask.contiguous().view(-1)
        tcl_mask=tcl_mask.byte().bool()

        # geometry losses
        loss_radii = torch.tensor(0.)
        loss_sin = torch.tensor(0.)
        loss_cos = torch.tensor(0.)
        tcl_train_mask = train_mask * tcl_mask
        if tcl_train_mask.sum().item() > 0:
            ones = torch.ones_like(top_pred[tcl_mask]).float()
            loss_top = F.smooth_l1_loss(top_pred[tcl_mask] / (top_map[tcl_mask]+0.01), ones, reduction='none')
            loss_bot = F.smooth_l1_loss(bot_pred[tcl_mask] / (bot_map[tcl_mask]+0.01), ones, reduction='none')
            loss_radii = torch.mean(loss_top + loss_bot)
            loss_sin = self.smooth_l1_loss(sin_pred[tcl_mask], sin_map[tcl_mask])
            loss_cos = self.smooth_l1_loss(cos_pred[tcl_mask], cos_map[tcl_mask])

        loss_tr=loss_tr.mean()
        loss_tcl=loss_tcl.mean()
        loss_kernel=loss_kernel.mean()
        loss_border=loss_border.mean()

        return loss_tr, loss_tcl, loss_sin, loss_cos, loss_radii, loss_kernel, loss_agg, loss_dis, loss_re, loss_border

