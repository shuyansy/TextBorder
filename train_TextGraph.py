import os
import gc
import time
from datetime import datetime
import torch
import numpy as np
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.utils.data as data
from torch.optim import lr_scheduler
from dataset import SynthText, TotalText, Ctw1500Text, Icdar15Text, Mlt2017Text, TD500Text
from network.loss import TextLoss
from network.textnet import TextNet
from util.augmentation import Augmentation
from util.config import config as cfg, update_config, print_config
from util.misc import AverageMeter
from util.misc import mkdirs, to_device
from util.option import BaseOptions
from util.visualize import visualize_network_output
#from util.summary import LogSummary
from util.shedule import FixLR
# import multiprocessing
# multiprocessing.set_start_method("spawn", force=True)

import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import os
import torch.multiprocessing as mp


lr = None
train_step = 0


def save_model(model, epoch, lr, optimzer):

    save_dir = os.path.join(cfg.save_dir, cfg.exp_name)
    if not os.path.exists(save_dir):
        mkdirs(save_dir)

    save_path = os.path.join(save_dir, 'textgraph_{}_{}.pth'.format(model.module.backbone_name, epoch))
    print('Saving to {}.'.format(save_path))
    state_dict = {
        'lr': lr,
        'epoch': epoch,
        'model': model.state_dict() if not cfg.mgpu else model.module.state_dict(),
        'optimizer': optimzer.state_dict()
    }
    torch.save(state_dict, save_path)


def load_model(model, model_path):
    print('Loading from {}'.format(model_path))
    state_dict = torch.load(model_path)
    model.module.load_state_dict(state_dict['model'])


def train(model, train_loader, criterion, scheduler, optimizer, epoch):

    global train_step

    losses = AverageMeter()
    batch_time = AverageMeter()
    data_time = AverageMeter()
    end = time.time()
    model.train()
    # scheduler.step()

    if dist.get_rank() == 0:
        print('Epoch: {} : LR = {}'.format(epoch, scheduler.get_lr()))

    for i, (img, train_mask, tr_mask, tcl_mask, radius_map, sin_map, cos_map, kernel,border) in enumerate(train_loader):

        data_time.update(time.time() - end)

        train_step += 1

        # img, train_mask, tr_mask, tcl_mask, radius_map, sin_map, cos_map, kernel, border \
        #     = to_device(img, train_mask, tr_mask, tcl_mask, radius_map, sin_map, cos_map, kernel, border)

        img=img.to(local_rank)
        train_mask=train_mask.to(local_rank)
        tr_mask = tr_mask.to(local_rank)
        tcl_mask = tcl_mask.to(local_rank)
        radius_map = radius_map.to(local_rank)
        sin_map = sin_map.to(local_rank)
        cos_map = cos_map.to(local_rank)
        kernel = kernel.to(local_rank)
        border =border.to(local_rank)

        output = model(img) #4*12*640*640


        tr_loss, tcl_loss, sin_loss, cos_loss, radii_loss, kernel_loss, loss_agg, loss_dis, loss_re, border_loss \
            = criterion(output, train_mask, tr_mask, tcl_mask, radius_map, sin_map, cos_map,kernel,border)

        loss = tr_loss + tcl_loss + sin_loss + cos_loss + radii_loss+ kernel_loss+  loss_agg+loss_dis+loss_re+  border_loss
        # loss = tr_loss + tcl_loss + sin_loss + cos_loss + radii_loss + kernel_loss +  loss_dis + loss_re + border_loss
        #loss = tr_loss + tcl_loss + sin_loss + cos_loss + radii_loss + kernel_loss

        # backward
        try:
            optimizer.zero_grad()
            loss.backward()
        except:
            print("loss gg")
            continue

        optimizer.step()
        losses.update(loss.item())
        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        gc.collect()


        if cfg.viz and i % cfg.viz_freq == 0:
            visualize_network_output(output, tr_mask, tcl_mask[:, :, :, 0], mode='train')


        if dist.get_rank()==0 and i % cfg.display_freq == 0:
            print('({:d} / {:d})  Loss: {:.4f}  tr_loss: {:.4f}  tcl_loss: {:.4f}  '
                  'sin_loss: {:.4f}  cos_loss: {:.4f}  radii_loss: {:.4f}  kernel_loss: {:.4f} loss_agg: {:.4f} loss_dis: {:.4f} loss_re: {:.4f} border_loss: {:.4f}  '
                  .format(i, len(train_loader), loss.item(), tr_loss.item(), tcl_loss.item(),
                          sin_loss.item(), cos_loss.item(), radii_loss.item(), kernel_loss.item(), loss_agg.item(),loss_dis.item(),loss_re.item(), border_loss.item()))


    if dist.get_rank()==0 and epoch % cfg.save_freq == 0:
        save_model(model, epoch, scheduler.get_lr(), optimizer)

    print('Training Loss: {}'.format(losses.avg))
    


def main():
    global lr

    if cfg.exp_name == 'Totaltext':
        trainset = TotalText(
            data_root= '/home/uircv/桌面/cv/ocr/datasets/Totaltext',
            ignore_list=None,
            is_training=True,
            transform=Augmentation(size=cfg.input_size, mean=cfg.means, std=cfg.stds)
        )
        # valset = TotalText(
        #     data_root='data/total-text-mat',
        #     ignore_list=None,
        #     is_training=False,
        #     transform=BaseTransform(size=cfg.input_size, mean=cfg.means, std=cfg.stds)
        # )
        valset = None

    elif cfg.exp_name == 'Synthtext':
        trainset = SynthText(
            data_root='/home/uircv/桌面/cv/ocr/datasets/SynthText/SynthText',
            is_training=True,
            transform=Augmentation(size=cfg.input_size, mean=cfg.means, std=cfg.stds)
        )
        valset = None

    elif cfg.exp_name == 'Ctw1500':
        trainset = Ctw1500Text(
            data_root='/home/shuyan/ocr/datasets/ctw1500',   #dataset path
            is_training=True,
            transform=Augmentation(size=cfg.input_size, mean=cfg.means, std=cfg.stds)
        )
        valset = None

    elif cfg.exp_name == 'Icdar2015':
        trainset = Icdar15Text(
            data_root='data/Icdar2015',
            is_training=True,
            transform=Augmentation(size=cfg.input_size, mean=cfg.means, std=cfg.stds)
        )
        valset = None
    elif cfg.exp_name == 'MLT2017':
        trainset = Mlt2017Text(
            data_root='data/MLT2017',
            is_training=True,
            transform=Augmentation(size=cfg.input_size, mean=cfg.means, std=cfg.stds)
        )
        valset = None

    elif cfg.exp_name == 'TD500':
        trainset = TD500Text(
            data_root='/home/uircv/桌面/cv/ocr/datasets/TD500' ,
            is_training=True,
            transform=Augmentation(size=cfg.input_size, mean=cfg.means, std=cfg.stds)
        )
        valset = None

    else:
        print("dataset name is not correct")

    # DDP：DDP backend初始化
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend='nccl' ,init_method='env://')




    train_sampler = torch.utils.data.distributed.DistributedSampler(trainset)
    train_loader = torch.utils.data.DataLoader(trainset,
        batch_size=cfg.batch_size, num_workers=cfg.num_workers, sampler=train_sampler)

    # train_loader = data.DataLoader(trainset, batch_size=cfg.batch_size,
    #                                shuffle=True, num_workers=cfg.num_workers, pin_memory=True)

    #log_dir = os.path.join(cfg.log_dir, datetime.now().strftime('%b%d_%H-%M-%S_') + cfg.exp_name)
    #logger = LogSummary(log_dir)

    # Model
    model = TextNet(backbone=cfg.net, is_training=True)


    model = DDP(model.cuda(), device_ids=[args.local_rank], broadcast_buffers=False, find_unused_parameters=True)

    # if cfg.mgpu:
    #     print("multi-gpu1--")
    #     model = nn.DataParallel(model)

    #model = model.to(cfg.device)

    if cfg.cuda:
        cudnn.benchmark = True

    if cfg.resume:
        load_model(model, cfg.resume)

    criterion = TextLoss().to(local_rank)

    lr = cfg.lr
    moment = cfg.momentum
    if cfg.optim == "Adam" or cfg.exp_name == 'Synthtext':
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    else:
        optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=moment)

    if cfg.exp_name == 'Synthtext':
        scheduler = FixLR(optimizer)
    else:
        scheduler = lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.90)

    print('Start training TextGraph_welcomeMEddpnew::--')
    for epoch in range(cfg.start_epoch, cfg.start_epoch + cfg.max_epoch+1):
        train_loader.sampler.set_epoch(epoch)
        scheduler.step()
        train(model, train_loader, criterion, scheduler, optimizer, epoch)  #train
    print('End.')

    if torch.cuda.is_available():
        torch.cuda.empty_cache()


if __name__ == "__main__":
    np.random.seed(2019)
    torch.manual_seed(2019)

    # parse arguments
    option = BaseOptions()

    args = option.initialize()

    update_config(cfg, args)
    print_config(cfg)
    local_rank = cfg.local_rank
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    # main
    main()

