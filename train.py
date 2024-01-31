import os
import sys
import datetime
sys.path.insert(0, '../')
sys.dont_write_bytecode = True

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from data import Data
from PDRNet import PDRNet
# from poolnet import PoolNet
import torch.cuda.amp as amp
import torchvision.utils as vutils
from torch.cuda.amp import GradScaler, autocast
import data

tmp_path ='/home/unname/ZLQ/zlq-1024/v2/tmp_see'


def bce_iou_loss(pred, mask):
    bce   = F.binary_cross_entropy_with_logits(pred, mask, reduction='mean')
    pred  = torch.sigmoid(pred)
    inter = (pred*mask).sum(dim=(2,3))
    union = (pred+mask).sum(dim=(2,3))
    iou   = 1-(inter+1)/(union-inter+1)

    return bce+iou.mean()

def train(Dataset, Network):
    ## dataset
    cfg    = Dataset.Config(datapath='/home/unname/ZLQ/zlq-1024/train_nosize_noexp/',listpath = '/home/unname/ZLQ/zlq-1024/train_pair_edge.lst',
                            savepath='/home/unname/ZLQ/zlq-1024/v2/models/15.2', mode='train', batch=8, lr=0.005, momen=0.9, decay=0.0005, epoch=65,iter_size=10,show_every =400)
    data   = Dataset.Data(cfg)
    loader = DataLoader(data, collate_fn=data.collate, batch_size=cfg.batch, shuffle=True, num_workers=0,drop_last=True)
    net    = Network(cfg)
    net.train(True)
    net.cuda()
    ## parameter
    base, head = [], []
    for name, param in net.named_parameters():
        if 'bkbone' in name:
            base.append(param)
        else:
            head.append(param)
    optimizer      = torch.optim.SGD([{'params':base}, {'params':head}], lr=cfg.lr, momentum=cfg.momen, weight_decay=cfg.decay, nesterov=True)
    scaler = GradScaler()
    global_step    = 0
    iter_num = len(loader.dataset) // cfg.batch
    for epoch in range(cfg.epoch):
        optimizer.param_groups[0]['lr'] = (1-abs((epoch+1)/(cfg.epoch+1)*2-1))*cfg.lr*0.1
        optimizer.param_groups[1]['lr'] = (1-abs((epoch+1)/(cfg.epoch+1)*2-1))*cfg.lr
        r_edge_loss, r_sal_loss, r_sum_loss = 0, 0, 0

        for step, (image, mask, edge) in enumerate(loader):

            image, sal_label, edge = image.cuda().float(), mask.cuda().float(),  edge.cuda().float()

            optimizer.zero_grad()
            with autocast():
                sal1_pred, sal2_pred, sal3_pred, sal4_pred, sal5_pred, sal6_pred, edge1_pred, edge2_pred, edge3_pred, edge4_pred, edge5_pred, edge6_pred =net(image)
                sal1_loss_fuse = bce_iou_loss(sal1_pred, sal_label)
                sal2_loss_fuse = bce_iou_loss(sal2_pred, sal_label)
                sal3_loss_fuse = bce_iou_loss(sal3_pred, sal_label)
                sal4_loss_fuse = bce_iou_loss(sal4_pred, sal_label)
                sal5_loss_fuse = bce_iou_loss(sal5_pred, sal_label)
                sal6_loss_fuse = bce_iou_loss(sal6_pred, sal_label)

                sal_loss_fuse = sal6_loss_fuse + sal5_loss_fuse + sal4_loss_fuse + sal3_loss_fuse + sal2_loss_fuse + sal1_loss_fuse 

                sal_loss = sal_loss_fuse / (cfg.iter_size * cfg.batch)
                r_sal_loss += sal_loss.data
                # print('edge1_pred.type:',edge1_pred.shape)
                edge1_loss = bce2d(edge1_pred, edge)
                edge2_loss = bce2d(edge2_pred, edge)
                edge3_loss = bce2d(edge3_pred, edge)
                edge4_loss = bce2d(edge4_pred, edge)
                edge5_loss = bce2d(edge5_pred, edge)
                edge6_loss = bce2d(edge6_pred, edge)

                edge_loss_fuse = edge1_loss + edge2_loss + edge3_loss + edge4_loss + edge5_loss + edge6_loss
                edge_loss = edge_loss_fuse / (cfg.iter_size * cfg.batch)
                r_edge_loss += edge_loss.data

                #loss = sal_loss + edge_loss
                loss = sal_loss_fuse + edge_loss_fuse
                r_sum_loss += loss.data

            scaler.scale(loss).backward()

            scaler.step(optimizer)
            scaler.update()
            global_step += 1
            if step%10 == 0:
                if step == 0:
                    x_showEvery = 1
                print('%s | epoch: [%2d/%2d], iter: [%5d/%5d]  ||  Edge : %10.4f  ||  Sal : %10.4f  ||  Sum : %10.4f' % (
                    datetime.datetime.now(), epoch, cfg.epoch, step, iter_num,
                    edge_loss_fuse,
                    sal_loss_fuse,
                    loss))
                # print('Learning rate: ' + str(cfg.lr))
                r_edge_loss, r_sal_loss, r_sum_loss= 0,0,0
                #no sigmoid ~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                sal1_pred = F.sigmoid(sal1_pred)
                edge1_pred = F.sigmoid(edge1_pred)
                vutils.save_image(sal1_pred.data, tmp_path + '/iter%d-sal-0.jpg' % step, normalize=True, padding=0)
                vutils.save_image(image.data, tmp_path + '/iter%d-sal-data.jpg' % step, padding=0)
                vutils.save_image(sal_label.data, tmp_path + '/iter%d-sal-target.jpg' % step, padding=0)
                vutils.save_image(edge.data, tmp_path + '/iter%d-edge_label.jpg' % step, padding=0)
                vutils.save_image(edge1_pred.data, tmp_path + '/iter%d-edge_pred.jpg' % step, padding=0)

        # if (epoch + 1) % 8 == 0:
        torch.save(net.state_dict(), cfg.savepath+'/epoch' + str(epoch+1) + '.pth')

def bce2d(input, target):
    assert(input.size() == target.size())
    pos = torch.eq(target, 1).float()
    neg = torch.eq(target, 0).float()

    num_pos = torch.sum(pos)
    num_neg = torch.sum(neg)
    num_total = num_pos + num_neg

    alpha = num_neg  / num_total
    beta = 1.1 * num_pos  / num_total
    # target pixel = 1 -> weight beta
    # target pixel = 0 -> weight 1-beta
    weights = alpha * pos + beta * neg
    bce_loss_edge = nn.BCEWithLogitsLoss(weight = weights, size_average=True)
    return bce_loss_edge(input, target)


if __name__=='__main__':
    train(data, PDRNet)
