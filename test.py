
import os
import sys
sys.path.insert(0, '../')
sys.dont_write_bytecode = True

import cv2
import numpy as np
import torchvision.utils as vutils
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import data
#from res_lx_2dai import resnet50
import time
from PDRNet import PDRNet

class Test(object):
    def __init__(self, Dataset, Network, path,dpath):
        ## dataset
        self.cfg    = Dataset.Config(datapath=dpath,listpath=path, snapshot='/home/unname/ZLQ/zlq-1024/v2/models/15/epoch65.pth', mode='test')
        self.data   = Dataset.Data(self.cfg)
        self.loader = DataLoader(self.data, batch_size=1, shuffle=False, num_workers=0)
        ## network
        self.net    = Network(self.cfg)
        self.net.train(False)
        self.net.cuda()
        self.net.load_state_dict(torch.load(self.cfg.snapshot))
    def save(self,dataset):
        with torch.no_grad():
            time_t = 0.0

            for image, shape, name in self.loader:
                image = image.cuda().float()
                time_start = time.time()
                res, sal2_pred, sal3_pred, sal4_pred, sal5_pred, sal6_pred, edge1_pred, sal11_pred, sal21_pred, sal31_pred, sal41_pred, sal51_pred,attentionf = self.net(image)
                print(attentionf.size())
                torch.cuda.synchronize()
                time_end = time.time()
                time_t = time_t + time_end - time_start
                # res1 = F.sigmoid(res)
                # save_path  = '/media/unname/ZZZLQ/zlq-1024/results/'+dataset+'/15(2)'
                save_path  ='/home/unname/ZLQ/zlq-1024/results/'+dataset+'/15'
                res = F.interpolate(res, shape, mode='bilinear', align_corners=True)
                # res1 = np.squeeze(res1.cpu().data.numpy())
                res = res.sigmoid().data.cpu().numpy().squeeze()
                # res = (res - res.min()) / (res.max() - res.min()+ 1e-8)
                res = (res - res.min()) / (res.max() - res.min())
                res = 255 * res
                # save_path  = '/media/unname/ZZZLQ/zlq-1024/results/'+dataset+'/15'
                if not os.path.exists(save_path):
                    os.makedirs(save_path)
                # print(save_path+'/'+name[0]+'.png')
                cv2.imwrite(save_path+'/'+name[0]+'.png', res)

            fps = len(self.loader) / time_t
            print('FPS is %f' %(fps))


if __name__=='__main__':
    test=['ECSSD','DUT-OMRON','DUTS-TE','HKU-IS','PASCAL-S']
    # test=['ECSSD']
    for dataset in test:
       for path in ['/home/unname/ALL_DATA/RGBdatasets/'+dataset+'/test.lst']:
          datapath='/home/unname/ALL_DATA/RGBdatasets/'+dataset+'/Imgs/'
          test = Test(data,PDRNet, path,datapath)
          test.save(dataset)
