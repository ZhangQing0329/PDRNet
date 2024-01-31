import os
import cv2
import torch
import numpy as np
from torch.utils.data import Dataset
mmm = 0
########################### Data Augmentation ###########################
class Normalize(object):
    def __init__(self, mean, std):
        self.mean = mean 
        self.std  = std
    
    def __call__(self, image, mask=None, edge=None):
        image = (image - self.mean)/self.std
        if mask is None:
            return image
        else:
            mask /= 255
            edge /=255
            edge[np.where(edge > 0.5)] = 1.

            return image, mask, edge

class RandomCrop(object):
    def __call__(self, image, mask,edge):
        H,W,_   = image.shape
        randw   = np.random.randint(W/8)
        randh   = np.random.randint(H/8)
        offseth = 0 if randh == 0 else np.random.randint(randh)
        offsetw = 0 if randw == 0 else np.random.randint(randw)
        p0, p1, p2, p3 = offseth, H+offseth-randh, offsetw, W+offsetw-randw
        return image[p0:p1,p2:p3, :], mask[p0:p1,p2:p3], edge[p0:p1,p2:p3]

class RandomFlip(object):
    def __call__(self, image, mask, edge):
        if np.random.randint(2)==0:
            return image[:,::-1,:], mask[:, ::-1], edge[:, ::-1]
        else:
            return image, mask, edge

class Resize(object):
    def __init__(self, H, W):
        self.H = H
        self.W = W

    def __call__(self, image, mask=None):
        image = cv2.resize(image, dsize=(self.W, self.H), interpolation=cv2.INTER_LINEAR)
        if mask is None:
            return image
        else:
            mask  = cv2.resize( mask, dsize=(self.W, self.H), interpolation=cv2.INTER_LINEAR)
            return image, mask


class ToTensor(object):
    def __call__(self, image, mask=None,edge=None):
        image = torch.from_numpy(image)
        image = image.permute(2, 0, 1)
        if mask is None:
            return image
        else:
            mask  = torch.from_numpy(mask)
            edge  = torch.from_numpy(edge)

            return image, mask, edge


########################### Config File ###########################
class Config(object):
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.mean   = np.array([[[124.55, 118.90, 102.94]]])
        self.std    = np.array([[[ 56.77,  55.97,  57.50]]])
        print('\nParameters...')
        for k, v in self.kwargs.items():
            print('%-10s: %s'%(k, v))

    def __getattr__(self, name):
        if name in self.kwargs:
            return self.kwargs[name]
        else:
            return None


########################### Dataset Class ###########################
class Data(Dataset):
    def __init__(self, cfg):
        self.cfg        = cfg
        self.normalize  = Normalize(mean=cfg.mean, std=cfg.std)
        self.randomcrop = RandomCrop()
        self.randomflip = RandomFlip()
        self.resize     = Resize(320, 320)
        self.totensor   = ToTensor()
        self.image_list = self.cfg.listpath

        with open(self.image_list, 'r') as f:
            self.sal_list = [x.strip() for x in f.readlines()]

        self.images, self.mask, self.edge = [], [], []
        for i in range(len(self.sal_list)):
            self.images.append(self.cfg.datapath+self.sal_list[i].split()[0])
            if self.cfg.mode == 'train':
                self.mask.append(self.cfg.datapath+self.sal_list[i].split()[1])
                self.edge.append(self.cfg.datapath+self.sal_list[i].split()[2])


    def __getitem__(self, idx):
        image_name  = self.images[idx]
        name = image_name.split('/')[-1]
        # print("image_name:", image_name[-8:-4])
        image = cv2.imread(image_name)[:,:,::-1].astype(np.float32)
        # image = cv2.imread(image_name).astype(np.float32)

        shape = image.shape[:2]

        if self.cfg.mode=='train':
            mask_name  = self.mask[idx]
            #print("mask_name:",mask_name)
            edge_name  = self.edge[idx]
            #print("edge_name:", edge_name)
            mask = cv2.imread(mask_name, 0).astype(np.float32)
            edge = cv2.imread(edge_name, 0).astype(np.float32)

            image, mask, edge = self.normalize(image, mask, edge)
            image, mask, edge = self.randomcrop(image, mask, edge)
            image, mask, edge = self.randomflip(image, mask, edge)
            return image, mask, edge
        else:
            image = self.normalize(image)
            image = self.resize(image)
            image = self.totensor(image)

            return image, shape, name[:-4]

    def collate(self, batch):
        size = [224, 256, 288, 320, 352][np.random.randint(0, 5)]
        image, mask, edge = [list(item) for item in zip(*batch)]
        for i in range(len(batch)):
            image[i] = cv2.resize(image[i], dsize=(size, size), interpolation=cv2.INTER_LINEAR)
            mask[i]  = cv2.resize(mask[i],  dsize=(size, size), interpolation=cv2.INTER_LINEAR)
            edge[i]  = cv2.resize(edge[i],  dsize=(size, size), interpolation=cv2.INTER_LINEAR)
        image = torch.from_numpy(np.stack(image, axis=0)).permute(0,3,1,2)
        mask  = torch.from_numpy(np.stack(mask, axis=0)).unsqueeze(1)
        edge  = torch.from_numpy(np.stack(edge, axis=0)).unsqueeze(1)
        return image, mask,edge

    def __len__(self):
        return len(self.images)


if __name__=='__main__':
    data = Data
