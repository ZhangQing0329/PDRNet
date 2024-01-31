import torch
from torch import nn
from torch.nn import init
import torch.nn.functional as F
import math
from torch.autograd import Variable
import numpy as np

# from .deeplab_resnet import resnet50_locate
# from .vgg import vgg16_locate
from torchvision.models.resnet import resnet50

def weight_init(module):
    for n, m in module.named_children():
        print('initialize: '+n)
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, (nn.BatchNorm2d, nn.InstanceNorm2d)):
            nn.init.ones_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Sequential):
            weight_init(m)
        elif isinstance(m, nn.ReLU):
            pass
        else:
            m.initialize()


class Bottleneck(nn.Module):
    def __init__(self, inplanes, planes, stride=1, downsample=None, dilation=1):
        super(Bottleneck, self).__init__()
        self.conv1      = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1        = nn.BatchNorm2d(planes)
        self.conv2      = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=(3*dilation-1)//2, bias=False, dilation=dilation)
        self.bn2        = nn.BatchNorm2d(planes)
        self.conv3      = nn.Conv2d(planes, planes*4, kernel_size=1, bias=False)
        self.bn3        = nn.BatchNorm2d(planes*4)
        self.downsample = downsample

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)), inplace=True)
        out = F.relu(self.bn2(self.conv2(out)), inplace=True)
        out = self.bn3(self.conv3(out))
        if self.downsample is not None:
            x = self.downsample(x)
        return F.relu(out+x, inplace=True)


class ResNet(nn.Module):
    def __init__(self):
        super(ResNet, self).__init__()
        self.inplanes = 64
        self.conv1    = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1      = nn.BatchNorm2d(64)
        self.layer1   = self.make_layer( 64, 3, stride=1, dilation=1)
        self.layer2   = self.make_layer(128, 4, stride=2, dilation=1)
        self.layer3   = self.make_layer(256, 6, stride=2, dilation=1)
        self.layer4   = self.make_layer(512, 3, stride=2, dilation=1)

    def make_layer(self, planes, blocks, stride, dilation):
        downsample    = nn.Sequential(nn.Conv2d(self.inplanes, planes*4, kernel_size=1, stride=stride, bias=False), nn.BatchNorm2d(planes*4))
        layers        = [Bottleneck(self.inplanes, planes, stride, downsample, dilation=dilation)]
        self.inplanes = planes*4
        for _ in range(1, blocks):
            layers.append(Bottleneck(self.inplanes, planes, dilation=dilation))
        return nn.Sequential(*layers)

    def forward(self, x):
        out6 = F.relu(self.bn1(self.conv1(x)), inplace=True)
        out1 = F.max_pool2d(out6, kernel_size=3, stride=2, padding=1)
        out2 = self.layer1(out1)
        out3 = self.layer2(out2)
        out4 = self.layer3(out3)
        out5 = self.layer4(out4)
        return out6, out2, out3, out4, out5

    def initialize(self):
        res50 = models.resnet50(pretrained=True)
        self.load_state_dict(res50.state_dict(), False)

class ResNet50(nn.Module):
    def __init__(self):
        super(ResNet50, self).__init__()
        r50 = resnet50(True)
        self.cv1 = r50.conv1
        self.bn1 = r50.bn1
        self.mxp = r50.maxpool
        self.re1 = r50.relu
        self.layer1 = r50.layer1
        self.layer2 = r50.layer2
        self.layer3 = r50.layer3
        self.layer4 = r50.layer4

    def forward(self, x):
        x1 = self.mxp(self.re1(self.bn1(self.cv1(x))))
        x2 = self.layer1(x1)
        x3 = self.layer2(x2)
        x4 = self.layer3(x3)
        x5 = self.layer4(x4)
        return x1, x2, x3, x4, x5

# Hierarchical Feature Screening Module
class HFSM(nn.Module):

    def __init__(self, in_channel, depth):
        super(HFSM, self).__init__()
        self.in_channel = in_channel
        self.depth = depth

        #self.pool = nn.AdaptiveMaxPool2d((1, 1))
        self.conv = nn.Sequential(nn.Conv2d(self.in_channel, self.depth, 1, 1), nn.BatchNorm2d(self.depth), nn.ReLU(inplace =True))

        self.senet_layer6 = SENet(self.depth)
        self.senet_layer12 = SENet(self.depth)
        self.senet_layer18 = SENet(self.depth)

        self.atrous_block1 = nn.Sequential(nn.Conv2d(self.in_channel, self.depth, 1, 1), nn.BatchNorm2d(self.depth), nn.ReLU(inplace=True))
        self.atrous_block6 = nn.Sequential(nn.Conv2d(self.in_channel, self.depth, 3, 1, padding=2, dilation=2), nn.BatchNorm2d(self.depth), nn.ReLU(inplace=True))
        self.atrous_block12 = nn.Sequential(nn.Conv2d(self.in_channel, self.depth, 3, 1, padding=3, dilation=3), nn.BatchNorm2d(self.depth), nn.ReLU(inplace=True))
        self.atrous_block18 = nn.Sequential(nn.Conv2d(self.in_channel, self.depth, 3, 1, padding=4, dilation=4), nn.BatchNorm2d(self.depth), nn.ReLU(inplace=True))

        self.conv12 = nn.Sequential(nn.Conv2d(self.depth * 2, self.depth, 1, 1), nn.BatchNorm2d(self.depth), nn.ReLU(inplace=True))
        self.conv18 = nn.Sequential(nn.Conv2d(self.depth * 3, self.depth, 1, 1), nn.BatchNorm2d(self.depth), nn.ReLU(inplace=True))

        self.conv_output = nn.Sequential(nn.Conv2d(self.depth * 5, self.depth, 1, 1), nn.BatchNorm2d(self.depth), nn.ReLU(inplace=True))

        self.conv3 = nn.Sequential(nn.Conv2d(self.in_channel, self.depth, 1, 1, 0), nn.BatchNorm2d(self.depth), nn.ReLU(inplace=True))

    def forward(self, x):
        feature6_size = x.size()
        # branch 1
        atrous_block1 = self.atrous_block1(x)

        # branch 2
        atrous_block6 = self.atrous_block6(x)
        d6 = self.senet_layer6(atrous_block6)

        # branch 3
        atrous_block12 = self.atrous_block12(x)
        atrous_block12 = torch.cat([d6, atrous_block12], dim=1)
        atrous_block12 = self.conv12(atrous_block12)
        d12 = self.senet_layer12(atrous_block12)

        # branch 4
        atrous_block18 = self.atrous_block18(x)
        atrous_block18 = torch.cat([d6, d12, atrous_block18], dim=1)
        atrous_block18 = self.conv18(atrous_block18)
        d18 = self.senet_layer18(atrous_block18)

        # branch 5
        pool_features = F.adaptive_avg_pool2d(x, (1,1))
        pool_features = self.conv(pool_features)
        pool_features = F.upsample(pool_features, size=feature6_size[2:], mode='bilinear')

        # fusion
        out = self.conv_output(torch.cat([atrous_block1, d6, d12, d18, pool_features], dim=1))

        return out + self.conv3(x)

# residual learning
class RLM(nn.Module):
    def __init__(self, in_hc=64, out_c=64):
        super(RLM, self).__init__()
        self.conv2 = SpatialAttention()

    def forward(self, x, df):
        x_size = x.size()
        out = self.conv2(x).repeat(1, x.size()[1], 1, 1)*x + df

        return out

# context enrichment module
class feature_exctraction(nn.Module):
    def __init__(self, in_channel, depth, kernel):
        super(feature_exctraction, self).__init__()
        self.in_channel = in_channel
        self.depth = depth
        self.kernel = kernel
        self.conv1 = nn.Sequential(nn.Conv2d(self.in_channel, self.depth, self.kernel, 1, (self.kernel - 1) // 2),
                                   nn.BatchNorm2d(self.depth), nn.ReLU(inplace=True))
        self.conv2 = nn.Sequential(nn.Conv2d(self.depth, self.depth, self.kernel, 1, (self.kernel - 1) // 2),
                                   nn.BatchNorm2d(self.depth), nn.ReLU(inplace=True))
        self.conv3 = nn.Sequential(nn.Conv2d(self.depth, self.depth, 3, 1, 1), nn.BatchNorm2d(self.depth), nn.ReLU(inplace=True))

    def forward(self, x):
        conv1 = self.conv1(x)
        conv2 = self.conv2(conv1)
        output = self.conv3(conv2)

        return output


class SANet(nn.Module):

    def __init__(self, in_dim):
        super(SANet, self).__init__()
        self.dim = in_dim
        self.k = 9
        self.conv1_1 = nn.Sequential(nn.Conv2d(in_dim, self.dim // 2, (1, self.k), 1, (0, self.k // 2)), nn.BatchNorm2d(self.dim // 2), nn.ReLU(inplace=True))
        self.conv1_2 = nn.Sequential(nn.Conv2d(in_dim, self.dim // 2, (self.k, 1), 1, (self.k // 2, 0)), nn.BatchNorm2d(self.dim // 2), nn.ReLU(inplace=True))
        self.conv2_1 = nn.Conv2d(self.dim // 2, 1, (self.k, 1), 1, (self.k // 2, 0))
        self.conv2_2 = nn.Conv2d(self.dim // 2, 1, (1, self.k), 1, (0, self.k // 2))
        #self.sigmoid = nn.Sigmoid()

    def forward(self, x):

        conv1_1 = self.conv1_1(x)
        conv1_2 = self.conv1_2(x)
        conv2_1 = self.conv2_1(conv1_1)
        conv2_2 = self.conv2_2(conv1_2)
        conv3 = torch.add(conv2_1, conv2_2)
        conv4 = torch.sigmoid(conv3)

        conv5 = conv4.repeat(1, self.dim // 2, 1, 1)

        return conv5


class SpatialAttention(nn.Module):
    def __init__(self):
        super(SpatialAttention, self).__init__()
        self.conv = nn.Conv2d(in_channels=2, out_channels=1, kernel_size=5, stride=1, padding=2,
                              bias=False)  # infer a one-channel attention map

    def forward(self, ftr):
        # ftr: [B, C, H, W]
        ftr_avg = torch.mean(ftr, dim=1, keepdim=True)  # [B, 1, H, W], average
        ftr_max, _ = torch.max(ftr, dim=1, keepdim=True)  # [B, 1, H, W], max
        ftr_cat = torch.cat([ftr_avg, ftr_max], dim=1)  # [B, 2, H, W]
        att_map = torch.sigmoid(self.conv(ftr_cat))  # [B, 1, H, W]
        return att_map


class Edge_detect(nn.Module):

    def __init__(self):
        super(Edge_detect, self).__init__()
        self.conv1 = nn.Sequential(nn.Conv2d(64, 64, 3, 1, 1), nn.BatchNorm2d(64), nn.ReLU(inplace=True))
        self.conv2 = nn.Sequential(nn.Conv2d(64, 64, 3, 1, 1), nn.BatchNorm2d(64))
        self.relu = nn.ReLU(inplace=True)
        self.conv3 = nn.Conv2d(64, 1, 1, 1, 0)

    def forward(self, x):
        conv1 = self.conv1(x)
        conv2 = self.conv2(conv1)
        output = self.conv3(self.relu(conv2 + x))

        return output


class PDRNet(nn.Module):
    def __init__(self, cfg, channel=64):
        self.cfg = cfg
        super(PDRNet, self).__init__()
        self.bkbone = ResNet50()
        self.predict_layer6 = HFSM(2048, 64)

        self.sanet_layer = SANet(128)

        self.fem_layer5 = feature_exctraction(512, 64, 7)
        self.fem_layer4 = feature_exctraction(512, 64, 5)
        self.fem_layer3 = feature_exctraction(256, 64, 5)
        self.fem_layer2 = feature_exctraction(128, 64, 3)
        self.fem_layer1 = feature_exctraction(64, 64, 3)

        self.con1 = nn.Sequential(nn.Conv2d(64, 64, 3,1, 1),nn.BatchNorm2d(64),nn.ReLU(inplace = True))
        self.con2 = nn.Sequential(nn.Conv2d(256, 128, 3,1, 1),nn.BatchNorm2d(128),nn.ReLU(inplace = True))
        self.con3 = nn.Sequential(nn.Conv2d(512, 256, 3,1, 1),nn.BatchNorm2d(256),nn.ReLU(inplace = True))
        self.con4 = nn.Sequential(nn.Conv2d(1024, 512,3,1, 1),nn.BatchNorm2d(512),nn.ReLU(inplace = True))
        self.con5 = nn.Sequential(nn.Conv2d(2048, 512,3,1,  1),nn.BatchNorm2d(512),nn.ReLU(inplace = True))

        self.fuse_5 = nn.Sequential(nn.Conv2d(128, 64, 3, 1, 1), nn.BatchNorm2d(64), nn.ReLU(inplace = True))
        self.fuse_4 = nn.Sequential(nn.Conv2d(128, 64, 3, 1, 1), nn.BatchNorm2d(64), nn.ReLU(inplace =True))
        self.fuse_3 = nn.Sequential(nn.Conv2d(128, 64, 3, 1, 1), nn.BatchNorm2d(64), nn.ReLU(inplace =True))
        self.fuse_2 = nn.Sequential(nn.Conv2d(128, 64, 3, 1, 1), nn.BatchNorm2d(64), nn.ReLU(inplace =True))
        self.fuse_1 = nn.Sequential(nn.Conv2d(128, 64, 3, 1, 1), nn.BatchNorm2d(64), nn.ReLU(inplace =True))

        self.fuse_5D = RLM(64, 64)
        self.fuse_4D = RLM(64, 64)
        self.fuse_3D = RLM(64, 64)
        self.fuse_2D = RLM(64, 64)
        self.fuse_1D = RLM(64, 64)

        self.predict_6 = nn.Conv2d(64, 1, 1, 1, 0)
        self.predict_5 = nn.Conv2d(64, 1, 1, 1, 0)
        self.predict_4 = nn.Conv2d(64, 1, 1, 1, 0)
        self.predict_3 = nn.Conv2d(64, 1, 1, 1, 0)
        self.predict_2 = nn.Conv2d(64, 1, 1, 1, 0)
        self.predict_1 = nn.Conv2d(64, 1, 1, 1, 0)

        self.edgedetect_1_back = Edge_detect()
        self.edgedetect_1_fore = Edge_detect()
        self.edgedetect_2_back = Edge_detect()
        self.edgedetect_2_fore = Edge_detect()
        self.edgedetect_1 = Edge_detect()
        self.edgedetect_2 = Edge_detect()

        #self.initialize()

    def forward(self, x):
        conv1, conv2, conv3, conv4, conv5 = self.bkbone(x)

        x_size = x.size()
        # stage 6
        h_5=conv5.size()[2]
        w_5=conv5.size()[3]
        feature6 =  F.adaptive_avg_pool2d(conv5,(h_5//2, w_5//2))
        out6 = self.predict_layer6(feature6)
        predict6 = F.upsample(self.predict_6(out6), size=x_size[2:], mode='bilinear')

        conv1 = self.con1(conv1)
        conv2 = self.con2(conv2)
        conv3 = self.con3(conv3)
        conv4 = self.con4(conv4)
        conv5 = self.con5(conv5)

        fem_layer5 = self.fem_layer5(conv5)
        fem_layer4 = self.fem_layer4(conv4)
        fem_layer3 = self.fem_layer3(conv3)
        fem_layer2 = self.fem_layer2(conv2)
        fem_layer1 = self.fem_layer1(conv1)

        # stage5 Dual-attention Residual Module
        up_out6 = F.upsample(out6, size=fem_layer5.size()[2:], mode='bilinear')
        out5_concat = torch.cat([fem_layer5, up_out6], dim=1)
        out5_attention = self.sanet_layer(out5_concat)
        out5_f = out5_attention * out5_attention * fem_layer5+fem_layer5
        out5_b = - (1 - out5_attention) * (1 - out5_attention) * fem_layer5+fem_layer5
        out5 = self.fuse_5(torch.cat([out5_f, out5_b], dim=1))
        out5 = self.fuse_5D(out5, up_out6)
        predict5 = F.upsample(self.predict_5(out5), size=x_size[2:], mode='bilinear')

        # stage4 Dual-attention Residual Module
        up_out5 = F.upsample(out5, size=fem_layer4.size()[2:], mode='bilinear')
        out4_concat = torch.cat([fem_layer4, up_out5], dim=1)
        out4_attention = self.sanet_layer(out4_concat)
        out4_f = out4_attention * fem_layer4+fem_layer4
        out4_b = - (1 - out4_attention) * fem_layer4+fem_layer4
        out4 = self.fuse_4(torch.cat([out4_f, out4_b], dim=1))
        out4 = self.fuse_4D(out4, up_out5)
        predict4 = F.upsample(self.predict_4(out4), size=x_size[2:], mode='bilinear')

        # stage3 Dual-attention Residual Module
        up_out4 = F.upsample(out4, size=fem_layer3.size()[2:], mode='bilinear')
        out3_concat = torch.cat([fem_layer3, up_out4], dim=1)
        out3_attention = self.sanet_layer(out3_concat)
        out3_f = out3_attention * fem_layer3+fem_layer3
        out3_b = - (1 - out3_attention) * fem_layer3+fem_layer3
        out3 = self.fuse_3(torch.cat([out3_f, out3_b], dim=1))
        out3 = self.fuse_3D(out3, up_out4)
        predict3 = F.upsample(self.predict_3(out3), size=x_size[2:], mode='bilinear')

        # stage2 Dual-attention Residual Module
        up_out3 = F.upsample(out3, size=fem_layer2.size()[2:], mode='bilinear')
        out2_concat = torch.cat([fem_layer2, up_out3], dim=1)
        out2_attention = self.sanet_layer(out2_concat)
        out2_f = out2_attention * fem_layer2+fem_layer2
        out2_b = - (1 - out2_attention) * fem_layer2+fem_layer2
        out2 = self.fuse_2(torch.cat([out2_f, out2_b], dim=1))
        out2 = self.fuse_2D(out2, up_out3)
        predict2 = F.upsample(self.predict_2(out2), size=x_size[2:], mode='bilinear')

        edge2_fore = self.edgedetect_2_fore(out2_f)
        edge2_back = self.edgedetect_2_back(out2_b)
        predict2_fore = F.upsample(edge2_fore, size=x_size[2:], mode='bilinear')
        predict2_back = F.upsample(edge2_back, size=x_size[2:], mode='bilinear')

        edge2_sal = self.edgedetect_2(out2)
        predict2_edge = F.upsample(edge2_sal, size=x_size[2:], mode='bilinear')

        # stage1 Dual-attention Residual Module
        up_out2 = F.upsample(out2, size=fem_layer1.size()[2:], mode='bilinear')
        out1_concat = torch.cat([fem_layer1, up_out2], dim=1)
        out1_attention = self.sanet_layer(out1_concat)
        out1_f = out1_attention * fem_layer1+fem_layer1
        out1_b = - (1 - out1_attention) * fem_layer1+fem_layer1
        out1 = self.fuse_1(torch.cat([out1_f, out1_b], dim=1))
        out1 = self.fuse_1D(out1, up_out2)
        predict1 = F.upsample(self.predict_1(out1), size=x_size[2:], mode='bilinear')

        edge1_fore = self.edgedetect_1_fore(out1_f)
        edge1_back = self.edgedetect_1_back(out1_b)
        predict1_fore = F.upsample(edge1_fore, size=x_size[2:], mode='bilinear')
        predict1_back = F.upsample(edge1_back, size=x_size[2:], mode='bilinear')

        edge1_sal = self.edgedetect_1(out1)
        predict1_edge = F.upsample(edge1_sal, size=x_size[2:], mode='bilinear')

        return predict1, predict2, predict3, predict4, predict5, predict6, predict1_fore, predict1_back, predict2_fore, predict2_back, predict1_edge, predict2_edge, (1-out3_attention)[:,:1,:,:]

