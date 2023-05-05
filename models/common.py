# YOLOv5 common modules

import math
from pathlib import Path

import numpy as np
import requests
import torch
import torch.nn as nn
from PIL import Image
import torch.nn.functional as F
from torch.autograd import Variable, Function
from utils.datasets import letterbox
from utils.general import non_max_suppression, make_divisible, scale_coords, xyxy2xywh
from utils.plots import color_list, plot_one_box
from utils.torch_utils import time_synchronized

# c1:in_channels   multi_mix_t:out_channels
# nn.Conv2d(self,in_channels,out_channels,kernel_size,stride=1,padding=0,dilation=1,groups=1,bias=True,padding_mode='zeros')



# backbone各个参数详解
def autopad(k, p=None):  # kernel, padding
    # Pad to 'same'
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p


def DWConv(c1, c2, k=1, s=1, act=True):
    # Depthwise convolution
    # input=output=g
    return Conv(c1, c2, k, s, g=math.gcd(c1, c2), act=act)

# 标准卷积层：区别于CBL,激活函数为 SiLU, conv + BN + SiLU
class Conv(nn.Module):
    # Standard convolution

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True):  # ch_in, ch_out, kernel, stride, padding, groups
        super(Conv, self).__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p), groups=g, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.SiLU() if act is True else (act if isinstance(act, nn.Module) else nn.Identity())

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

    def fuseforward(self, x):
        return self.act(self.conv(x))   # 少了BN


# 输入：RGB thermal
# 输出：RGB thermal
class Conv_t(nn.Module):
    # Standard convolution

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True):  # ch_in, ch_out, kernel, stride, padding, groups
        super(Conv_t, self).__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p), groups=g, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.SiLU() if act is True else (act if isinstance(act, nn.Module) else nn.Identity())

        self.conv_t = nn.Conv2d(c1, c2, k, s, autopad(k, p), groups=g, bias=False)
        self.bn_t = nn.BatchNorm2d(c2)
        self.act_t = nn.SiLU() if act is True else (act if isinstance(act, nn.Module) else nn.Identity())

    def forward(self, x ,y):
        x = self.act(self.bn(self.conv(x)))
        y = self.act_t(self.bn_t(self.conv_t(y)))
        return x,y

    def fuseforward(self, x, y):
        x = self.act(self.conv(x))
        y = self.act_t(self.conv_t(y))
        return x,y







# 标准bottleneck
# Res_Unit: feature map 尺寸不变,自动填充
# Conv(c1, c_, 1, 1)  == conv1x1(c1,c_)+BN(c_)+SiLU()
class Bottleneck(nn.Module):
    # Standard bottleneck
    def __init__(self, c1, c2, shortcut=True, g=1, e=0.5):  # ch_in, ch_out, shortcut, groups, expansion
        super(Bottleneck, self).__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)   #  conv1x1(c1,c2*e) == conv1x1(c1,c_)+BN(c_)+SiLU()
        self.cv2 = Conv(c_, c2, 3, 1, g=g)   #  conv3x3(c2*e,c2) == conv3x3(c_,c2)+BN(c2)+SiLU()
        self.add = shortcut and c1 == c2

    def forward(self, x):
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))



class BottleneckCSP(nn.Module):
    # CSP Bottleneck https://github.com/WongKinYiu/CrossStagePartialNetworks
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):  # ch_in, ch_out, number, shortcut, groups, expansion
        super(BottleneckCSP, self).__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = nn.Conv2d(c1, c_, 1, 1, bias=False)
        self.cv3 = nn.Conv2d(c_, c_, 1, 1, bias=False)
        self.cv4 = Conv(2 * c_, c2, 1, 1)
        self.bn = nn.BatchNorm2d(2 * c_)  # applied to cat(cv2, cv3)
        self.act = nn.LeakyReLU(0.1, inplace=True)
        self.m = nn.Sequential(*[Bottleneck(c_, c_, shortcut, g, e=1.0) for _ in range(n)])

    def forward(self, x):
        y1 = self.cv3(self.m(self.cv1(x)))
        y2 = self.cv2(x)
        return self.cv4(self.act(self.bn(torch.cat((y1, y2), dim=1))))


class C3(nn.Module):
    # CSP Bottleneck with 3 convolutions
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):  # ch_in, ch_out, number, shortcut, groups, expansion
        super(C3, self).__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c1, c_, 1, 1)
        self.cv3 = Conv(2 * c_, c2, 1)  # act=FReLU(c2)
        self.m = nn.Sequential(*[Bottleneck(c_, c_, shortcut, g, e=1.0) for _ in range(n)])
        # self.m = nn.Sequential(*[CrossConv(c_, c_, 3, 1, g, 1.0, shortcut) for _ in range(n)])

    def forward(self, x):
        return self.cv3(torch.cat((self.m(self.cv1(x)), self.cv2(x)), dim=1))


# 输入：RGB, thermal
# 输出：处理后的 RGB, thermal
class C3_t(nn.Module):
    # CSP Bottleneck with 3 convolutions
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):  # ch_in, ch_out, number, shortcut, groups, expansion
        super(C3_t, self).__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c1, c_, 1, 1)
        self.cv3 = Conv(2 * c_, c2, 1)  # act=FReLU(c2)
        self.m = nn.Sequential(*[Bottleneck(c_, c_, shortcut, g, e=1.0) for _ in range(n)])
        # self.m = nn.Sequential(*[CrossConv(c_, c_, 3, 1, g, 1.0, shortcut) for _ in range(n)])

        self.cv1_t = Conv(c1, c_, 1, 1)
        self.cv2_t = Conv(c1, c_, 1, 1)
        self.cv3_t = Conv(2 * c_, c2, 1)  # act=FReLU(c2)
        self.m_t = nn.Sequential(*[Bottleneck(c_, c_, shortcut, g, e=1.0) for _ in range(n)])




    def forward(self, x ,y):
        x = self.cv3(torch.cat((self.m(self.cv1(x)), self.cv2(x)), dim=1))
        y = self.cv3_t(torch.cat((self.m_t(self.cv1_t(y)), self.cv2_t(y)), dim=1))
        return x,y
class C3_t_cat(nn.Module):
    # CSP Bottleneck with 3 convolutions
    #相比于C3_t，区别为最后将得到的输出x,y沿dim=1拼接到一起输出一个通道。
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):  # ch_in, ch_out, number, shortcut, groups, expansion
        super(C3_t_cat, self).__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c1, c_, 1, 1)
        self.cv3 = Conv(2 * c_, c2, 1)  # act=FReLU(c2)
        self.m = nn.Sequential(*[Bottleneck(c_, c_, shortcut, g, e=1.0) for _ in range(n)])
        # self.m = nn.Sequential(*[CrossConv(c_, c_, 3, 1, g, 1.0, shortcut) for _ in range(n)])

        self.cv1_t = Conv(c1, c_, 1, 1)
        self.cv2_t = Conv(c1, c_, 1, 1)
        self.cv3_t = Conv(2 * c_, c2, 1)  # act=FReLU(c2)
        self.m_t = nn.Sequential(*[Bottleneck(c_, c_, shortcut, g, e=1.0) for _ in range(n)])




    def forward(self, x ,y):
        x = self.cv3(torch.cat((self.m(self.cv1(x)), self.cv2(x)), dim=1))
        y = self.cv3_t(torch.cat((self.m_t(self.cv1_t(y)), self.cv2_t(y)), dim=1))
        return torch.cat((x,y),dim=1)

# gy:RGB+thermal2
# 将第5个模块c3进行修改,去掉CSP结构,直接作为多模态融合方案
class C4(nn.Module):
    # CSP Bottleneck with 3 convolutions
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):  # ch_in, ch_out, number, shortcut, groups, expansion
        super(C4, self).__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv1_t = Conv(c1, c_, 1, 1)
        # self.cv2 = Conv(c1, c_, 1, 1)
        self.cv3 = Conv(2 * c_, c2, 1)  # act=FReLU(c2)
        self.m = nn.Sequential(*[Bottleneck(c_, c_, shortcut, g, e=1.0) for _ in range(n)])
        self.m_t = nn.Sequential(*[Bottleneck(c_, c_, shortcut, g, e=1.0) for _ in range(n)])
        # self.m = nn.Sequential(*[CrossConv(c_, c_, 3, 1, g, 1.0, shortcut) for _ in range(n)])

    def forward(self,x,y):
        return self.cv3(torch.cat((self.m(self.cv1(x)), self.m_t(self.cv1_t(y))), dim=1))
class C4_plard_new(nn.Module):
    # CSP Bottleneck with 3 convolutions
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):  # ch_in, ch_out, number, shortcut, groups, expansion
        super(C4_plard_new, self).__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv1_t = Conv(c1, c_, 1, 1)
        # self.cv2 = Conv(c1, c_, 1, 1)
        self.cv3 = Conv(2 * c_, c2, 1)  # act=FReLU(c2)
        self.m = nn.Sequential(*[Bottleneck(c_, c_, shortcut, g, e=1.0) for _ in range(n)])
        self.m_t = nn.Sequential(*[Bottleneck(c_, c_, shortcut, g, e=1.0) for _ in range(n)])
        self.GAP=nn.AdaptiveAvgPool2d(1)
        self.conv1=Conv(c1,c1,1)
        self.conv1_t=Conv(c1,c1,1)
        self.conv2=Conv(c1,c1,1)
        self.conv2_t=Conv(c1,c1,1)
        self.x_tr=nn.Conv2d(c1,c1,kernel_size=1,stride=1,padding=0,bias=True)
        self.x_t_tr=nn.Conv2d(c1,c1,kernel_size=1,stride=1,padding=0,bias=True)
        self.tr_alpha=nn.Conv2d(2*c1,c1,kernel_size=1,stride=1,padding=0,bias=True)
        self.tr_beta=nn.Conv2d(2*c1,c1,kernel_size=1,stride=1,padding=0,bias=True)
        self.relu=nn.ReLU(inplace=True)
        self.x_tr_2=nn.Conv2d(c1,c1,kernel_size=1,stride=1,padding=0,bias=True)
        self.x_t_tr_2=nn.Conv2d(c1,c1,kernel_size=1,stride=1,padding=0,bias=True)
        self.x_feat_fuse=nn.Conv2d(c1,c1,kernel_size=1,stride=1,padding=0,bias=True)
        self.y_feat_fuse=nn.Conv2d(c1,c1,kernel_size=1,stride=1,padding=0,bias=True)
        self.tr_alpha_copy=nn.Conv2d(2*c1,c1,kernel_size=1,stride=1,padding=0,bias=True)
        self.tr_beta_copy=nn.Conv2d(2*c1,c1,kernel_size=1,stride=1,padding=0,bias=True)
        # self.m = nn.Sequential(*[CrossConv(c_, c_, 3, 1, g, 1.0, shortcut) for _ in range(n)])

    def forward(self,x,y):
        x_tr=self.x_tr(x)
        x_tr=self.relu(x_tr)
        y_tr=self.x_t_tr(y)
        y_tr=self.relu(y_tr)
        cond=torch.cat((x_tr,y_tr),1)
        alpha=self.tr_alpha(cond)
        beta=self.tr_beta(cond)
        alpha_copy=self.tr_alpha_copy(cond)
        beta_copy=self.tr_beta_copy(cond)
        y_feat=self.x_tr_2(y)
        y_feat=(alpha+1)*y_feat+beta
        y_feat=self.relu(y_feat)
        y_feat_fuse=self.y_feat_fuse(y_feat)
        x_feat=self.x_tr_2(x)
        x_feat=(alpha_copy+1)*x_feat+beta_copy
        x_feat=self.relu(x_feat)
        x_feat_fuse=self.x_feat_fuse(x_feat)
        y=y+0.1*x_feat_fuse
        x=x+0.1*y_feat_fuse
        fdx=x-y#N,C,H,W
        vx=F.tanh(self.GAP(fdx))#N,C,1,1
        x_res=x+vx*y
        fdy=y-x
        vy=F.tanh(self.GAP(fdy))
        y_res=y+vy*x
        x=self.conv1(x)
        x_res=self.conv2(x_res)
        y=self.conv1_t(y)
        y_res=self.conv2_t(y_res)
        x=x+x_res
        y=y+y_res
        return self.cv3(torch.cat((self.m(self.cv1(x)), self.m_t(self.cv1_t(y))), dim=1))

class C4_ma(nn.Module):
    # CSP Bottleneck with 3 convolutions
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):  # ch_in, ch_out, number, shortcut, groups, expansion
        super(C4_ma, self).__init__()
        self.GAP=nn.AdaptiveAvgPool2d(1)
        self.conv1=Conv(c1,c1,1)
        self.conv1_t=Conv(c1,c1,1)
        self.conv2=Conv(c1,c1,1)
        self.conv2_t=Conv(c1,c1,1)
        self.x_tr=nn.Conv2d(c1,c1,kernel_size=1,stride=1,padding=0,bias=True)
        self.x_t_tr=nn.Conv2d(c1,c1,kernel_size=1,stride=1,padding=0,bias=True)
        self.tr_alpha=nn.Conv2d(2*c1,c1,kernel_size=1,stride=1,padding=0,bias=True)
        self.tr_beta=nn.Conv2d(2*c1,c1,kernel_size=1,stride=1,padding=0,bias=True)
        self.relu=nn.ReLU(inplace=True)
        self.x_tr_2=nn.Conv2d(c1,c1,kernel_size=1,stride=1,padding=0,bias=True)
        self.x_t_tr_2=nn.Conv2d(c1,c1,kernel_size=1,stride=1,padding=0,bias=True)
        self.x_feat_fuse=nn.Conv2d(c1,c1,kernel_size=1,stride=1,padding=0,bias=True)
        self.y_feat_fuse=nn.Conv2d(c1,c1,kernel_size=1,stride=1,padding=0,bias=True)
        self.tr_alpha_copy=nn.Conv2d(2*c1,c1,kernel_size=1,stride=1,padding=0,bias=True)
        self.tr_beta_copy=nn.Conv2d(2*c1,c1,kernel_size=1,stride=1,padding=0,bias=True)
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv1_t = Conv(c1, c_, 1, 1)
        # self.cv2 = Conv(c1, c_, 1, 1)
        self.cv3 = Conv(2 * c_, c2, 1)  # act=FReLU(c2)
        self.m = nn.Sequential(*[Bottleneck(c_, c_, shortcut, g, e=1.0) for _ in range(n)])
        self.m_t = nn.Sequential(*[Bottleneck(c_, c_, shortcut, g, e=1.0) for _ in range(n)])
        # self.m = nn.Sequential(*[CrossConv(c_, c_, 3, 1, g, 1.0, shortcut) for _ in range(n)])

    def forward(self,x,y):
        x_tr=self.x_tr(x)
        x_tr=self.relu(x_tr)
        y_tr=self.x_t_tr(y)
        y_tr=self.relu(y_tr)
        cond=torch.cat((x_tr,y_tr),1)
        alpha=self.tr_alpha(cond)
        beta=self.tr_beta(cond)
        alpha_copy=self.tr_alpha_copy(cond)
        beta_copy=self.tr_beta_copy(cond)
        y_feat=self.x_tr_2(y)
        y_feat=(alpha+1)*y_feat+beta
        y_feat=self.relu(y_feat)
        y_feat_fuse=self.y_feat_fuse(y_feat)
        x_feat=self.x_tr_2(x)
        x_feat=(alpha_copy+1)*x_feat+beta_copy
        x_feat=self.relu(x_feat)
        x_feat_fuse=self.x_feat_fuse(x_feat)
#         y=y+0.1*x_feat_fuse
#         x=x+0.1*y_feat_fuse
        x=y_feat_fuse
        y=x_feat_fuse
        fdx=x-y#N,C,H,W
        vx=F.tanh(self.GAP(fdx))#N,C,1,1
        x_res=x+vx*y
        fdy=y-x
        vy=F.tanh(self.GAP(fdy))
        y_res=y+vy*x
        x=self.conv1(x)
        x_res=self.conv2(x_res)
        y=self.conv1_t(y)
        y_res=self.conv2_t(y_res)
        x=x+x_res
        y=y+y_res
        return self.cv3(torch.cat((self.m(self.cv1(x)), self.m_t(self.cv1_t(y))), dim=1))
class C4_simple(nn.Module):
    # CSP Bottleneck with 3 convolutions
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):  # ch_in, ch_out, number, shortcut, groups, expansion
        super(C4_simple, self).__init__()
        self.GAP=nn.AdaptiveAvgPool2d(1)
        self.conv1=Conv(c1,c1,1)
        self.conv1_t=Conv(c1,c1,1)
        self.conv2=Conv(c1,c1,1)
        self.conv2_t=Conv(c1,c1,1)
        self.x_tr=nn.Conv2d(c1,c1,kernel_size=1,stride=1,padding=0,bias=True)
        self.x_t_tr=nn.Conv2d(c1,c1,kernel_size=1,stride=1,padding=0,bias=True)
        self.tr_alpha=nn.Conv2d(2*c1,c1,kernel_size=1,stride=1,padding=0,bias=True)
        self.tr_beta=nn.Conv2d(2*c1,c1,kernel_size=1,stride=1,padding=0,bias=True)
        self.relu=nn.ReLU(inplace=True)
        self.x_tr_2=nn.Conv2d(c1,c1,kernel_size=1,stride=1,padding=0,bias=True)
        self.x_t_tr_2=nn.Conv2d(c1,c1,kernel_size=1,stride=1,padding=0,bias=True)
        self.x_feat_fuse=nn.Conv2d(c1,c1,kernel_size=1,stride=1,padding=0,bias=True)
        self.y_feat_fuse=nn.Conv2d(c1,c1,kernel_size=1,stride=1,padding=0,bias=True)
        self.tr_alpha_copy=nn.Conv2d(2*c1,c1,kernel_size=1,stride=1,padding=0,bias=True)
        self.tr_beta_copy=nn.Conv2d(2*c1,c1,kernel_size=1,stride=1,padding=0,bias=True)
        self.cv3 = Conv(2 * c1, c2, 1)  # act=FReLU(c2)

    def forward(self,x,y):
        x_tr=self.x_tr(x)
        x_tr=self.relu(x_tr)
        y_tr=self.x_t_tr(y)
        y_tr=self.relu(y_tr)
        cond=torch.cat((x_tr,y_tr),1)
        alpha=self.tr_alpha(cond)
        beta=self.tr_beta(cond)
        alpha_copy=self.tr_alpha_copy(cond)
        beta_copy=self.tr_beta_copy(cond)
        y_feat=self.x_tr_2(y)
        y_feat=(alpha+1)*y_feat+beta
        y_feat=self.relu(y_feat)
        y_feat_fuse=self.y_feat_fuse(y_feat)
        x_feat=self.x_tr_2(x)
        x_feat=(alpha_copy+1)*x_feat+beta_copy
        x_feat=self.relu(x_feat)
        x_feat_fuse=self.x_feat_fuse(x_feat)
#         y=y+0.1*x_feat_fuse
#         x=x+0.1*y_feat_fuse
        x=y_feat_fuse
        y=x_feat_fuse
        fdx=x-y#N,C,H,W
        vx=F.tanh(self.GAP(fdx))#N,C,1,1
        x_res=x+vx*y
        fdy=y-x
        vy=F.tanh(self.GAP(fdy))
        y_res=y+vy*x
        x=self.conv1(x)
        x_res=self.conv2(x_res)
        y=self.conv1_t(y)
        y_res=self.conv2_t(y_res)
        x=x+x_res
        y=y+y_res
        return self.cv3(torch.cat((x,y), dim=1))
class C4_new(nn.Module):
    # CSP Bottleneck with 3 convolutions
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):  # ch_in, ch_out, number, shortcut, groups, expansion
        super(C4_new, self).__init__()
        self.first_level=C3_t(c1,c1)
        self.GAP=nn.AdaptiveAvgPool2d(1)
        self.conv1=Conv(c1,c1,1)
        self.conv1_t=Conv(c1,c1,1)
        self.conv2=Conv(c1,c1,1)
        self.conv2_t=Conv(c1,c1,1)
        self.x_tr=nn.Conv2d(c1,c1,kernel_size=1,stride=1,padding=0,bias=True)
        self.x_t_tr=nn.Conv2d(c1,c1,kernel_size=1,stride=1,padding=0,bias=True)
        self.tr_alpha=nn.Conv2d(2*c1,c1,kernel_size=1,stride=1,padding=0,bias=True)
        self.tr_beta=nn.Conv2d(2*c1,c1,kernel_size=1,stride=1,padding=0,bias=True)
        self.relu=nn.ReLU(inplace=True)
        self.x_tr_2=nn.Conv2d(c1,c1,kernel_size=1,stride=1,padding=0,bias=True)
        self.x_t_tr_2=nn.Conv2d(c1,c1,kernel_size=1,stride=1,padding=0,bias=True)
        self.x_feat_fuse=nn.Conv2d(c1,c1,kernel_size=1,stride=1,padding=0,bias=True)
        self.y_feat_fuse=nn.Conv2d(c1,c1,kernel_size=1,stride=1,padding=0,bias=True)
        self.tr_alpha_copy=nn.Conv2d(2*c1,c1,kernel_size=1,stride=1,padding=0,bias=True)
        self.tr_beta_copy=nn.Conv2d(2*c1,c1,kernel_size=1,stride=1,padding=0,bias=True)
        self.cv3 = Conv(2 * c1, c2, 1)  # act=FReLU(c2)

    def forward(self,x,y):
        x,y=self.first_level(x,y)
        x_tr=self.x_tr(x)
        x_tr=self.relu(x_tr)
        y_tr=self.x_t_tr(y)
        y_tr=self.relu(y_tr)
        cond=torch.cat((x_tr,y_tr),1)
        alpha=self.tr_alpha(cond)
        beta=self.tr_beta(cond)
        alpha_copy=self.tr_alpha_copy(cond)
        beta_copy=self.tr_beta_copy(cond)
        y_feat=self.x_tr_2(y)
        y_feat=(alpha+1)*y_feat+beta
        y_feat=self.relu(y_feat)
        y_feat_fuse=self.y_feat_fuse(y_feat)
        x_feat=self.x_tr_2(x)
        x_feat=(alpha_copy+1)*x_feat+beta_copy
        x_feat=self.relu(x_feat)
        x_feat_fuse=self.x_feat_fuse(x_feat)
#         y=y+0.1*x_feat_fuse
#         x=x+0.1*y_feat_fuse
        x=y_feat_fuse
        y=x_feat_fuse
        fdx=x-y#N,C,H,W
        vx=F.tanh(self.GAP(fdx))#N,C,1,1
        x_res=x+vx*y
        fdy=y-x
        vy=F.tanh(self.GAP(fdy))
        y_res=y+vy*x
        x=self.conv1(x)
        x_res=self.conv2(x_res)
        y=self.conv1_t(y)
        y_res=self.conv2_t(y_res)
        x=x+x_res
        y=y+y_res
        return self.cv3(torch.cat((x,y), dim=1))
class C4_newv2(nn.Module):
    # CSP Bottleneck with 3 convolutions
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):  # ch_in, ch_out, number, shortcut, groups, expansion
        super(C4_newv2, self).__init__()
        self.first_level=C3_t(c1,c1)
        self.x_tr=nn.Conv2d(c1,c1,kernel_size=1,stride=1,padding=0,bias=True)
        self.x_t_tr=nn.Conv2d(c1,c1,kernel_size=1,stride=1,padding=0,bias=True)
        self.tr_alpha=nn.Conv2d(2*c1,c1,kernel_size=1,stride=1,padding=0,bias=True)
        self.tr_beta=nn.Conv2d(2*c1,c1,kernel_size=1,stride=1,padding=0,bias=True)
        self.relu=nn.ReLU(inplace=True)
        self.x_tr_2=nn.Conv2d(c1,c1,kernel_size=1,stride=1,padding=0,bias=True)
        self.x_t_tr_2=nn.Conv2d(c1,c1,kernel_size=1,stride=1,padding=0,bias=True)
        self.x_feat_fuse=nn.Conv2d(c1,c1,kernel_size=1,stride=1,padding=0,bias=True)
        self.y_feat_fuse=nn.Conv2d(c1,c1,kernel_size=1,stride=1,padding=0,bias=True)
        self.tr_alpha_copy=nn.Conv2d(2*c1,c1,kernel_size=1,stride=1,padding=0,bias=True)
        self.tr_beta_copy=nn.Conv2d(2*c1,c1,kernel_size=1,stride=1,padding=0,bias=True)
        self.cv3 = Conv(2 * c1, c2, 1)  # act=FReLU(c2)

    def forward(self,x,y):
        x,y=self.first_level(x,y)
        x_tr=self.x_tr(x)
        x_tr=self.relu(x_tr)
        y_tr=self.x_t_tr(y)
        y_tr=self.relu(y_tr)
        cond=torch.cat((x_tr,y_tr),1)
        alpha=self.tr_alpha(cond)
        beta=self.tr_beta(cond)
        alpha_copy=self.tr_alpha_copy(cond)
        beta_copy=self.tr_beta_copy(cond)
        y_feat=self.x_tr_2(y)
        y_feat=(alpha+1)*y_feat+beta
        y_feat=self.relu(y_feat)
        y_feat_fuse=self.y_feat_fuse(y_feat)
        x_feat=self.x_tr_2(x)
        x_feat=(alpha_copy+1)*x_feat+beta_copy
        x_feat=self.relu(x_feat)
        x_feat_fuse=self.x_feat_fuse(x_feat)
        x=y_feat_fuse
        y=x_feat_fuse
        return self.cv3(torch.cat((x,y), dim=1))
class C4_newv1(nn.Module):
    # CSP Bottleneck with 3 convolutions
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):  # ch_in, ch_out, number, shortcut, groups, expansion
        super(C4_newv1, self).__init__()
        self.first_level=C3_t(c1,c1)
        self.GAP=nn.AdaptiveAvgPool2d(1)
        self.conv1=Conv(c1,c1,1)
        self.conv1_t=Conv(c1,c1,1)
        self.conv2=Conv(c1,c1,1)
        self.conv2_t=Conv(c1,c1,1)
        self.cv3 = Conv(2 * c1, c2, 1)  # act=FReLU(c2)

    def forward(self,x,y):
        x,y=self.first_level(x,y)
        fdx=x-y#N,C,H,W
        vx=F.tanh(self.GAP(fdx))#N,C,1,1
        x_res=x+vx*y
        fdy=y-x
        vy=F.tanh(self.GAP(fdy))
        y_res=y+vy*x
        x=self.conv1(x)
        x_res=self.conv2(x_res)
        y=self.conv1_t(y)
        y_res=self.conv2_t(y_res)
        x=x+x_res
        y=y+y_res
        return self.cv3(torch.cat((x,y), dim=1))
class C4_newv3(nn.Module):
    # CSP Bottleneck with 3 convolutions
    def __init__(self, channels, c2, n=1, shortcut=True, g=1, e=0.5):  # ch_in, ch_out, number, shortcut, groups, expansion
        super(C4_newv3, self).__init__()
        inter_channels = int(channels // 4)

        self.local_att = nn.Sequential(
            nn.Conv2d(channels, inter_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(inter_channels, channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(channels),
        )

        self.global_att = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, inter_channels, kernel_size=1, stride=1, padding=0),
            nn.ReLU(inplace=True),
            nn.Conv2d(inter_channels, channels, kernel_size=1, stride=1, padding=0),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, y):
        xa = x + y
        xl = self.local_att(xa)
        xg = self.global_att(xa)

        xlg = xl + xg
        wei = self.sigmoid(xlg)

        xo = 2 * x * wei + 2 * y * (1 - wei)
        return xo
class C4_newv4(nn.Module):
    # CSP Bottleneck with 3 convolutions
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):  # ch_in, ch_out, number, shortcut, groups, expansion
        super(C4_newv4, self).__init__()
        inter_channels = int(c1 // 4)

        self.local_att = nn.Sequential(
            nn.Conv2d(c1, inter_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(inter_channels, c1, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(c1),
        )

        self.global_att = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(c1, inter_channels, kernel_size=1, stride=1, padding=0),
            nn.ReLU(inplace=True),
            nn.Conv2d(inter_channels, c1, kernel_size=1, stride=1, padding=0),
        )
        self.sigmoid = nn.Sigmoid()
        self.x_tr=nn.Conv2d(c1,c1,kernel_size=1,stride=1,padding=0,bias=True)
        self.x_t_tr=nn.Conv2d(c1,c1,kernel_size=1,stride=1,padding=0,bias=True)
        self.tr_alpha=nn.Conv2d(2*c1,c1,kernel_size=1,stride=1,padding=0,bias=True)
        self.tr_beta=nn.Conv2d(2*c1,c1,kernel_size=1,stride=1,padding=0,bias=True)
        self.relu=nn.ReLU(inplace=True)
        self.x_tr_2=nn.Conv2d(c1,c1,kernel_size=1,stride=1,padding=0,bias=True)
        self.x_t_tr_2=nn.Conv2d(c1,c1,kernel_size=1,stride=1,padding=0,bias=True)
        self.x_feat_fuse=nn.Conv2d(c1,c1,kernel_size=1,stride=1,padding=0,bias=True)
        self.y_feat_fuse=nn.Conv2d(c1,c1,kernel_size=1,stride=1,padding=0,bias=True)
        self.tr_alpha_copy=nn.Conv2d(2*c1,c1,kernel_size=1,stride=1,padding=0,bias=True)
        self.tr_beta_copy=nn.Conv2d(2*c1,c1,kernel_size=1,stride=1,padding=0,bias=True)
    def forward(self, x, y):
        x_tr=self.x_tr(x)
        x_tr=self.relu(x_tr)
        y_tr=self.x_t_tr(y)
        y_tr=self.relu(y_tr)
        cond=torch.cat((x_tr,y_tr),1)
        alpha=self.tr_alpha(cond)
        beta=self.tr_beta(cond)
        alpha_copy=self.tr_alpha_copy(cond)
        beta_copy=self.tr_beta_copy(cond)
        y_feat=self.x_tr_2(y)
        y_feat=(alpha+1)*y_feat+beta
        y_feat=self.relu(y_feat)
        y_feat_fuse=self.y_feat_fuse(y_feat)
        x_feat=self.x_tr_2(x)
        x_feat=(alpha_copy+1)*x_feat+beta_copy
        x_feat=self.relu(x_feat)
        x_feat_fuse=self.x_feat_fuse(x_feat)
#         y=y+0.1*x_feat_fuse
#         x=x+0.1*y_feat_fuse
        x=y_feat_fuse
        y=x_feat_fuse
        xa = x + y
        xl = self.local_att(xa)
        xg = self.global_att(xa)

        xlg = xl + xg
        wei = self.sigmoid(xlg)

        xo = 2 * x * wei + 2 * y * (1 - wei)
        return xo
class SPP(nn.Module):
    # Spatial pyramid pooling layer used in YOLOv3-SPP
    def __init__(self, c1, c2, k=(5, 9, 13)):
        super(SPP, self).__init__()
        c_ = c1 // 2  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_ * (len(k) + 1), c2, 1, 1)
        self.m = nn.ModuleList([nn.MaxPool2d(kernel_size=x, stride=1, padding=x // 2) for x in k])

    def forward(self, x):
        x = self.cv1(x)
        return self.cv2(torch.cat([x] + [m(x) for m in self.m], 1))
class SPP_t(nn.Module):
    def __init__(self, c1, c2, k=(5, 9, 13)):#channel_in,channel_out,kernel
        super(SPP_t, self).__init__()
        c_ = c1 // 2  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_ * (len(k) + 1), c2, 1, 1)
        self.m = nn.ModuleList([nn.MaxPool2d(kernel_size=x, stride=1, padding=x // 2) for x in k])
        self.cv1_t=Conv(c1, c_, 1, 1)
        self.cv2_t=Conv(c_ * (len(k) + 1), c2, 1, 1)
    def forward(self, x,y):
        x = self.cv1(x)
        y=self.cv1_t(y)
        return self.cv2(torch.cat([x] + [m(x) for m in self.m], 1)),self.cv2_t(torch.cat([y] + [m(y) for m in self.m], 1))
# x(b,c,w,h)-->y(b,4c,w/2,h/2)
class multi_mix_t(nn.Module):
    def __init__(self,c1,c2):#c1,c2用不到，因为输入和输出通道数是完全一样的，无需改变通道数。为了模型保持一致所以这里加上。
        super(multi_mix_t,self).__init__()
        self.GAP=nn.AdaptiveAvgPool2d(1)
    def forward(self,x,y):
        fdx=x-y#N,C,H,W
        vx=F.tanh(self.GAP(fdx))#N,C,1,1
        x_res=x+vx*y
        fdy=y-x
        vy=F.tanh(self.GAP(fdy))
        y_res=y+vy*x
        return x_res,y_res
class multi_mix_CBL_t(nn.Module):
    def __init__(self,c1,c2):#c1,c2用不到，因为输入和输出通道数是完全一样的，无需改变通道数。为了模型保持一致所以这里加上。
        super(multi_mix_CBL_t,self).__init__()
        self.GAP=nn.AdaptiveAvgPool2d(1)
        self.conv1=Conv(c1,2*c1,1,2)
        self.conv1_t=Conv(c1,2*c1,1,2)
        self.conv2=Conv(c1,2*c1,3,2)
        self.conv2_t=Conv(c1,2*c1,3,2)
    def forward(self,x,y):
        fdx=x-y#N,C,H,W
        vx=F.tanh(self.GAP(fdx))#N,C,1,1
        x_res=x+vx*y
        fdy=y-x
        vy=F.tanh(self.GAP(fdy))
        y_res=y+vy*x
        x=self.conv1(x)
        x_res=self.conv2(x_res)
        y=self.conv1_t(y)
        y_res=self.conv2_t(y_res)
        return x+x_res,y+y_res
class DeformConv2D(nn.Module):
    def __init__(self, inc, outc, kernel_size=3, padding=1, bias=None):#channel_in,channel_out,kernel,padding,bias
        super(DeformConv2D, self).__init__()
        self.kernel_size = kernel_size
        self.padding = padding
        self.zero_padding = nn.ZeroPad2d(padding)
        self.conv_kernel = nn.Conv2d(inc, outc, kernel_size=kernel_size, stride=kernel_size, bias=bias)

    def forward(self, x, offset):
        dtype = offset.data.type()
        ks = self.kernel_size
        N = offset.size(1) // 2

        # Change offset's order from [x1, x2, ..., y1, y2, ...] to [x1, y1, x2, y2, ...]
        # Codes below are written to make sure same results of MXNet implementation.
        # You can remove them, and it won't influence the module's performance.
        offsets_index = Variable(torch.cat([torch.arange(0, 2*N, 2), torch.arange(1, 2*N+1, 2)]), requires_grad=False).type_as(x).long()
        offsets_index = offsets_index.unsqueeze(dim=0).unsqueeze(dim=-1).unsqueeze(dim=-1).expand(*offset.size())
        offset = torch.gather(offset, dim=1, index=offsets_index)
        # ------------------------------------------------------------------------

        if self.padding:
            x = self.zero_padding(x)

        # (b, 2N, h, w)
        p = self._get_p(offset, dtype)

        # (b, h, w, 2N)
        p = p.contiguous().permute(0, 2, 3, 1)
        q_lt = Variable(p.data, requires_grad=False).floor()
        q_rb = q_lt + 1

        q_lt = torch.cat([torch.clamp(q_lt[..., :N], 0, x.size(2)-1), torch.clamp(q_lt[..., N:], 0, x.size(3)-1)], dim=-1).long()
        q_rb = torch.cat([torch.clamp(q_rb[..., :N], 0, x.size(2)-1), torch.clamp(q_rb[..., N:], 0, x.size(3)-1)], dim=-1).long()
        q_lb = torch.cat([q_lt[..., :N], q_rb[..., N:]], -1)
        q_rt = torch.cat([q_rb[..., :N], q_lt[..., N:]], -1)

        # (b, h, w, N)
        mask = torch.cat([p[..., :N].lt(self.padding)+p[..., :N].gt(x.size(2)-1-self.padding),
                          p[..., N:].lt(self.padding)+p[..., N:].gt(x.size(3)-1-self.padding)], dim=-1).type_as(p)
        mask = mask.detach()
        floor_p = p - (p - torch.floor(p))
        p = p*(1-mask) + floor_p*mask
        p = torch.cat([torch.clamp(p[..., :N], 0, x.size(2)-1), torch.clamp(p[..., N:], 0, x.size(3)-1)], dim=-1)

        # bilinear kernel (b, h, w, N)
        g_lt = (1 + (q_lt[..., :N].type_as(p) - p[..., :N])) * (1 + (q_lt[..., N:].type_as(p) - p[..., N:]))
        g_rb = (1 - (q_rb[..., :N].type_as(p) - p[..., :N])) * (1 - (q_rb[..., N:].type_as(p) - p[..., N:]))
        g_lb = (1 + (q_lb[..., :N].type_as(p) - p[..., :N])) * (1 - (q_lb[..., N:].type_as(p) - p[..., N:]))
        g_rt = (1 - (q_rt[..., :N].type_as(p) - p[..., :N])) * (1 + (q_rt[..., N:].type_as(p) - p[..., N:]))

        # (b, c, h, w, N)
        x_q_lt = self._get_x_q(x, q_lt, N)
        x_q_rb = self._get_x_q(x, q_rb, N)
        x_q_lb = self._get_x_q(x, q_lb, N)
        x_q_rt = self._get_x_q(x, q_rt, N)

        # (b, c, h, w, N)
        x_offset = g_lt.unsqueeze(dim=1) * x_q_lt + \
                   g_rb.unsqueeze(dim=1) * x_q_rb + \
                   g_lb.unsqueeze(dim=1) * x_q_lb + \
                   g_rt.unsqueeze(dim=1) * x_q_rt

        x_offset = self._reshape_x_offset(x_offset, ks)
        out = self.conv_kernel(x_offset)

        return out

    def _get_p_n(self, N, dtype):
        p_n_x, p_n_y = np.meshgrid(range(-(self.kernel_size-1)//2, (self.kernel_size-1)//2+1),
                          range(-(self.kernel_size-1)//2, (self.kernel_size-1)//2+1), indexing='ij')
        # (2N, 1)
        p_n = np.concatenate((p_n_x.flatten(), p_n_y.flatten()))
        p_n = np.reshape(p_n, (1, 2*N, 1, 1))
        p_n = Variable(torch.from_numpy(p_n).type(dtype), requires_grad=False)

        return p_n

    @staticmethod
    def _get_p_0(h, w, N, dtype):
        p_0_x, p_0_y = np.meshgrid(range(1, h+1), range(1, w+1), indexing='ij')
        p_0_x = p_0_x.flatten().reshape(1, 1, h, w).repeat(N, axis=1)
        p_0_y = p_0_y.flatten().reshape(1, 1, h, w).repeat(N, axis=1)
        p_0 = np.concatenate((p_0_x, p_0_y), axis=1)
        p_0 = Variable(torch.from_numpy(p_0).type(dtype), requires_grad=False)

        return p_0

    def _get_p(self, offset, dtype):
        N, h, w = offset.size(1)//2, offset.size(2), offset.size(3)

        # (1, 2N, 1, 1)
        p_n = self._get_p_n(N, dtype)
        # (1, 2N, h, w)
        p_0 = self._get_p_0(h, w, N, dtype)
        p = p_0 + p_n + offset
        return p

    def _get_x_q(self, x, q, N):
        b, h, w, _ = q.size()
        padded_w = x.size(3)
        c = x.size(1)
        # (b, c, h*w)
        x = x.contiguous().view(b, c, -1)

        # (b, h, w, N)
        index = q[..., :N]*padded_w + q[..., N:]  # offset_x*w + offset_y
        # (b, c, h*w*N)
        index = index.contiguous().unsqueeze(dim=1).expand(-1, c, -1, -1, -1).contiguous().view(b, c, -1)

        x_offset = x.gather(dim=-1, index=index).contiguous().view(b, c, h, w, N)

        return x_offset

    @staticmethod
    def _reshape_x_offset(x_offset, ks):
        b, c, h, w, N = x_offset.size()
        x_offset = torch.cat([x_offset[..., s:s+ks].contiguous().view(b, c, h, w*ks) for s in range(0, N, ks)], dim=-1)
        x_offset = x_offset.contiguous().view(b, c, h*ks, w*ks)

        return x_offset
class multi_mix_CBL_PLARD_t(nn.Module):
    def __init__(self,c1,c2):#c1,c2用不到，因为输入和输出通道数是完全一样的，无需改变通道数。为了模型保持一致所以这里加上。
        super(multi_mix_CBL_PLARD_t,self).__init__()
        self.GAP=nn.AdaptiveAvgPool2d(1)
        self.conv1=Conv(c1,2*c1,1,2)
        self.conv1_t=Conv(c1,2*c1,1,2)
        self.conv2=Conv(c1,2*c1,3,2)
        self.conv2_t=Conv(c1,2*c1,3,2)
        self.x_tr=nn.Conv2d(c1,c1,kernel_size=1,stride=1,padding=0,bias=True)
        self.x_t_tr=nn.Conv2d(c1,c1,kernel_size=1,stride=1,padding=0,bias=True)
        self.tr_alpha=nn.Conv2d(2*c1,c1,kernel_size=1,stride=1,padding=0,bias=True)
        self.tr_beta=nn.Conv2d(2*c1,c1,kernel_size=1,stride=1,padding=0,bias=True)
        self.relu=nn.ReLU(inplace=True)
        self.x_tr_2=nn.Conv2d(c1,c1,kernel_size=1,stride=1,padding=0,bias=True)
        self.x_t_tr_2=nn.Conv2d(c1,c1,kernel_size=1,stride=1,padding=0,bias=True)
        self.x_feat_fuse=nn.Conv2d(c1,c1,kernel_size=1,stride=1,padding=0,bias=True)
        self.y_feat_fuse=nn.Conv2d(c1,c1,kernel_size=1,stride=1,padding=0,bias=True)
        self.tr_alpha_copy=nn.Conv2d(2*c1,c1,kernel_size=1,stride=1,padding=0,bias=True)
        self.tr_beta_copy=nn.Conv2d(2*c1,c1,kernel_size=1,stride=1,padding=0,bias=True)
    def forward(self,x,y):
        x_tr=self.x_tr(x)
        x_tr=self.relu(x_tr)
        y_tr=self.x_t_tr(y)
        y_tr=self.relu(y_tr)
        cond=torch.cat((x_tr,y_tr),1)
        alpha=self.tr_alpha(cond)
        beta=self.tr_beta(cond)
        alpha_copy=self.tr_alpha_copy(cond)
        beta_copy=self.tr_beta_copy(cond)
        y_feat=self.x_tr_2(y)
        y_feat=(alpha+1)*y_feat+beta
        y_feat=self.relu(y_feat)
        y_feat_fuse=self.y_feat_fuse(y_feat)
        x_feat=self.x_tr_2(x)
        x_feat=(alpha_copy+1)*x_feat+beta_copy
        x_feat=self.relu(x_feat)
        x_feat_fuse=self.x_feat_fuse(x_feat)
        y=y+0.1*x_feat_fuse
        x=x+0.1*y_feat_fuse
        fdx=x-y#N,C,H,W
        vx=F.tanh(self.GAP(fdx))#N,C,1,1
        x_res=x+vx*y
        fdy=y-x
        vy=F.tanh(self.GAP(fdy))
        y_res=y+vy*x
        x=self.conv1(x)
        x_res=self.conv2(x_res)
        y=self.conv1_t(y)
        y_res=self.conv2_t(y_res)
        return x+x_res,y+y_res
class twoConvSum(nn.Module):
    def __init__(self,H,W):
        super(twoConvSum, self).__init__()
        self.convW = Conv(2, 2,(1,W),1)
        self.convH = Conv(2, 2, (H, 1), 1)
        self.final_conv=nn.Sequential(
            nn.Conv2d(2,1,1),
            nn.Sigmoid()
        )
    def forward(self,x):
        x=torch.cat( (torch.max(x,1)[0].unsqueeze(1), torch.mean(x,1).unsqueeze(1)), dim=1 )
        print(x.size())
        print('***********')
        for i in range(2):
            this=torch.cat((x,x[:,:,:,:]),dim=3)
            print(this.size())
            print('***********')
            x=self.convW(this)
            x=self.convH(torch.cat((x,x[:,:,:,:]),dim=2))
        x=self.final_conv(x)
        return x
class multi_mix_CBL_PLARD_WH_t(nn.Module):
    def __init__(self,c1,c2):#c1,c2用不到，因为输入和输出通道数是完全一样的，无需改变通道数。为了模型保持一致所以这里加上。
        super(multi_mix_CBL_PLARD_WH_t,self).__init__()
        self.GAP=nn.AdaptiveAvgPool2d(1)
        self.conv1=Conv(c1,2*c1,1,2)
        self.conv1_t=Conv(c1,2*c1,1,2)
        self.conv2=Conv(c1,2*c1,3,2)
        self.conv2_t=Conv(c1,2*c1,3,2)
        self.x_tr=nn.Conv2d(c1,c1,kernel_size=1,stride=1,padding=0,bias=True)
        self.x_t_tr=nn.Conv2d(c1,c1,kernel_size=1,stride=1,padding=0,bias=True)
        self.tr_alpha=nn.Conv2d(2*c1,c1,kernel_size=1,stride=1,padding=0,bias=True)
        self.tr_beta=nn.Conv2d(2*c1,c1,kernel_size=1,stride=1,padding=0,bias=True)
        self.relu=nn.ReLU(inplace=True)
        self.x_tr_2=nn.Conv2d(c1,c1,kernel_size=1,stride=1,padding=0,bias=True)
        self.x_t_tr_2=nn.Conv2d(c1,c1,kernel_size=1,stride=1,padding=0,bias=True)
        self.x_feat_fuse=nn.Conv2d(c1,c1,kernel_size=1,stride=1,padding=0,bias=True)
        self.y_feat_fuse=nn.Conv2d(c1,c1,kernel_size=1,stride=1,padding=0,bias=True)
        self.tr_alpha_copy=nn.Conv2d(2*c1,c1,kernel_size=1,stride=1,padding=0,bias=True)
        self.tr_beta_copy=nn.Conv2d(2*c1,c1,kernel_size=1,stride=1,padding=0,bias=True)
        self.wh_conv_alpha=Conv(2*c1,2*c1,1,1)
        self.wh_conv_alpha_copy=Conv(2*c1,2*c1,1,1)
    def forward(self,x,y):
        W,H=x.size()[2],x.size()[-1]
        x_tr=self.x_tr(x)
        x_tr=self.relu(x_tr)
        y_tr=self.x_t_tr(y)
        y_tr=self.relu(y_tr)
        cond=torch.cat((x_tr,y_tr),1)
#         device = torch.device('cuda')
#         cond=cond.to(device)
#         cond.to(torch.float16)
        cond1=self.wh_conv_alpha_copy(cond)
        alpha=self.tr_alpha(cond1)
        beta=self.tr_beta(cond1)
        cond2=self.wh_conv_alpha_copy(cond)
        alpha_copy=self.tr_alpha_copy(cond2)
        beta_copy=self.tr_beta_copy(cond2)
        y_feat=self.x_tr_2(y)
        y_feat=(alpha+1)*y_feat+beta
        y_feat=self.relu(y_feat)
        y_feat_fuse=self.y_feat_fuse(y_feat)
        x_feat=self.x_tr_2(x)
        x_feat=(alpha_copy+1)*x_feat+beta_copy
        x_feat=self.relu(x_feat)
        x_feat_fuse=self.x_feat_fuse(x_feat)
        y=y+0.1*x_feat_fuse
        x=x+0.1*y_feat_fuse
        fdx=x-y#N,C,H,W
        vx=F.tanh(self.GAP(fdx))#N,C,1,1
        x_res=x+vx*y
        fdy=y-x
        vy=F.tanh(self.GAP(fdy))
        y_res=y+vy*x
        x=self.conv1(x)
        x_res=self.conv2(x_res)
        y=self.conv1_t(y)
        y_res=self.conv2_t(y_res)
        return x+x_res,y+y_res
class multi_mix_CBL_PLARD_deformal_t(nn.Module):
    def __init__(self,c1,c2):#c1,c2用不到，因为输入和输出通道数是完全一样的，无需改变通道数。为了模型保持一致所以这里加上。
        super(multi_mix_CBL_PLARD_deformal_t,self).__init__()
#         self.conv_alpha = nn.Conv2d(
#         2*c1,
#         1* 2 * 3 * 3,
#         kernel_size=(3, 3),
#         stride=(1, 1),
#         padding=(1, 1),
#         bias=False).cuda()
#         self.conv_beta=nn.Conv2d(
#         2*c1,
#         1* 2 * 3 * 3,
#         kernel_size=(3, 3),
#         stride=(1, 1),
#         padding=(1, 1),
#         bias=False).cuda()
#         self.conv_alpha_copy = nn.Conv2d(
#         2*c1,
#         1* 2 * 3 * 3,
#         kernel_size=(3, 3),
#         stride=(1, 1),
#         padding=(1, 1),
#         bias=False).cuda()
#         self.conv_beta_copy=nn.Conv2d(
#         2*c1,
#         1* 2 * 3 * 3,
#         kernel_size=(3, 3),
#         stride=(1, 1),
#         padding=(1, 1),
#         bias=False).cuda()
        self.conv_alpha = nn.Conv2d(
        2*c1,
        1* 2 * 1 * 1,
        kernel_size=(1, 1),
        stride=(1, 1),
        padding=(0, 0),
        bias=False).cuda()
        self.conv_beta=nn.Conv2d(
        2*c1,
        1* 2 * 1 * 1,
        kernel_size=(1, 1),
        stride=(1, 1),
        padding=(0, 0),
        bias=False).cuda()
        self.conv_alpha_copy = nn.Conv2d(
        2*c1,
        1* 2 * 1 * 1,
        kernel_size=(1, 1),
        stride=(1, 1),
        padding=(0, 0),
        bias=False).cuda()
        self.conv_beta_copy=nn.Conv2d(
        2*c1,
        1* 2 * 1 * 1,
        kernel_size=(1, 1),
        stride=(1, 1),
        padding=(0, 0),
        bias=False).cuda()
        self.GAP=nn.AdaptiveAvgPool2d(1)
        self.conv1=Conv(c1,2*c1,1,2)
        self.conv1_t=Conv(c1,2*c1,1,2)
        self.conv2=Conv(c1,2*c1,3,2)
        self.conv2_t=Conv(c1,2*c1,3,2)
        self.x_tr=nn.Conv2d(c1,c1,kernel_size=1,stride=1,padding=0,bias=True)
        self.x_t_tr=nn.Conv2d(c1,c1,kernel_size=1,stride=1,padding=0,bias=True)
#         self.tr_alpha=DeformConv2D(2*c1,c1)
#         self.tr_beta=DeformConv2D(2*c1,c1)
        self.tr_alpha=DeformConv2D(2*c1,c1,1,0)
        self.tr_beta=DeformConv2D(2*c1,c1,1,0)
        self.relu=nn.ReLU(inplace=True)
        self.x_tr_2=nn.Conv2d(c1,c1,kernel_size=1,stride=1,padding=0,bias=True)
        self.x_t_tr_2=nn.Conv2d(c1,c1,kernel_size=1,stride=1,padding=0,bias=True)
        self.x_feat_fuse=nn.Conv2d(c1,c1,kernel_size=1,stride=1,padding=0,bias=True)
        self.y_feat_fuse=nn.Conv2d(c1,c1,kernel_size=1,stride=1,padding=0,bias=True)
        self.tr_alpha_copy=DeformConv2D(2*c1,c1,1,0)
        self.tr_beta_copy=DeformConv2D(2*c1,c1,1,0)
#         self.tr_alpha_copy=DeformConv2D(2*c1,c1)
#         self.tr_beta_copy=DeformConv2D(2*c1,c1)
    def forward(self,x,y):
        x_tr=self.x_tr(x)
        x_tr=self.relu(x_tr)
        y_tr=self.x_t_tr(y)
        y_tr=self.relu(y_tr)
        cond=torch.cat((x_tr,y_tr),1)
        alpha_offset=self.conv_alpha(cond)
        alpha=self.tr_alpha(cond,alpha_offset)
        beta_offset=self.conv_beta(cond)
        beta=self.tr_beta(cond,beta_offset)
        alpha_offset_copy=self.conv_alpha_copy(cond)
        alpha_copy=self.tr_alpha_copy(cond,alpha_offset_copy)
        beta_offset_copy=self.conv_beta_copy(cond)
        beta_copy=self.tr_beta_copy(cond,beta_offset_copy)
        y_feat=self.x_tr_2(y)
        y_feat=(alpha+1)*y_feat+beta
        y_feat=self.relu(y_feat)
        y_feat_fuse=self.y_feat_fuse(y_feat)
        x_feat=self.x_tr_2(x)
        x_feat=(alpha_copy+1)*x_feat+beta_copy
        x_feat=self.relu(x_feat)
        x_feat_fuse=self.x_feat_fuse(x_feat)
        y=y+0.1*x_feat_fuse
        x=x+0.1*y_feat_fuse
        fdx=x-y#N,C,H,W
        vx=F.tanh(self.GAP(fdx))#N,C,1,1
        x_res=x+vx*y
        fdy=y-x
        vy=F.tanh(self.GAP(fdy))
        y_res=y+vy*x
        x=self.conv1(x)
        x_res=self.conv2(x_res)
        y=self.conv1_t(y)
        y_res=self.conv2_t(y_res)
        return x+x_res,y+y_res
class multi_mix_CBL_STN_t(nn.Module):
    def __init__(self,c1,c2):#c1,c2用不到，因为输入和输出通道数是完全一样的，无需改变通道数。为了模型保持一致所以这里加上。
        super(multi_mix_CBL_STN_t,self).__init__()
        self.GAP=nn.AdaptiveAvgPool2d(1)
        self.conv1=Conv(c1,2*c1,1,2)
        self.conv1_t=Conv(c1,2*c1,1,2)
        self.conv2=Conv(c1,2*c1,3,2)
        self.conv2_t=Conv(c1,2*c1,3,2)
        self.fc_x1=nn.Linear(1024,128)
        self.fc_x2=nn.Linear(128,6)
        self.conv_x=nn.Conv2d(1,1024,1,1)
        self.fc_y1=nn.Linear(1024,128)
        self.fc_y2=nn.Linear(128,6)
        self.conv_y=nn.Conv2d(1,1024,1,1)
        self.B=16
        self.C=64
        self.H=160
        self.W=160
    def forward(self,x,y):
        B,C,H,W=x.size()
        cond=torch.cat((x,y),1)
        batch_imgs=x
        theta=cond.mean(1)
        theta=theta.unsqueeze(1)
        theta=self.conv_x(theta)
        theta=self.GAP(theta)
        theta=theta.squeeze()
        theta=self.fc_x1(theta)
        theta=self.fc_x2(theta)
        theta=theta.view(-1,2,3)
        affine_grid_points=F.affine_grid(theta,torch.Size((self.B,self.C,self.H,self.W)))
        rois=F.grid_sample(batch_imgs,affine_grid_points)
        x=rois
        batch_imgs_t=y
        theta=cond.mean(1)
        theta=theta.unsqueeze(1)
        theta=self.conv_y(theta)
        theta=self.GAP(theta)
        theta=theta.squeeze()
        theta=self.fc_y1(theta)
        theta=self.fc_y2(theta)
        theta=theta.view(-1,2,3)
        affine_grid_points=F.affine_grid(theta,torch.Size((self.B,self.C,self.H,self.W)))
        rois=F.grid_sample(batch_imgs,affine_grid_points)
        y=rois
        fdx=x-y#N,C,H,W
        vx=F.tanh(self.GAP(fdx))#N,C,1,1
        x_res=x+vx*y
        fdy=y-x
        vy=F.tanh(self.GAP(fdy))
        y_res=y+vy*x
        x=self.conv1(x)
        x_res=self.conv2(x_res)
        y=self.conv1_t(y)
        y_res=self.conv2_t(y_res)
        return x+x_res,y+y_res
class multi_mix_CBL_STN_t1(nn.Module):
    def __init__(self,c1,c2):#c1,c2用不到，因为输入和输出通道数是完全一样的，无需改变通道数。为了模型保持一致所以这里加上。
        super(multi_mix_CBL_STN_t1,self).__init__()
        self.GAP=nn.AdaptiveAvgPool2d(1)
        self.conv1=Conv(c1,2*c1,1,2)
        self.conv1_t=Conv(c1,2*c1,1,2)
        self.conv2=Conv(c1,2*c1,3,2)
        self.conv2_t=Conv(c1,2*c1,3,2)
        self.fc_x1=nn.Linear(1024,128)
        self.fc_x2=nn.Linear(128,6)
        self.conv_x=nn.Conv2d(1,1024,1,1)
        self.fc_y1=nn.Linear(1024,128)
        self.fc_y2=nn.Linear(128,6)
        self.conv_y=nn.Conv2d(1,1024,1,1)
        self.B=16
        self.C=128
        self.H=80
        self.W=80
    def forward(self,x,y):
        B,C,H,W=x.size()
        cond=torch.cat((x,y),1)
        batch_imgs=x
        theta=cond.mean(1)
        theta=theta.unsqueeze(1)
        theta=self.conv_x(theta)
        theta=self.GAP(theta)
        theta=theta.squeeze()
        theta=self.fc_x1(theta)
        theta=self.fc_x2(theta)
        theta=theta.view(-1,2,3)
        affine_grid_points=F.affine_grid(theta,torch.Size((self.B,self.C,self.H,self.W)))
        rois=F.grid_sample(batch_imgs,affine_grid_points)
        x=rois
        batch_imgs_t=y
        theta=cond.mean(1)
        theta=theta.unsqueeze(1)
        theta=self.conv_y(theta)
        theta=self.GAP(theta)
        theta=theta.squeeze()
        theta=self.fc_y1(theta)
        theta=self.fc_y2(theta)
        theta=theta.view(-1,2,3)
        affine_grid_points=F.affine_grid(theta,torch.Size((self.B,self.C,self.H,self.W)))
        rois=F.grid_sample(batch_imgs,affine_grid_points)
        y=rois
        fdx=x-y#N,C,H,W
        vx=F.tanh(self.GAP(fdx))#N,C,1,1
        x_res=x+vx*y
        fdy=y-x
        vy=F.tanh(self.GAP(fdy))
        y_res=y+vy*x
        x=self.conv1(x)
        x_res=self.conv2(x_res)
        y=self.conv1_t(y)
        y_res=self.conv2_t(y_res)
        return x+x_res,y+y_res
class multi_mix_CBL_STN_t2(nn.Module):
    def __init__(self,c1,c2):#c1,c2用不到，因为输入和输出通道数是完全一样的，无需改变通道数。为了模型保持一致所以这里加上。
        super(multi_mix_CBL_STN_t2,self).__init__()
        self.GAP=nn.AdaptiveAvgPool2d(1)
        self.conv1=Conv(c1,2*c1,1,2)
        self.conv1_t=Conv(c1,2*c1,1,2)
        self.conv2=Conv(c1,2*c1,3,2)
        self.conv2_t=Conv(c1,2*c1,3,2)
        self.fc_x1=nn.Linear(1024,128)
        self.fc_x2=nn.Linear(128,6)
        self.conv_x=nn.Conv2d(1,1024,1,1)
        self.fc_y1=nn.Linear(1024,128)
        self.fc_y2=nn.Linear(128,6)
        self.conv_y=nn.Conv2d(1,1024,1,1)
        self.B=16
        self.C=256
        self.H=40
        self.W=40
    def forward(self,x,y):
        B,C,H,W=x.size()
        cond=torch.cat((x,y),1)
        batch_imgs=x
        theta=cond.mean(1)
        theta=theta.unsqueeze(1)
        theta=self.conv_x(theta)
        theta=self.GAP(theta)
        theta=theta.squeeze()
        theta=self.fc_x1(theta)
        theta=self.fc_x2(theta)
        theta=theta.view(-1,2,3)
        affine_grid_points=F.affine_grid(theta,torch.Size((self.B,self.C,self.H,self.W)))
        rois=F.grid_sample(batch_imgs,affine_grid_points)
        x=rois
        batch_imgs_t=y
        theta=cond.mean(1)
        theta=theta.unsqueeze(1)
        theta=self.conv_y(theta)
        theta=self.GAP(theta)
        theta=theta.squeeze()
        theta=self.fc_y1(theta)
        theta=self.fc_y2(theta)
        theta=theta.view(-1,2,3)
        affine_grid_points=F.affine_grid(theta,torch.Size((self.B,self.C,self.H,self.W)))
        rois=F.grid_sample(batch_imgs,affine_grid_points)
        y=rois
        fdx=x-y#N,C,H,W
        vx=F.tanh(self.GAP(fdx))#N,C,1,1
        x_res=x+vx*y
        fdy=y-x
        vy=F.tanh(self.GAP(fdy))
        y_res=y+vy*x
        x=self.conv1(x)
        x_res=self.conv2(x_res)
        y=self.conv1_t(y)
        y_res=self.conv2_t(y_res)
        return x+x_res,y+y_res
class multi_mix_CBL_PLARD_modify_t(nn.Module):
    def __init__(self,c1,c2):#c1,c2用不到，因为输入和输出通道数是完全一样的，无需改变通道数。为了模型保持一致所以这里加上。
        super(multi_mix_CBL_PLARD_modify_t,self).__init__()
        self.GAP=nn.AdaptiveAvgPool2d(1)
        self.conv1=Conv(c1,2*c1,1,2)
        self.conv1_t=Conv(c1,2*c1,1,2)
        self.conv2=Conv(c1,2*c1,3,2)
        self.conv2_t=Conv(c1,2*c1,3,2)
        self.x_tr=nn.Conv2d(c1,c1,kernel_size=1,stride=1,padding=0,bias=True)
        self.x_t_tr=nn.Conv2d(c1,c1,kernel_size=1,stride=1,padding=0,bias=True)
        self.tr_alpha=nn.Conv2d(2*c1,c1,kernel_size=1,stride=1,padding=0,bias=True)
        self.tr_beta=nn.Conv2d(2*c1,c1,kernel_size=1,stride=1,padding=0,bias=True)
        self.relu=nn.ReLU(inplace=True)
        self.x_tr_2=nn.Conv2d(c1,c1,kernel_size=1,stride=1,padding=0,bias=True)
        self.x_t_tr_2=nn.Conv2d(c1,c1,kernel_size=1,stride=1,padding=0,bias=True)
        self.x_feat_fuse=nn.Conv2d(c1,c1,kernel_size=1,stride=1,padding=0,bias=True)
        self.y_feat_fuse=nn.Conv2d(c1,c1,kernel_size=1,stride=1,padding=0,bias=True)
        self.tr_alpha_copy=nn.Conv2d(2*c1,c1,kernel_size=1,stride=1,padding=0,bias=True)
        self.tr_beta_copy=nn.Conv2d(2*c1,c1,kernel_size=1,stride=1,padding=0,bias=True)
    def forward(self,x,y):
        x_tr=self.x_tr(x)
        x_tr=self.relu(x_tr)
        y_tr=self.x_t_tr(y)
        y_tr=self.relu(y_tr)
        cond=torch.cat((x_tr,y_tr),1)
        alpha=self.tr_alpha(cond)
        beta=self.tr_beta(cond)
        alpha_copy=self.tr_alpha_copy(cond)
        beta_copy=self.tr_beta_copy(cond)
        y_feat=self.x_tr_2(y)
        y_feat=(alpha+1)*y_feat+beta
        y_feat=self.relu(y_feat)
        y_feat_fuse=self.y_feat_fuse(y_feat)
        x_feat=self.x_tr_2(x)
        x_feat=(alpha_copy+1)*x_feat+beta_copy
        x_feat=self.relu(x_feat)
        x_feat_fuse=self.x_feat_fuse(x_feat)
#         y=y+0.1*x_feat_fuse
#         x=x+0.1*y_feat_fuse
        x=y_feat_fuse
        y=x_feat_fuse
        fdx=x-y#N,C,H,W
        vx=F.tanh(self.GAP(fdx))#N,C,1,1
        x_res=x+vx*y
        fdy=y-x
        vy=F.tanh(self.GAP(fdy))
        y_res=y+vy*x
        x=self.conv1(x)
        x_res=self.conv2(x_res)
        y=self.conv1_t(y)
        y_res=self.conv2_t(y_res)
        return x+x_res,y+y_res
class multi_mix_CBL_PLARD_modified_deformal_t(nn.Module):
    def __init__(self,c1,c2):#c1,c2用不到，因为输入和输出通道数是完全一样的，无需改变通道数。为了模型保持一致所以这里加上。
        super(multi_mix_CBL_PLARD_modified_deformal_t,self).__init__()
        self.conv_alpha = nn.Conv2d(
        2*c1,
        1* 2 * 1 * 1,
        kernel_size=(1, 1),
        stride=(1, 1),
        padding=(0, 0),
        bias=False).cuda()
        self.conv_beta=nn.Conv2d(
        2*c1,
        1* 2 * 1 * 1,
        kernel_size=(1, 1),
        stride=(1, 1),
        padding=(0, 0),
        bias=False).cuda()
        self.conv_alpha_copy = nn.Conv2d(
        2*c1,
        1* 2 * 1 * 1,
        kernel_size=(1, 1),
        stride=(1, 1),
        padding=(0, 0),
        bias=False).cuda()
        self.conv_beta_copy=nn.Conv2d(
        2*c1,
        1* 2 * 1 * 1,
        kernel_size=(1, 1),
        stride=(1, 1),
        padding=(0, 0),
        bias=False).cuda()
        self.GAP=nn.AdaptiveAvgPool2d(1)
        self.conv1=Conv(c1,2*c1,1,2)
        self.conv1_t=Conv(c1,2*c1,1,2)
        self.conv2=Conv(c1,2*c1,3,2)
        self.conv2_t=Conv(c1,2*c1,3,2)
        self.x_tr=nn.Conv2d(c1,c1,kernel_size=1,stride=1,padding=0,bias=True)
        self.x_t_tr=nn.Conv2d(c1,c1,kernel_size=1,stride=1,padding=0,bias=True)
        self.tr_alpha=DeformConv2D(2*c1,c1,1,0)
        self.tr_beta=DeformConv2D(2*c1,c1,1,0)
        self.relu=nn.ReLU(inplace=True)
        self.x_tr_2=nn.Conv2d(c1,c1,kernel_size=1,stride=1,padding=0,bias=True)
        self.x_t_tr_2=nn.Conv2d(c1,c1,kernel_size=1,stride=1,padding=0,bias=True)
        self.x_feat_fuse=nn.Conv2d(c1,c1,kernel_size=1,stride=1,padding=0,bias=True)
        self.y_feat_fuse=nn.Conv2d(c1,c1,kernel_size=1,stride=1,padding=0,bias=True)
        self.tr_alpha_copy=DeformConv2D(2*c1,c1,1,0)
        self.tr_beta_copy=DeformConv2D(2*c1,c1,1,0)
    def forward(self,x,y):
        x_tr=self.x_tr(x)
        x_tr=self.relu(x_tr)
        y_tr=self.x_t_tr(y)
        y_tr=self.relu(y_tr)
        cond=torch.cat((x_tr,y_tr),1)
        alpha_offset=self.conv_alpha(cond)
        alpha=self.tr_alpha(cond,alpha_offset)
        beta_offset=self.conv_beta(cond)
        beta=self.tr_beta(cond,beta_offset)
        alpha_offset_copy=self.conv_alpha_copy(cond)
        alpha_copy=self.tr_alpha_copy(cond,alpha_offset_copy)
        beta_offset_copy=self.conv_beta_copy(cond)
        beta_copy=self.tr_beta_copy(cond,beta_offset_copy)
        y_feat=self.x_tr_2(y)
        y_feat=(alpha+1)*y_feat+beta
        y_feat=self.relu(y_feat)
        y_feat_fuse=self.y_feat_fuse(y_feat)
        x_feat=self.x_tr_2(x)
        x_feat=(alpha_copy+1)*x_feat+beta_copy
        x_feat=self.relu(x_feat)
        x_feat_fuse=self.x_feat_fuse(x_feat)
#         y=y+0.1*x_feat_fuse
#         x=x+0.1*y_feat_fuse
        x=y_feat_fuse
        y=x_feat_fuse
        fdx=x-y#N,C,H,W
        vx=F.tanh(self.GAP(fdx))#N,C,1,1
        x_res=x+vx*y
        fdy=y-x
        vy=F.tanh(self.GAP(fdy))
        y_res=y+vy*x
        x=self.conv1(x)
        x_res=self.conv2(x_res)
        y=self.conv1_t(y)
        y_res=self.conv2_t(y_res)
        return x+x_res,y+y_res
class multi_mix_CBL_PLARD_modified_wh_t(nn.Module):
    def __init__(self,c1,c2):#c1,c2用不到，因为输入和输出通道数是完全一样的，无需改变通道数。为了模型保持一致所以这里加上。
        super(multi_mix_CBL_PLARD_modified_wh_t,self).__init__()
        self.GAP=nn.AdaptiveAvgPool2d(1)
        self.conv1=Conv(c1,2*c1,1,2)
        self.conv1_t=Conv(c1,2*c1,1,2)
        self.conv2=Conv(c1,2*c1,3,2)
        self.conv2_t=Conv(c1,2*c1,3,2)
        self.x_tr=nn.Conv2d(c1,c1,kernel_size=1,stride=1,padding=0,bias=True)
        self.x_t_tr=nn.Conv2d(c1,c1,kernel_size=1,stride=1,padding=0,bias=True)
        self.tr_alpha=nn.Conv2d(2*c1,c1,kernel_size=1,stride=1,padding=0,bias=True)
        self.tr_beta=nn.Conv2d(2*c1,c1,kernel_size=1,stride=1,padding=0,bias=True)
        self.relu=nn.ReLU(inplace=True)
        self.x_tr_2=nn.Conv2d(c1,c1,kernel_size=1,stride=1,padding=0,bias=True)
        self.x_t_tr_2=nn.Conv2d(c1,c1,kernel_size=1,stride=1,padding=0,bias=True)
        self.x_feat_fuse=nn.Conv2d(c1,c1,kernel_size=1,stride=1,padding=0,bias=True)
        self.y_feat_fuse=nn.Conv2d(c1,c1,kernel_size=1,stride=1,padding=0,bias=True)
        self.tr_alpha_copy=nn.Conv2d(2*c1,c1,kernel_size=1,stride=1,padding=0,bias=True)
        self.tr_beta_copy=nn.Conv2d(2*c1,c1,kernel_size=1,stride=1,padding=0,bias=True)
        self.wh_conv_alpha=Conv(2*c1,2*c1,1,1)
        self.wh_conv_alpha_copy=Conv(2*c1,2*c1,1,1)
    def forward(self,x,y):
        W,H=x.size()[2],x.size()[-1]
        x_tr=self.x_tr(x)
        x_tr=self.relu(x_tr)
        y_tr=self.x_t_tr(y)
        y_tr=self.relu(y_tr)
        cond=torch.cat((x_tr,y_tr),1)
#         device = torch.device('cuda')
#         cond=cond.to(device)
#         cond.to(torch.float16)
        cond1=self.wh_conv_alpha_copy(cond)
        alpha=self.tr_alpha(cond1)
        beta=self.tr_beta(cond1)
        cond2=self.wh_conv_alpha_copy(cond)
        alpha_copy=self.tr_alpha_copy(cond2)
        beta_copy=self.tr_beta_copy(cond2)
        y_feat=self.x_tr_2(y)
        y_feat=(alpha+1)*y_feat+beta
        y_feat=self.relu(y_feat)
        y_feat_fuse=self.y_feat_fuse(y_feat)
        x_feat=self.x_tr_2(x)
        x_feat=(alpha_copy+1)*x_feat+beta_copy
        x_feat=self.relu(x_feat)
        x_feat_fuse=self.x_feat_fuse(x_feat)
#         y=y+0.1*x_feat_fuse
#         x=x+0.1*y_feat_fuse
        x=y_feat_fuse
        y=x_feat_fuse
        fdx=x-y#N,C,H,W
        vx=F.tanh(self.GAP(fdx))#N,C,1,1
        x_res=x+vx*y
        fdy=y-x
        vy=F.tanh(self.GAP(fdy))
        y_res=y+vy*x
        x=self.conv1(x)
        x_res=self.conv2(x_res)
        y=self.conv1_t(y)
        y_res=self.conv2_t(y_res)
        return x+x_res,y+y_res
class Focus(nn.Module):
    # Focus wh information into c-space
    # c1 c2 = 3 32
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True):  # ch_in, ch_out, kernel, stride, padding, groups
        super(Focus, self).__init__()
        self.conv = Conv(c1 * 4, c2, k, s, p, g, act)
        # self.contract = Contract(gain=2)


    def forward(self, x):  # x(b,c,w,h) -> y(b,4c,w/2,h/2)
        # print(x.shape)    # torch.Size([1, 3, 256, 256])
        y = self.conv(torch.cat([x[..., ::2, ::2], x[..., 1::2, ::2], x[..., ::2, 1::2], x[..., 1::2, 1::2]], 1))
        # print(y.shape)    # torch.Size([1, 32, 128, 128])
        return y
        # return self.conv(self.contract(x))


class Focus_t(nn.Module):
    # Focus wh information into c-space
    # c1 c2 = 3 32
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True):  # ch_in, ch_out, kernel, stride, padding, groups
        super(Focus_t, self).__init__()
        self.conv = Conv(c1 * 4, c2, k, s, p, g, act)
        # self.contract = Contract(gain=2)

        self.conv_t = Conv(c1 * 4, c2, k, s, p, g, act)


    def forward(self, x , xx):  # x(b,c,w,h) -> y(b,4c,w/2,h/2)
        # print(x.shape)    # torch.Size([1, 3, 256, 256])
        y = self.conv(torch.cat([x[..., ::2, ::2], x[..., 1::2, ::2], x[..., ::2, 1::2], x[..., 1::2, 1::2]], 1))
        yy = self.conv_t(torch.cat([xx[..., ::2, ::2], xx[..., 1::2, ::2], xx[..., ::2, 1::2], xx[..., 1::2, 1::2]], 1))
        # print(y.shape)    # torch.Size([1, 32, 128, 128])
        return y, yy




# class Focus_cat(nn.Module):
#     # Focus wh information into c-space
#     def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True):  # ch_in, ch_out, kernel, stride, padding, groups
#         super(Focus_cat, self).__init__()
#         self.conv = Conv(c1, c2-1, k, s, p, g, act)
#         self.conv_ = Conv(3 * 4, c2 , k, s, p, g, act)
#         self.max_pool = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))
#     def forward(self, x):  # x(b,c,w,h) -> y(b,4c,w/2,h/2)
#         if x.size()[1]>3:
#             x_radar = x[:, 3:, :, :]
#             x_radar = self.max_pool(x_radar)
#             # x = x[:, :3, :, :]
#             x_focus = self.conv(torch.cat([x[..., ::2, ::2], x[..., 1::2, ::2], x[..., ::2, 1::2], x[..., 1::2, 1::2]], 1))
#             return torch.cat((x_focus, x_radar), dim=1)
#         else:
#             return self.conv_(torch.cat([x[..., ::2, ::2], x[..., 1::2, ::2], x[..., ::2, 1::2], x[..., 1::2, 1::2]], 1))
#







class Contract(nn.Module):
    # Contract width-height into channels, i.e. x(1,64,80,80) to x(1,256,40,40)
    def __init__(self, gain=2):
        super().__init__()
        self.gain = gain

    def forward(self, x):
        N, C, H, W = x.size()  # assert (H / s == 0) and (W / s == 0), 'Indivisible gain'
        s = self.gain
        x = x.view(N, C, H // s, s, W // s, s)  # x(1,64,40,2,40,2)
        x = x.permute(0, 3, 5, 1, 2, 4).contiguous()  # x(1,2,2,64,40,40)
        return x.view(N, C * s * s, H // s, W // s)  # x(1,256,40,40)


class Expand(nn.Module):
    # Expand channels into width-height, i.e. x(1,64,80,80) to x(1,16,160,160)
    def __init__(self, gain=2):
        super().__init__()
        self.gain = gain

    def forward(self, x):
        N, C, H, W = x.size()  # assert C / s ** 2 == 0, 'Indivisible gain'
        s = self.gain
        x = x.view(N, s, s, C // s ** 2, H, W)  # x(1,2,2,16,80,80)
        x = x.permute(0, 3, 4, 1, 5, 2).contiguous()  # x(1,16,80,2,80,2)
        return x.view(N, C // s ** 2, H * s, W * s)  # x(1,16,160,160)


class Concat(nn.Module):
    # Concatenate a list of tensors along dimension
    def __init__(self, dimension=1):
        super(Concat, self).__init__()
        self.d = dimension

    def forward(self, x):
        return torch.cat(x, self.d)


class NMS(nn.Module):
    # Non-Maximum Suppression (NMS) module
    conf = 0.25  # confidence threshold
    iou = 0.45  # IoU threshold
    classes = None  # (optional list) filter by class

    def __init__(self):
        super(NMS, self).__init__()

    def forward(self, x):
        return non_max_suppression(x[0], conf_thres=self.conf, iou_thres=self.iou, classes=self.classes)


class autoShape(nn.Module):
    # input-robust model wrapper for passing cv2/np/PIL/torch inputs. Includes preprocessing, inference and NMS
    conf = 0.25  # NMS confidence threshold
    iou = 0.45  # NMS IoU threshold
    classes = None  # (optional list) filter by class

    def __init__(self, model):
        super(autoShape, self).__init__()
        self.model = model.eval()

    def autoshape(self):
        print('autoShape already enabled, skipping... ')  # model already converted to model.autoshape()
        return self

    def forward(self, imgs, size=640, augment=False, profile=False):
        # Inference from various sources. For height=720, width=1280, RGB images example inputs are:
        #   filename:   imgs = 'data/samples/zidane.jpg'
        #   URI:             = 'https://github.com/ultralytics/yolov5/releases/download/v1.0/zidane.jpg'
        #   OpenCV:          = cv2.imread('image.jpg')[:,:,::-1]  # HWC BGR to RGB x(720,1280,3)
        #   PIL:             = Image.open('image.jpg')  # HWC x(720,1280,3)
        #   numpy:           = np.zeros((720,1280,3))  # HWC
        #   torch:           = torch.zeros(16,3,720,1280)  # BCHW
        #   multiple:        = [Image.open('image1.jpg'), Image.open('image2.jpg'), ...]  # list of images

        t = [time_synchronized()]
        p = next(self.model.parameters())  # for device and type
        if isinstance(imgs, torch.Tensor):  # torch
            return self.model(imgs.to(p.device).type_as(p), augment, profile)  # inference

        # Pre-process
        n, imgs = (len(imgs), imgs) if isinstance(imgs, list) else (1, [imgs])  # number of images, list of images
        shape0, shape1, files = [], [], []  # image and inference shapes, filenames
        for i, im in enumerate(imgs):
            if isinstance(im, str):  # filename or uri
                im, f = Image.open(requests.get(im, stream=True).raw if im.startswith('http') else im), im  # open
                im.filename = f  # for uri
            files.append(Path(im.filename).with_suffix('.jpg').name if isinstance(im, Image.Image) else f'image{i}.jpg')
            im = np.array(im)  # to numpy
            if im.shape[0] < 5:  # image in CHW
                im = im.transpose((1, 2, 0))  # reverse dataloader .transpose(2, 0, 1)
            im = im[:, :, :3] if im.ndim == 3 else np.tile(im[:, :, None], 3)  # enforce 3ch input
            s = im.shape[:2]  # HWC
            shape0.append(s)  # image shape
            g = (size / max(s))  # gain
            shape1.append([y * g for y in s])
            imgs[i] = im  # update
        shape1 = [make_divisible(x, int(self.stride.max())) for x in np.stack(shape1, 0).max(0)]  # inference shape
        x = [letterbox(im, new_shape=shape1, auto=False)[0] for im in imgs]  # pad
        x = np.stack(x, 0) if n > 1 else x[0][None]  # stack
        x = np.ascontiguousarray(x.transpose((0, 3, 1, 2)))  # BHWC to BCHW
        x = torch.from_numpy(x).to(p.device).type_as(p) / 255.  # uint8 to fp16/32
        t.append(time_synchronized())

        # Inference
        with torch.no_grad():
            y = self.model(x, augment, profile)[0]  # forward
        t.append(time_synchronized())

        # Post-process
        y = non_max_suppression(y, conf_thres=self.conf, iou_thres=self.iou, classes=self.classes)  # NMS
        for i in range(n):
            scale_coords(shape1, y[i][:, :4], shape0[i])
        t.append(time_synchronized())

        return Detections(imgs, y, files, t, self.names, x.shape)


class Detections:
    # detections class for YOLOv5 inference results
    def __init__(self, imgs, pred, files, times=None, names=None, shape=None):
        super(Detections, self).__init__()
        d = pred[0].device  # device
        gn = [torch.tensor([*[im.shape[i] for i in [1, 0, 1, 0]], 1., 1.], device=d) for im in imgs]  # normalizations
        self.imgs = imgs  # list of images as numpy arrays
        self.pred = pred  # list of tensors pred[0] = (xyxy, conf, cls)
        self.names = names  # class names
        self.files = files  # image filenames
        self.xyxy = pred  # xyxy pixels
        self.xywh = [xyxy2xywh(x) for x in pred]  # xywh pixels
        self.xyxyn = [x / g for x, g in zip(self.xyxy, gn)]  # xyxy normalized
        self.xywhn = [x / g for x, g in zip(self.xywh, gn)]  # xywh normalized
        self.n = len(self.pred)  # number of images (batch size)
        self.t = tuple((times[i + 1] - times[i]) * 1000 / self.n for i in range(3))  # timestamps (ms)
        self.s = shape  # inference BCHW shape

    def display(self, pprint=False, show=False, save=False, render=False, save_dir=''):
        colors = color_list()
        for i, (img, pred) in enumerate(zip(self.imgs, self.pred)):
            str = f'image {i + 1}/{len(self.pred)}: {img.shape[0]}x{img.shape[1]} '
            if pred is not None:
                for c in pred[:, -1].unique():
                    n = (pred[:, -1] == c).sum()  # detections per class
                    str += f"{n} {self.names[int(c)]}{'s' * (n > 1)}, "  # add to string
                if show or save or render:
                    for *box, conf, cls in pred:  # xyxy, confidence, class
                        label = f'{self.names[int(cls)]} {conf:.2f}'
                        plot_one_box(box, img, label=label, color=colors[int(cls) % 10])
            img = Image.fromarray(img.astype(np.uint8)) if isinstance(img, np.ndarray) else img  # from np
            if pprint:
                print(str.rstrip(', '))
            if show:
                img.show(self.files[i])  # show
            if save:
                f = Path(save_dir) / self.files[i]
                img.save(f)  # save
                print(f"{'Saving' * (i == 0)} {f},", end='' if i < self.n - 1 else ' done.\n')
            if render:
                self.imgs[i] = np.asarray(img)

    def print(self):
        self.display(pprint=True)  # print results
        print(f'Speed: %.1fms pre-process, %.1fms inference, %.1fms NMS per image at shape {tuple(self.s)}' % self.t)

    def show(self):
        self.display(show=True)  # show results

    def save(self, save_dir='results/'):
        Path(save_dir).mkdir(exist_ok=True)
        self.display(save=True, save_dir=save_dir)  # save results

    def render(self):
        self.display(render=True)  # render results
        return self.imgs

    def __len__(self):
        return self.n

    def tolist(self):
        # return a list of Detections objects, i.e. 'for result in results.tolist():'
        x = [Detections([self.imgs[i]], [self.pred[i]], self.names) for i in range(self.n)]
        for d in x:
            for k in ['imgs', 'pred', 'xyxy', 'xyxyn', 'xywh', 'xywhn']:
                setattr(d, k, getattr(d, k)[0])  # pop out of list
        return x


class Classify(nn.Module):
    # Classification head, i.e. x(b,c1,20,20) to x(b,c2)
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1):  # ch_in, ch_out, kernel, stride, padding, groups
        super(Classify, self).__init__()
        self.aap = nn.AdaptiveAvgPool2d(1)  # to x(b,c1,1,1)
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p), groups=g)  # to x(b,c2,1,1)
        self.flat = nn.Flatten()

    def forward(self, x):
        z = torch.cat([self.aap(y) for y in (x if isinstance(x, list) else [x])], 1)  # cat if list
        return self.flat(self.conv(z))  # flatten to x(b,c2)
