from functools import total_ordering
import cv2 as cv
import numpy as np
import torch.nn as nn
import torch
import torch.nn.functional as F
from torch.nn.modules.conv import Conv2d
import config
import random
import lpips
import pytorch_ssim

import torchvision.models.vgg as vgg

class PerceptualLoss(nn.Module):

    def __init__(self):
        super(PerceptualLoss, self).__init__()
        self.vgg_layers = vgg.vgg16(pretrained=True).features
        self.layer_name=['3','8','15','22','29','end']
    def forward(self, x):
        res=[]
        id=0
        for name, module in self.vgg_layers._modules.items():
            x = module(x)
            # print(name)
            if name==self.layer_name[id]:
                res.append(x)
                id=id+1
        return res
class oomyLoss(nn.Module):
    def __init__(self,relative_weight=0.1):
        super(myLoss,self).__init__()
        self.weight=relative_weight
        self.loss_fn_vgg = PerceptualLoss().cuda()
        self.loss_fn_vgg.eval()
        self.mse_loss=nn.MSELoss()
        self.loss_ssim=pytorch_ssim.SSIM(window_size=11)
    
    def forward(self,pred,label):
        ssimloss=1-self.loss_ssim(pred/255,label/255)
        pred_features = self.loss_fn_vgg(pred)
        label_features = self.loss_fn_vgg(label)
        perceptualloss=0

        for i in range(len(label_features)):
            perceptualloss+=self.weight*pred_features[i].dist(label_features[i], p=2)/torch.numel(pred_features[i])
        return ssimloss+perceptualloss
class myLoss(nn.Module):
    def __init__(self,relative_weight=0.2):
        super(myLoss,self).__init__()
        self.weight=relative_weight
        self.loss_fn_vgg = lpips.LPIPS(net='vgg').cuda()
        self.loss_ssim=pytorch_ssim.SSIM(window_size=11)
        # self.loss_l1=nn.L1Loss()
    
    def forward(self,pred,label):
        ssimloss=1-self.loss_ssim(pred/255,label/255)
        # l1loss=self.loss_l1(pred,label)
        # 5|batch * c * h * w
        perceptualloss=self.loss_fn_vgg(((pred/255)*2-1),((label/255)*2-1))*self.weight
        return ssimloss+perceptualloss.mean()
        # return l1loss+perceptualloss.mean()
class FeatureExtraction(nn.Module):#input=3+1(channels) output=3+1+8=12
    # i-1到i-4共4帧，对每个帧都要color(3)和depth(1)
    def __init__(self,in_channels,out_channels):
        super(FeatureExtraction,self).__init__()
        self.f_e_net=nn.Sequential(
            nn.Conv2d(in_channels,out_channels=32,kernel_size=3,padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=32,out_channels=32,kernel_size=3,padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=32,out_channels=out_channels,kernel_size=3,padding=1),
            nn.ReLU(inplace=True),
        )
    def _initialize_weights(self):
        for m in self.f_e_net.children():    
            if isinstance(m,nn.Conv2d):
                torch.nn.init.xavier_uniform_(m.weight, gain=1)
                
    def forward(self,x):
        feature=self.f_e_net(x)
        y=torch.cat([x,feature],1)
        return y

class Unet_conv(nn.Module):
    def __init__(self,in_channels,out_channels,mid_channels=None):
        super(Unet_conv,self).__init__()
        stride=1
        if not mid_channels:
            mid_channels = out_channels
            stride=2
        self.conv=nn.Sequential(
            nn.Conv2d(in_channels=in_channels,out_channels=mid_channels,kernel_size=3,padding=1,stride=stride),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=mid_channels,out_channels=out_channels,kernel_size=3,padding=1),
            nn.ReLU(inplace=True)
        )
    def _initialize_weights(self):
        for m in self.conv.children():
            if isinstance(m,nn.Conv2d):
                torch.nn.init.xavier_uniform_(m.weight, gain=1)
    def forward(self,x):
        return self.conv(x)

class Unet_up(nn.Module):# upsample structure in U-net
    def __init__(self,in1_channels,in2_channels,out_channels,mid_channels=None,bilinear=False):
        super(Unet_up,self).__init__()
        if not mid_channels:
            mid_channels=out_channels
        if not bilinear:
            self.up=nn.ConvTranspose2d(in1_channels,in1_channels//2,kernel_size=2,stride=2)
            self.conv=Unet_conv(in1_channels//2+in2_channels,out_channels,mid_channels)
        else:
            self.up=nn.UpsamplingBilinear2d(scale_factor=2)
            self.conv=Unet_conv(in1_channels+in2_channels,out_channels,mid_channels)
        
    def _initialize_weights(self):
        self.conv._initialize_weights()
    def forward(self, pre_x,  skip_x):
        up_x=self.up(pre_x)
        # [0]-batch [1]-channels
        offset1=skip_x.size()[2]-up_x.size()[2]
        offset2=skip_x.size()[3]-up_x.size()[3]

        if (offset1==0 and offset2==0):
            padding_x=up_x
        else:
            padding_x=F.pad(up_x,[offset1//2,offset1-offset1//2,
                            offset2//2,offset2-offset2//2])     #padding is negative, size become smaller

        return self.conv(torch.cat([skip_x,padding_x],1))
def myZeroUpsampling(img):
    batch,channels,h,w=img.shape
    a=img.unsqueeze(-1)
    b=torch.zeros(batch,channels,h,w,config.ws-1,device=img.device)
    c=torch.cat([a,b],axis=-1).reshape(batch,channels,h,w*config.ws).transpose(2,3).unsqueeze(-1)

    d=torch.zeros(batch,channels,w*config.ws,h,config.hs-1,device=img.device)
    res=torch.cat([c,d],axis=-1).reshape(batch,channels,w*config.ws,h*config.hs).transpose(2,3)
    return res
class FuseReconstruction(nn.Module): # U-net structure
    def __init__(self,channels,out_channels=3,bilinear=False):
        super(FuseReconstruction,self).__init__()
        self.conv1=Unet_conv(channels,32,64)
        self.conv2=Unet_conv(32,64)
        self.conv3=Unet_conv(64,128)
        self.up1=Unet_up(128,64,64,64,bilinear)
        self.up2=Unet_up(64,32,out_channels,32,bilinear)
    def _initialize_weights(self):
        self.conv1._initialize_weights()
        self.conv2._initialize_weights()
        self.conv3._initialize_weights()
        self.up1._initialize_weights()
        self.up2._initialize_weights()
    def forward(self,F):
        x1=self.conv1(F)
        x2=self.conv2(x1)
        x3=self.conv3(x2)
        z=self.up1(x3,x2)
        output=self.up2(z,x1)
        return output


class superNet(nn.Module):
    def __init__(self):
        super(superNet,self).__init__()
        self.cur_extract=FeatureExtraction(12,24)
        self.fusion_reconstruct=FuseReconstruction(channels=36,out_channels=3,bilinear=False)#TODO:
    def _initialize_weights(self):
        self.cur_extract._initialize_weights()
        self.fusion_reconstruct._initialize_weights()
    def forward(self,cur_rgb,cur_gbuffer):# batchsize * (4 or 2) * h * w
        cur_rgb_feature=myZeroUpsampling(cur_rgb)
        x=torch.cat([cur_rgb_feature,cur_gbuffer],dim=1)
        feature=self.cur_extract(x)
        output=self.fusion_reconstruct(feature)
        return output