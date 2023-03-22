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
            # nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            #nn.LeakyReLU(0.2),
            nn.Conv2d(in_channels=32,out_channels=32,kernel_size=3,padding=1),
            # nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=32,out_channels=out_channels,kernel_size=3,padding=1),
            # nn.BatchNorm2d(out_channels),
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
            # nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=mid_channels,out_channels=out_channels,kernel_size=3,padding=1),
            # nn.BatchNorm2d(out_channels),
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
class BiProjection(nn.Module):
    def __init__(self,channels):
        super(BiProjection,self).__init__()
        self.rgb_conv=nn.Sequential(
            nn.Conv2d(channels,channels,kernel_size=3,padding=1),
            nn.ReLU(inplace=True)
        )
        self.gbuffer_conv=nn.Sequential(
            nn.Conv2d(channels,channels,kernel_size=3,padding=1),
            nn.ReLU(inplace=True)
        )
        self.mask_gen=nn.Sequential(
            nn.Conv2d(channels*2,1,kernel_size=1,padding=0),
            nn.Sigmoid()
        )
    def forward(self,F_rgb,F_gbuffer):
        encode_rgb=self.rgb_conv(F_rgb)
        encode_gbuffer=self.gbuffer_conv(F_gbuffer)
        encode_all=torch.cat([encode_rgb,encode_gbuffer],dim=1)
        mask=self.mask_gen(encode_all)
        return torch.cat([F_rgb+mask*encode_gbuffer,F_gbuffer+(1-mask)*encode_rgb],dim=1)
## from https://github.com/moskomule/senet.pytorch/blob/master/senet/se_module.py
class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )
    def _initialize_weights(self):
        #torch.nn.init.xavier_uniform_(self.avg_pool.weight,gain=1)
        for m in self.fc.children():
            if isinstance(m,nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight, gain=1)
    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)
class CEE_Fuse(nn.Module):
    def __init__(self,in_channels):
        super(CEE_Fuse,self).__init__()
        #论文中并没有说不使用bias，但所有实现都没有bias
        self.residual=nn.Sequential(
            nn.Conv2d(in_channels*2,in_channels,kernel_size=1,bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels,in_channels,kernel_size=3,padding=1,bias=False),
            nn.BatchNorm2d(in_channels)
            #,nn.ReLU(inplace=True)
        )
        self.SE=SELayer(in_channels)
        self.last_conv=nn.Sequential(
            nn.Conv2d(in_channels,in_channels*2,kernel_size=1),
            nn.ReLU(inplace=True)
        )
    def _initialize_weights(self):
        
        self.SE._initialize_weights()
        for m in self.residual.children():
            if isinstance(m,nn.Conv2d):
                torch.nn.init.xavier_uniform_(m.weight, gain=1)
    def forward(self,F_rgb,F_gbuffer):
        F_gbuffer=self.SE(F_gbuffer)
        F_all=torch.cat([F_rgb,F_gbuffer],dim=1)
        F_res=self.residual(F_all)
        F_res_rgb=F_res+F_rgb
        F_res_all=torch.cat([F_res_rgb,F_gbuffer],dim=1)
        return F_res_all

        # F_all=torch.cat([F_rgb,F_gbuffer],dim=1)
        # F_all=F_gbuffer+F_rgb
        # return F_all
        # F_res=self.residual(F_all)
        # F_res_rgb=F_res+F_rgb
        
        # F_res_all=torch.cat([F_res_rgb,F_gbuffer],dim=1)
        # return F_res_all
        # result=self.SE(F_res_all)
        
        # return result
def otherzeroUpsampling(img):#x:(4 or 12) channels,and return a tensor batchsize*2*(h*3)*(w*3) 'y' 
    tmp=torch.rand(img.shape[0],img.shape[1],img.shape[2]*config.hs,img.shape[3]*config.ws,device=img.device)
    _,indices=F.max_pool2d_with_indices(tmp,config.hs)
    output=F.max_unpool2d(img,indices,config.hs,config.hs)
    return output
def zeroUpsampling(img,scale_factor=None):
    if scale_factor==None:
        scale_factor=(config.ws,config.hs)
    output = torch.zeros((img.shape[0],img.shape[1],img.shape[2]*scale_factor[0],img.shape[3]*scale_factor[1]),device=img.device)
    output[:, :, ::scale_factor[0], ::scale_factor[1]] = img
    return output
def myZeroUpsampling(img):
    batch,channels,h,w=img.shape
    a=img.unsqueeze(-1)
    b=torch.zeros(batch,channels,h,w,config.ws-1,device=img.device)
    c=torch.cat([a,b],axis=-1).reshape(batch,channels,h,w*config.ws).transpose(2,3).unsqueeze(-1)

    d=torch.zeros(batch,channels,w*config.ws,h,config.hs-1,device=img.device)
    res=torch.cat([c,d],axis=-1).reshape(batch,channels,w*config.ws,h*config.hs).transpose(2,3)
    return res
class FuseReconstruction(nn.Module): # U-net structure
    def __init__(self,rgb_channels,gbuffer_channels,out_channels=3,bilinear=False):
        super(FuseReconstruction,self).__init__()
        self.rgb_conv1=Unet_conv(rgb_channels,16,32)
        self.gbuffer_conv1=Unet_conv(gbuffer_channels,16,32)
        self.fuse1=CEE_Fuse(16)
        #self.rgb_pool1=nn.MaxPool2d(kernel_size=2)
        #self.gbuffer_pool1=nn.MaxPool2d(kernel_size=2)
        

        self.rgb_conv2=Unet_conv(16,32)
        self.gbuffer_conv2=Unet_conv(16,32)
        self.fuse2=CEE_Fuse(32)
        #self.rgb_pool2=nn.MaxPool2d(kernel_size=2)
        #self.gbuffer_pool2=nn.MaxPool2d(kernel_size=2)

        self.rgb_conv3=Unet_conv(32,64)
        self.gbuffer_conv3=Unet_conv(32,64)
        self.fuse3=CEE_Fuse(64)


        
        self.up1=Unet_up(128,64,64,64,bilinear)
        self.up2=Unet_up(64,32,out_channels,32,bilinear)
    def _initialize_weights(self):
        self.rgb_conv1._initialize_weights()
        self.rgb_conv2._initialize_weights()
        self.rgb_conv3._initialize_weights()
        self.gbuffer_conv1._initialize_weights()
        self.gbuffer_conv2._initialize_weights()
        self.gbuffer_conv3._initialize_weights()
        self.fuse1._initialize_weights()
        self.fuse2._initialize_weights()
        self.fuse3._initialize_weights()
        self.up1._initialize_weights()
        self.up2._initialize_weights()
    def forward(self,F_rgb,F_gbuffer):
        x1=self.rgb_conv1(F_rgb)
        x2=self.rgb_conv2(x1)
        x3=self.rgb_conv3(x2)
        y1=self.gbuffer_conv1(F_gbuffer)
        y2=self.gbuffer_conv2(y1)
        y3=self.gbuffer_conv3(y2)
        f1=self.fuse1(x1,y1)
        f2=self.fuse2(x2,y2)
        f3=self.fuse3(x3,y3)
        # print(f1.shape,f2.shape,f3.shape)
        
        z=self.up1(f3,f2)
        output=self.up2(z,f1)
        return output


class superNet(nn.Module):
    def __init__(self):
        super(superNet,self).__init__()
        self.cur_rgb_extract=FeatureExtraction(3,15)
        # self.tem_rgb_extract=FeatureExtraction(6,12)
        ####
        self.cur_gbuffer_extract=FeatureExtraction(9,9)
        # self.rgb_fu=nn.Sequential(
        #     nn.Conv2d(18*2,18,kernel_size=3,padding=1),
        #     nn.ReLU(inplace=True)
        # )
        ####
        self.fusion_reconstruct=FuseReconstruction(rgb_channels=18,gbuffer_channels=18,out_channels=3,bilinear=False)#TODO:
        self.last_conv=nn.Conv2d(18,3,kernel_size=1)
    def _initialize_weights(self):
        # print(self.modules())
 
        self.cur_rgb_extract._initialize_weights()
        self.cur_gbuffer_extract._initialize_weights()
        self.fusion_reconstruct._initialize_weights()
        torch.nn.init.xavier_uniform_(self.last_conv.weight, gain=1)
    def forward(self,cur_rgb,cur_gbuffer):# batchsize * (4 or 2) * h * w
        #prev_rgbd : batchsize * prevnums * 4 * h * w
        b,c,h,w=cur_rgb.shape
        # print(cur_gbuffer.shape,cur_rgb.shape)
        # cur_rgb=zeroUpsampling(cur_rgb)
        # cur_rgb=F.interpolate(cur_rgb,size=(4*h,4*w),mode='bicubic',align_corners=True)
        # cur_gbuffer=torch.cat([cur_gbuffer[:,0:4],cur_gbuffer[:,7:]],dim=1)
        # cur_gbuffer=cur_gbuffer[:,0:8]
        cur_rgb_feature=self.cur_rgb_extract(cur_rgb[:,:3])
        cur_gbuffer_feature=self.cur_gbuffer_extract(cur_gbuffer)
        ####
        # tem_rgb_feature=self.tem_rgb_extract(cur_rgb[:,3:])
        # print(tem_rgb_feature.shape,cur)
        # cur_rgb_feature=self.rgb_fu(torch.cat([cur_rgb_feature,tem_rgb_feature],dim=1))
        ####
        cur_rgb_feature=myZeroUpsampling(cur_rgb_feature)
        # cur_rgb_feature=F.interpolate(cur_rgb_feature,size=(4*h,4*w),mode='bilinear',align_corners=True)
        output=self.fusion_reconstruct(cur_rgb_feature,cur_gbuffer_feature)
        return output