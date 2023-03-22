import torch.utils.data as data
import torch
import os
import cv2 as cv
cv.setNumThreads(0) 
import numpy as np
import config
import time
from utils import npToneSimple,npDeToneSimple
import torchvision.transforms as transforms
#UPDATE
class superTrainDataset(data.Dataset):
    # 创建LiverDataset类的实例时，就是在调用init初始化
    def __init__(self, typeIndicator, transform=None):  # root表示图片路径
        self.indicator = typeIndicator
        
        self.totalNum = 0

        # self.totalSetNum = len(config.basePaths)
        
        self.imgSet = []
        for path in config.basePaths:
            imgs = os.listdir(path)
            setNum = len(imgs)
            self.imgSet.append(imgs)
            self.totalNum += setNum
        

    def mapIndex2PathAndIndex(self, index):
        remain = index
        for setIndex,ims in enumerate(self.imgSet):
            if remain < len(ims):
                # return config.basePaths[setIndex], ims[remain].split(".")[2]
                return config.basePaths[setIndex], ims[remain]
            else:
                remain -= len(ims)

        return None, -1
    def __getitem__(self, index):

        # idx=self.start + index
        path, img = self.mapIndex2PathAndIndex(index)
        #print(path+"MedievalDocks.traindata.{}.npy".format(idx))
        # data=np.load(path+img)
        # print(path+img,data['label'].shape)
        try:
            data=np.load(path+img)
            # print(data.shape)
            # img=data['cur_rgb']
            
            # img=img.transpose(1,2,0)
            # print(img.shape,img.dtype,type(img))
            # print("???")
            # print(img.dtype,type(img))
            # img=cv.resize(img,(0,0),fx=2,fy=2,interpolation=cv.INTER_CUBIC).transpose(2,0,1)
            label=torch.tensor(data['label']).float()
            cur_rgb=torch.tensor(data['cur_rgb']).float()
            cur_gbuffer=torch.tensor(data['cur_gbuffer']).float()
            #prev_rgbd=torch.tensor(data['prev_rgbd']).reshape(-1,4,cur_rgb.shape[1],cur_rgb.shape[2])
            #mv=torch.tensor(data['mv'])
            #cur_gbuffer=torch.tensor(data['d_a_n'][0:7])
            #prev_gbuffer=torch.tensor(data['d_a_n'][7:]).reshape(-1,7,cur_rgb.shape[1],cur_rgb.shape[2])
            #prev_rgbd=torch.cat([prev_rgb,prev_gbuffer[:,0:1]],dim=1)
            return cur_rgb,cur_gbuffer,label
            '''c
            
            label=torch.tensor(data[:,:,0:3].transpose([2,0,1]))
            data=cv.boxFilter(data,-1,(config.hs,config.ws))
            data=cv.resize(data,(data.shape[1]//config.ws,data.shape[0]//config.hs))
            data=torch.tensor(data.transpose([2,0,1]))
            cur_rgbd=data[0:4,:,:]
            prev_rgbd=torch.cat([data[6:10,:,:],data[12:16,:,:],data[18:22,:,:],data[24:28,:,:]],0).reshape(config.previous_frames,4,cur_rgbd.shape[1],cur_rgbd.shape[2])
            mv=torch.cat([(data[4:6,:,:]/config.hs),data[10:12,:,:]/config.hs,data[16:18,:,:]/config.hs,data[22:24,:,:]/config.hs,data[28:30,:,:]/config.hs],0)
            
            return cur_rgbd,prev_rgbd,mv,label'''
            
        except:
            print(path,img)
            # print(idx)

    def __len__(self):
        return self.totalNum
class superTestDataset(data.Dataset):
    # 创建LiverDataset类的实例时，就是在调用init初始化
    def __init__(self, typeIndicator, transform=None):  # root表示图片路径
        self.indicator = typeIndicator
        
        self.totalNum = 0

        # self.totalSetNum = len(config.basePaths)
        
        self.imgSet = []
        for path in config.testPaths:
            imgs = os.listdir(path)
            setNum = len(imgs)
            self.imgSet.append(imgs)
            self.totalNum += setNum
        

    def mapIndex2PathAndIndex(self, index):
        remain = index
        for setIndex,ims in enumerate(self.imgSet):
            if remain < len(ims):
                # return config.basePaths[setIndex], ims[remain].split(".")[2]
                return config.testPaths[setIndex], ims[remain]
            else:
                remain -= len(ims)

        return None, -1
    def __getitem__(self, index):

        # idx=self.start + index
        path, img = self.mapIndex2PathAndIndex(index)
        #print(path+"MedievalDocks.traindata.{}.npy".format(idx))
        try:
            data=np.load(path+img)
            
            label=torch.tensor(data['label']).float()
            cur_rgb=torch.tensor(data['cur_rgb']).float()
            cur_gbuffer=torch.tensor(data['cur_gbuffer']).float()
            #prev_rgbd=torch.tensor(data['prev_rgbd']).reshape(-1,4,cur_rgb.shape[1],cur_rgb.shape[2])
            #mv=torch.tensor(data['mv'])
            #cur_gbuffer=torch.tensor(data['d_a_n'][0:7])
            #prev_gbuffer=torch.tensor(data['d_a_n'][7:]).reshape(-1,7,cur_rgb.shape[1],cur_rgb.shape[2])
            #prev_rgbd=torch.cat([prev_rgb,prev_gbuffer[:,0:1]],dim=1)
            return cur_rgb,cur_gbuffer,label
        except:
            print(path)
            print(idx)

    def __len__(self):
        return self.totalNum

'''class MedTestDataset(data.Dataset):
    # 创建LiverDataset类的实例时，就是在调用init初始化
    def __init__(self, transform=None):  # root表示图片路径
        self.TAAimgs=os.listdir(config.TestTAADir)
        self.NOAAimgs=os.listdir(config.TestNo_TAADir)

    def __getitem__(self, index):
        TAAimgstr = self.TAAimgs[index]
        NOAAimgstr = self.NOAAimgs[index]
        TAAidx = TAAimgstr.split(".")[1]
        NOAAidx = NOAAimgstr.split(".")[1]
        assert TAAidx == NOAAidx
        idx=TAAidx
        TAAimg = ImgRead(config.TestTAADir,int(idx),cvtrgb=True)
        NOTAAimg = ImgRead(config.TestNo_TAADir,int(idx),cvtrgb=True)
        if config.TrainingType =="NoGubffer":
            #TAA and NOAA image are all, pass
            pass
        elif config.TrainingType=="Gubffer-Input" or config.TrainingType=="Gubffer-Att" or config.TrainingType=="Gbuffer-Input-Att":
            Normalimg = ImgRead(config.TestGbufferDir,idx,prefix=config.TestNormalPrefix,cvtrgb=True)
            Depthimg = ImgRead(config.TestGbufferDir,idx,prefix=config.TestDepthPrefix,cvtrgb=True)
            Depthimg = (Depthimg-Depthimg.min())/(Depthimg.max()-Depthimg.min()+1e-6)
            Roughnessimg = ImgRead(config.TestGbufferDir,idx,prefix=config.TestRoughnessPrefix,cvtrgb=True)
        labelimg = ImgRead(config.TestLabelDir,idx,cvtrgb=True)

        input = np.concatenate([NOTAAimg,TAAimg],axis=2)
        #prepare mask
        mask = input.copy()
        mask[mask==-1]=0.0
        mask[mask!=0.0]=1.0
        if config.TrainingType == "NoGubffer":
            #input,mask,label
            return torch.tensor(input.transpose([2,0,1])),torch.tensor(mask.transpose([2,0,1])),torch.tensor(labelimg.transpose([2,0,1]))
        elif config.TrainingType == "Gubffer-Input":
            #input,mask,label
            input = np.concatenate([input,Normalimg,Depthimg,Roughnessimg],axis=2)
            return torch.tensor(input.transpose([2,0,1])),torch.tensor(mask.transpose([2,0,1])),torch.tensor(labelimg.transpose([2,0,1]))
        elif config.TrainingType=="Gubffer-Att":
            attinput = np.concatenate([Normalimg,Depthimg,Roughnessimg],axis=2)
            #input,mask,attinput,label
            return torch.tensor(input.transpose([2, 0, 1])), torch.tensor(mask.transpose([2, 0, 1])),\
                   torch.tensor(attinput.transpose([2,0,1])),torch.tensor(labelimg.transpose([2, 0, 1]))
        elif config.TrainingType=="Gbuffer-Input-Att":
            #input,mask,attinput,label
            input = np.concatenate([input,Normalimg,Depthimg,Roughnessimg],axis=2)
            attinput = np.concatenate([Normalimg,Depthimg,Roughnessimg],axis=2)
            return torch.tensor(input.transpose([2, 0, 1])), torch.tensor(mask.transpose([2, 0, 1])), \
                   torch.tensor(attinput.transpose([2, 0, 1])), torch.tensor(labelimg.transpose([2, 0, 1]))
        else:
            assert 0
    def __len__(self):
        return len(self.TAAimgs)
'''