import cv2 as cv
from cv2 import CV_16S
cv.setNumThreads(0) 
import numpy as np
import os
import math
import torch
import config
def exr2png(img):
    img_gamma_correct = np.clip(np.power(img, 1/2.2), 0, 1)
    img_fixed = np.uint8(img_gamma_correct*255)
    return img_fixed
def calcSSIM(prediction, target):
    C1 = (0.01 * 255)**2
    C2 = (0.03 * 255)**2
    img1 = prediction.astype(np.float64)
    img2 = target.astype(np.float64)
    kernel = cv.getGaussianKernel(11, 1.5)
    window = np.outer(kernel, kernel.transpose())
    mu1 = cv.filter2D(img1, -1, window)[5:-5, 5:-5]  # valid
    mu2 = cv.filter2D(img2, -1, window)[5:-5, 5:-5]
    mu1_sq = mu1**2
    mu2_sq = mu2**2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = cv.filter2D(img1**2, -1, window)[5:-5, 5:-5] - mu1_sq
    sigma2_sq = cv.filter2D(img2**2, -1, window)[5:-5, 5:-5] - mu2_sq
    sigma12 = cv.filter2D(img1 * img2, -1, window)[5:-5, 5:-5] - mu1_mu2
    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
    return ssim_map.mean()
def cnn_paras_count(net):
    """cnn参数量统计, 使用方式cnn_paras_count(net)"""
    # Find total parameters and trainable parameters
    total_params = sum(p.numel() for p in net.parameters())
    print(f'{total_params:,} total parameters.')
    total_trainable_params = sum(p.numel() for p in net.parameters() if p.requires_grad)
    print(f'{total_trainable_params:,} training parameters.')
    return total_params, total_trainable_params
def calcPSNR(img1,img2):
    mse = np.mean( (img1/255. - img2/255.) ** 2 )
    if mse < 1.0e-10:
        return 100
    PIXEL_MAX = 1
    return 20 * math.log10(PIXEL_MAX / math.sqrt(mse))
def ImgRead(mPath,idx,prefix= None,format=".exr",cvtGray=False,cvtrgb=False):
    files=os.listdir(mPath)
    if prefix == None:
        prefix=files[0].split(".")[0]
    if format==".exr":
        img = cv.imread(os.path.join(mPath,prefix+"."+str(idx).zfill(4)+format),cv.IMREAD_UNCHANGED)
    else:
        img = cv.imread(os.path.join(mPath,prefix+"."+str(idx).zfill(4)+format))
    if cvtrgb == True:
        img=cv.cvtColor(img,cv.COLOR_BGR2RGB)
    if cvtGray==True:
        img = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
    return img
def ImgWrite(mPath,prefix,idx,img):
    cv.imwrite(os.path.join(mPath,prefix+str(idx).zfill(4)+".exr"),img)

def ImgReadWithPrefix(mPath,idx,p=None,prefix= None,format=".exr",cvtGray=False,cvtrgb=False):
    if p == None:
        return ImgRead(mPath, idx, prefix, format, cvtGray, cvtrgb)
    files=os.listdir(mPath)
    if prefix == None:
        prefix=files[0].split(".")[0]
    if format==".exr":
        img = cv.imread(os.path.join(mPath,prefix+"."+str(idx).zfill(4)+"."+p+format),cv.IMREAD_UNCHANGED)
    else:
        img = cv.imread(os.path.join(mPath,prefix+"."+str(idx).zfill(4)+format))
    if cvtrgb == True:
        img=cv.cvtColor(img,cv.COLOR_BGR2RGB)
    if cvtGray==True:
        img = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
    return img

def ReadData(path,augment=True):

    total = np.load(path)["i"]

    img = total[:,:,0:3]
    img3 = total[:,:,3:6]
    img5 = total[:,:,6:9]
    gt = total[:,:,21:24]
    depth = total[:,:,26:27]
    img = cv.cvtColor(img.astype(np.float32), cv.COLOR_RGB2BGR)
    img3 = cv.cvtColor(img.astype(np.float32), cv.COLOR_RGB2BGR)
    img5 = cv.cvtColor(img.astype(np.float32), cv.COLOR_RGB2BGR)
    gt=cv.cvtColor(img.astype(np.float32), cv.COLOR_RGB2BGR)
    depth = depth.astype(np.float32)

    depth = (depth - depth.min()) / (depth.max() - depth.min() + 1e-6)

    
    if augment:
        rd=np.random.uniform()
        if rd<0.2:
            img = np.flip(img,0)
            depth = np.flip(depth,0)
        elif rd<0.3:
            img = np.flip(img,1)
            depth = np.flip(depth,1)
        elif rd<0.35:
            img = np.flip(img,(0,1))
            depth = np.flip(depth,(0,1))

    return img,img3,img5,gt,depth
def generateData(savePath):
        
    
    '''path=config.basePaths[0]
    for id in range(279,377):
        img=cv.imread(path+"MedievalDocksPreTonemapHDRColor.0%d.exr"%(id),cv.IMREAD_UNCHANGED)
        depth=cv.imread(path+"MedievalDocksSceneDepth.0%d.exr"%(id),cv.IMREAD_UNCHANGED)
        mv=cv.imread(path+"MedievalDocksMotionVector.0%d.exr"%(id),cv.IMREAD_UNCHANGED)
        #print("MedievalDocksPreTonemapHDRColor.0%d.exr"%(id))
        #print(img[:,:,0:3].shape,depth[:,:,0:1].shape)
        rgbd=np.concatenate((img[:,:,0:3],depth[:,:,0:1],mv[:,:,1:3]),axis=2)
        #print(rgbd.shape)
        print(rgbd.shape)
        print(mv[540,960])
        np.save(savePath+"MedievalDocks.rgbdmv.0%d"%(id),rgbd)
    path=config.basePaths[1]
    for id in range(811,860):
        img=cv.imread(path+"MedievalDocksPreTonemapHDRColor.0%d.exr"%(id),cv.IMREAD_UNCHANGED)
        depth=cv.imread(path+"MedievalDocksSceneDepth.0%d.exr"%(id),cv.IMREAD_UNCHANGED)
        mv=cv.imread(path+"MedievalDocksMotionVector.0%d.exr"%(id),cv.IMREAD_UNCHANGED)
        #print("MedievalDocksPreTonemapHDRColor.0%d.exr"%(id))
        #print(img[:,:,0:3].shape,depth[:,:,0:1].shape)
        rgbd=np.concatenate((img[:,:,0:3],depth[:,:,0:1],mv[:,:,1:3]),axis=2)
        
        np.save(savePath+"MedievalDocks.rgbdmv.0%d"%(id),rgbd)'''
        
    total=0
    for id in range(283,377):
        data=np.load("../rgbdmv/MedievalDocks.rgbdmv.0%d.npy"%(id))
        for i in range(1,5):
            data=np.concatenate((data,np.load("../rgbdmv/MedievalDocks.rgbdmv.0%d.npy"%(id-i))),axis=2)
        np.save(savePath+"MedievalDocks.traindata.%04d"%(total),data)
        total=total+1
    for id in range(815,860):
        data=np.load("../rgbdmv/MedievalDocks.rgbdmv.0%d.npy"%(id))
        for i in range(1,5):
            data=np.concatenate((data,np.load("../rgbdmv/MedievalDocks.rgbdmv.0%d.npy"%(id-i))),axis=2)
        np.save(savePath+"MedievalDocks.traindata.%04d"%(total),data)
        total=total+1
    
def npToneSimple(img):
    '''errors = img == -1.0
    result =  np.log(np.ones(img.shape, np.float32) + img)
    result[errors] = 0.0
    return result'''
    return img


def npDeToneSimple(img):
    '''result = np.exp(img) - np.ones(img.shape, np.float32)
    result[result < 0.] = 0.
    return result'''
    return img

def torchToneSimple(img):
    '''errors = img == -1.0
    result =  torch.log(1 + img)
    result[errors] = 0.0
    return result'''
    return img


def torchDeToneSimple(img):
    '''print(img.max(),img.min())
    result = torch.exp(img) - 1
    result[result < 0.] = 0.
    return result'''
    return img

#generateData("D:/Python-Test/TrainData/")