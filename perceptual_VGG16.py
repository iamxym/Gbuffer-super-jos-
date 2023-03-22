import torch
from torch.autograd import Variable
import torchvision
from torchvision import datasets, transforms, models
import os
import cv2 as cv
import numpy as np

def load_model():
    global model
    model = models.vgg16(pretrained=True)
    model.eval()
    model.cuda()
    # 查看迁移模型细节
    #print("迁移VGG16:\n", model.features[:])

def extract_conv(x): #batch * 3 * h * w
    if x.shape[1]!=3:
        print("Conv loss channels error!")
    x=x.type(torch.cuda.FloatTensor).requires_grad_()
    output=[]
    feature_model=model.features[:5]
    output.append(feature_model(x).data.cpu())
    feature_model=model.features[:10]
    output.append(feature_model(x).data.cpu())
    feature_model=model.features[:17]
    output.append(feature_model(x).data.cpu())
    feature_model=model.features[:24]
    output.append(feature_model(x).data.cpu())
    feature_model=model.features[:31]
    output.append(feature_model(x).data.cpu())
    return output
#load_model()
#img=cv.imread("test_feature.png",cv.IMREAD_UNCHANGED)
#print(img.shape)
#img=torch.tensor(img.transpose([2,0,1])[np.newaxis,:,:,:])

#feature_list=extract_conv(img)
#for i in range(len(feature_list)):
    #print(feature_list[i].squeeze(0).numpy().transpose([1,2,0]).shape)
    #cv.imwrite("feature-%d.jpg"%(i),feature_list[i].squeeze(0).numpy().transpose([1,2,0]))
