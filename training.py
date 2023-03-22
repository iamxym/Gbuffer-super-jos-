import config
import cv2 as cv
import os
cv.setNumThreads(0) 
import torch
import torch.utils.data as data
import time
import torch.nn as nn
from Loaders import superTrainDataset
from net import myLoss
import net
import numpy as np
def observe(path):
    data=np.load(path)
    label=torch.tensor(data['label']).float()
    cur_rgb=torch.tensor(data['cur_rgb']).float()
    cur_gbuffer=torch.tensor(data['cur_gbuffer']).float()
    return cur_rgb.unsqueeze(0),cur_gbuffer.unsqueeze(0),label.unsqueeze(0)
def display():
    path='G:/DisplaySet/MD_displayDataset/0042/'
    type='MedievalDocks'
    id='0042'
    albedo=cv.imread(path+type+'BaseColor.%s.png'%(id),cv.IMREAD_UNCHANGED)
    normal=cv.imread(path+type+'WorldNormal.%s.png'%(id),cv.IMREAD_UNCHANGED)
    # print(self.path+type+'LR_'+self.extra_pre+self.data+'PreTonemapHDRColor.%s.png'%(id))
    # print(path+type+'SceneDepth.%s.png'%(id))
    # print(path+type+'SceneDepth.%s.png'%(id))
    depth=cv.imread(path+type+'SceneDepth.%s.png'%(id))[:,:,0:1]
    
    
    roughness=cv.imread(path+type+'Roughness.%s.png'%(id))[:,:,0:1]
    metallic=cv.imread(path+type+'Metallic.%s.png'%(id))[:,:,0:1]
    # print(albedo.shape,normal.shape,depth.shape,roughness.shape,metallic.shape)
    img=cv.imread(path+'LR_'+type+'PreTonemapHDRColor.%s.png'%(id),cv.IMREAD_UNCHANGED)
    # img=cv.resize(img,dsize=(0,0),fx=1/2,fy=1/2,interpolation=cv.INTER_CUBIC)
    label=cv.imread(path+type+'PreTonemapHDRColor.%s.png'%(id),cv.IMREAD_UNCHANGED)
    label=torch.tensor(label.transpose([2,0,1])).float().unsqueeze(0)
    cur_rgb=torch.tensor(img.transpose([2,0,1])).float().unsqueeze(0)
    cur_gbuffer=torch.tensor(np.concatenate((depth,albedo,normal,roughness,metallic),axis=2).transpose([2,0,1])).float().unsqueeze(0)
    print(cur_rgb.shape,cur_gbuffer.shape,label.shape)
    return cur_rgb,cur_gbuffer[:,0:1],label
def train(dataLoader,modelPath,modelName):
    model=net.superNet()
    model=model.to(torch.device("cuda:0"))
    model._initialize_weights()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learningrate)
    #scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=config.total_epochs,eta_min=1e-6)
    scheduler=torch.optim.lr_scheduler.StepLR(optimizer, config.scheduler_step, gamma=config.scheduler_gamma, last_epoch=-1)
    criterion=myLoss(relative_weight=0.2)
    # criterion=nn.L1Loss()

    # training code here
    for epoch in range(config.total_epochs):
        #print(modelPath+str(epoch)+"epoch_"+modelName)
        model.train()
        iter=0
        loss_all=0
        startTime = time.time()
        for cur_rgb,cur_gbuffer,label in dataLoader:
            
            cur_rgb=cur_rgb.cuda()
            label=label.cuda()
            cur_gbuffer=cur_gbuffer.cuda()
            optimizer.zero_grad()
            pred=model(cur_rgb,cur_gbuffer)  #the net's forward
            pred=pred.clip(0,255)
            loss=criterion(pred,label)
            loss.backward()                             #the Derivation(Loss') of the Loss
            optimizer.step()                            #update the parameter in net
            iter+=1
            loss_all+=loss
            if iter%config.printPeriod==1:
                print(loss)
        # if epoch>20:
        scheduler.step()
        
        endTime = time.time()
        print("epoch time is {}".format(endTime - startTime))
        print("%d mean loss for train is %f"%(epoch,loss_all/iter))
        
        # if epoch%10==2:
        #     model.eval()
        #     cur_rgb,cur_gbuffer,label=display()
        #     cur_rgb=cur_rgb.cuda()
        #     label=label.cuda()
        #     cur_gbuffer=cur_gbuffer.cuda()
        #     pred=model(cur_rgb,cur_gbuffer).clip(0,255)
        #     cv.imwrite("input%d.png"%(iter),cur_rgb[0].data.cpu().numpy().transpose([1,2,0]))
        #     cv.imwrite('G:/DisplaySet/MD_displayDataset/0042/'+"pred%d.png"%(epoch),pred[0].data.cpu().numpy().transpose([1,2,0]))
            # cv.imwrite("label%d.png"%(iter),label[0].data.cpu().numpy().transpose([1,2,0]))
        if (epoch%10==9):
            torch.save(model.state_dict(),modelPath+str(epoch)+"epoch_"+modelName)
    torch.save(model.state_dict(), modelPath+modelName)
if __name__ =="__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    trainDataset=superTrainDataset(0)
    trainLoader = data.DataLoader(trainDataset,config.batch_size,shuffle=False,num_workers=1, pin_memory=True)
    # _,__,___=display()
    # print(_.shape,__.shape,___.shape)
    # train(trainLoader, "D:/Python-Test/Models/","RF-super-net-model.pth")
    train(trainLoader, "D:/Python-Test/Models/","MD-unet-model.pth")
    #train(trainLoader,"./spatial-upsampling-superNet-model.pth")
    