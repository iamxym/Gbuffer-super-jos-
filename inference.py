import config
import cv2 as cv
import os
cv.setNumThreads(0) 
import torch
import net_bak
import torch.utils.data as data
import time
from Loaders import superTestDataset,superTrainDataset
from utils import npToneSimple,npDeToneSimple,torchToneSimple,torchDeToneSimple,calcPSNR,calcSSIM,cnn_paras_count
from net_bak import myLoss
import torch.nn.functional as F
def test(modelPath,dataLoader):
    model=net_bak.superNet()
    model=model.to(torch.device("cuda:0"))
    model_dict=torch.load(modelPath, map_location="cuda:0")
    model.load_state_dict(model_dict)
    # criterion=myLoss()
    
    model.eval()
    
    iter=0
    ssim_all=0
    ssim=0
    psnr=0
    psnr_all=0
    cnn_paras_count(model)
    startTime = time.time()
    with torch.no_grad():
        for cur_rgb,cur_gbuffer,label in dataLoader:
            cur_rgb=cur_rgb.cuda()
            label=label.cuda()
            cur_gbuffer=cur_gbuffer.cuda()
            # torch.cuda.synchronize()
            # t1=time.time()
            
            pred1=model(cur_rgb,cur_gbuffer) #the net's forward
            pred1=pred1.clip(0,255)
            # torch.cuda.synchronize()
            # t2=time.time()
            # print(t2-t1)
            # infer_time+=t2-t1
            #loss=criterion(pred1,label)
            psnr=calcPSNR(pred1[0].data.cpu().numpy(),label[0].data.cpu().numpy())
            ssim=calcSSIM(pred1[0].data.cpu().numpy().transpose([1,2,0]),label[0].data.cpu().numpy().transpose([1,2,0]))
            psnr_all+=psnr
            ssim_all+=ssim
            iter+=1
            if (iter<=30):
                # cv.imwrite("ours/input%d.png"%(iter),cur_rgb[0,0:3].data.cpu().numpy().transpose([1,2,0]))
                cv.imwrite("ours/pred%d.png"%(iter),pred1[0].data.cpu().numpy().transpose([1,2,0]))
                # cv.imwrite("ours/label%d.png"%(iter),label[0].data.cpu().numpy().transpose([1,2,0]))
                # print(iter,psnr,ssim)
            
            # print(psnr,ssim)
        endTime = time.time()
        print("total time is %f"%(endTime - startTime))
        print("ssim: %f,psnr: %f"%(ssim_all/iter,psnr_all/iter))
if __name__ =="__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    testDataset=superTestDataset(0)
    testLoader = data.DataLoader(testDataset,1,shuffle=False,num_workers=6, pin_memory=True)
    
    test("D:/Python-Test/Models/MD-supernew-net-model.pth",testLoader)
    