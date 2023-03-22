import cv2 as cv
cv.setNumThreads(0) 
import numpy as np
import math
import os
import torch
import time
import torch.utils.data as data
import net_bak
class displayDataset(data.Dataset):
    # 创建LiverDataset类的实例时，就是在调用init初始化
    def __init__(self,net,predir,end_n,data='Bunker',extra_pre=None,start_n=5, transform=None):  # root表示图片路径
        
        self.totalNum = end_n-start_n+1
        self.start_n=start_n
        self.path=predir
        self.net=net
        self.data=data
        self.nul=torch.zeros(1)
        if (extra_pre==None):
            self.extra_pre=''
        else:
            self.extra_pre=extra_pre

    def __getitem__(self, index):
        id=str(index+self.start_n).zfill(4)
        
        if self.net=='ours':
            # albedo=cv.imread(self.path+self.extra_pre+self.data+'BaseColor.%s.png'%(id),cv.IMREAD_UNCHANGED)
            albedo=cv.imread("G:/DisplaySet/RF_aa/aa_RedwoodForestBaseColorAA.%s.png"%(id),cv.IMREAD_UNCHANGED)
            normal=cv.imread(self.path+self.extra_pre+self.data+'WorldNormal.%s.png'%(id),cv.IMREAD_UNCHANGED)
            # print(self.path+'LR_'+self.extra_pre+self.data+'PreTonemapHDRColor.%s.png'%(id))
            # print(self.path+self.extra_pre+self.data+'SceneDepth.%s.png'%(id))
            # print(self.path+self.extra_pre+self.data+'SceneDepth.%s.png'%(id))
            depth=cv.imread(self.path+self.extra_pre+self.data+'SceneDepth.%s.png'%(id))[:,:,0:1]
            
            
            roughness=cv.imread(self.path+self.extra_pre+self.data+'Roughness.%s.png'%(id))[:,:,0:1]
            metallic=cv.imread(self.path+self.extra_pre+self.data+'Metallic.%s.png'%(id))[:,:,0:1]
            # print(albedo.shape,normal.shape,depth.shape,roughness.shape,metallic.shape)
            # img=cv.imread(self.path+'LR_'+self.extra_pre+self.data+'PreTonemapHDRColor.%s.png'%(id),cv.IMREAD_UNCHANGED)
            img=cv.imread('G:/DisplaySet/RF_AA/lR_aa_'+self.extra_pre+self.data+'PreTonemapHDRColor.%s.png'%(id),cv.IMREAD_UNCHANGED)
            # img=cv.resize(img,dsize=(0,0),fx=1/2,fy=1/2,interpolation=cv.INTER_CUBIC)
            label=cv.imread(self.path+self.extra_pre+self.data+'PreTonemapHDRColor.%s.png'%(id),cv.IMREAD_UNCHANGED)
            label=torch.tensor(label.transpose([2,0,1])).float()
            cur_rgb=torch.tensor(img.transpose([2,0,1])).float()
            cur_gbuffer=torch.tensor(np.concatenate((depth,albedo,normal,roughness,metallic),axis=2).transpose([2,0,1])).float()
            return cur_rgb,cur_gbuffer,self.nul,label
        elif self.net=='NSRR':
            list=[]
            data=None
            for i in range(index+self.start_n-4,index+self.start_n+1):
                rgb=cv.imread(self.path+'LR_'+self.extra_pre+self.data+"PreTonemapHDRColor.%s.png"%(str(i).zfill(4)),cv.IMREAD_UNCHANGED)[:,:,0:3]
                depth=cv.imread(self.path+'LR_'+self.extra_pre+self.data+"SceneDepth.%s.png"%(str(i).zfill(4)),cv.IMREAD_UNCHANGED)[:,:,0:1]
                mv=cv.imread(self.path+'LR_'+self.extra_pre+self.data+"MotionVector.%s.exr"%(str(i).zfill(4)),cv.IMREAD_UNCHANGED)[:,:,1:3]
                label=cv.imread(self.path+self.extra_pre+self.data+'PreTonemapHDRColor.%s.png'%(str(i).zfill(4)),cv.IMREAD_UNCHANGED)
                data=np.concatenate((rgb,depth,mv),axis=2)
                list.append(data)
            label=torch.tensor(label.transpose([2,0,1])).float()
            total=np.concatenate((data[:,:,0:4],list[3][:,:,0:4],list[2][:,:,0:4],list[1][:,:,0:4],list[0][:,:,0:4]),axis=2)
            mv=np.concatenate((data[:,:,4:6],list[3][:,:,4:6],list[2][:,:,4:6],list[1][:,:,4:6]),axis=2)
            total=total.transpose([2,0,1])
            cur_rgbd=torch.tensor(total[0:4]).float()
            prev_rgbd=torch.tensor(total[4:]).float()
            mv=torch.tensor(mv.transpose([2,0,1]))
            return cur_rgbd,prev_rgbd,mv,label
        elif self.net=='RRN':
            pass
        else:
            img=cv.imread(self.path+'LR_'+self.extra_pre+self.data+'PreTonemapHDRColor.%s.png'%(id),cv.IMREAD_UNCHANGED)
            label=cv.imread(self.path+self.extra_pre+self.data+'PreTonemapHDRColor.%s.png'%(id),cv.IMREAD_UNCHANGED)
            label=torch.tensor(label.transpose([2,0,1])).float()
            cur_rgb=torch.tensor(img.transpose([2,0,1])).float()
            return cur_rgb,self.nul,self.nul,label

    def __len__(self):
        return self.totalNum

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
def calcPSNR(img1,img2):
    mse = np.mean( (img1/255. - img2/255.) ** 2 )
    if mse < 1.0e-10:
        return 100
    PIXEL_MAX = 1
    return 20 * math.log10(PIXEL_MAX / math.sqrt(mse))
def display(modelPath,dataset):
    dataLoader = data.DataLoader(dataset,1,shuffle=False,num_workers=1, pin_memory=False)
    savepath=dataset.path+dataset.data+'res/'
    # savepath=dataset.path+'MDres/'
    if not os.path.exists(savepath):
        os.mkdir(savepath)
    modelType=dataset.net
    model=None
    if modelType=="NSRR":
        model=net_bak.superNet()
    elif modelType=="ours":
        model=net_bak.superNet()
    elif modelType=="RCAN":
        model=rcan.RCAN()
    if model==None:
        return None
    model=model.to(torch.device("cuda:0"))
    model_dict=torch.load(modelPath, map_location="cuda:0")
    # iii=list(model_dict)
    # for i in range(len(model_dict)):
        # print(iii[i],iii[i].shape)
    model.load_state_dict(model_dict)
    
    model.eval()
    
    iter=0
    ssim_all=0
    ssim=0
    psnr=0
    psnr_all=0
    with torch.no_grad():
        for cur_rgb,a,b,label in dataLoader:
            cur_rgb=cur_rgb.cuda()
            # label=label.cuda()
            a=a.cuda()
            b=b.cuda()
            pred=None
            torch.cuda.synchronize()
            t1=time.time()
            if modelType=="NSRR":
                pred=model(cur_rgb,a,b) #the net's forward
            elif modelType=="ours":
                pred=model(cur_rgb,a)
            else:
                pred=model(cur_rgb)
            torch.cuda.synchronize()
            t2=time.time()
            # print("consume:",t2-t1)
            pred=pred.clip(0,255)
            psnr=0
            ssim=0
            psnr=calcPSNR(pred[0].data.cpu().numpy(),label[0].data.cpu().numpy())
            ssim=calcSSIM(pred[0].data.cpu().numpy().transpose([1,2,0]),label[0].data.cpu().numpy().transpose([1,2,0]))
            cv.imwrite(savepath+"aa-%s.png"%(dataset.start_n+iter),pred[0].data.cpu().numpy().transpose([1,2,0]))
            iter=iter+1
            psnr_all+=psnr
            ssim_all+=ssim
            # print(psnr,ssim)
        print("ssim: %f,psnr: %f"%(ssim_all/iter,psnr_all/iter))
if __name__ =="__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    displaydataset=displayDataset("ours","G:/DisplaySet/videoSet/RF/",start_n=130,end_n=401,data='RedwoodForest',extra_pre='')
    
    
    display("D:/Python-Test/Models/RF-supernew-net-model.pth",displaydataset)