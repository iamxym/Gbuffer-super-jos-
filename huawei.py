import cv2 as cv
cv.setNumThreads(0) 
import numpy as np
import os
import torch
import torch.utils.data as data
class displayDataset(data.Dataset):
    # 创建LiverDataset类的实例时，就是在调用init初始化
    def __init__(self,predir,end_n,data='Bunker',extra_pre=None,start_n=5, transform=None):  # root表示图片路径
        
        self.totalNum = end_n-start_n+1
        self.start_n=start_n
        self.path=predir
        self.data=data
        if (extra_pre==None):
            self.extra_pre=''
        else:
            self.extra_pre=extra_pre

    def __getitem__(self, index):
        id=str(index+self.start_n).zfill(4)
        
        albedo=cv.imread(self.path+self.extra_pre+self.data+'BaseColor.%s.png'%(id),cv.IMREAD_UNCHANGED)
        normal=cv.imread(self.path+self.extra_pre+self.data+'WorldNormal.%s.png'%(id),cv.IMREAD_UNCHANGED)
        depth=cv.imread(self.path+self.extra_pre+self.data+'SceneDepth.%s.png'%(id))[:,:,0:1]
        
        
        roughness=cv.imread(self.path+self.extra_pre+self.data+'Roughness.%s.png'%(id))[:,:,0:1]
        metallic=cv.imread(self.path+self.extra_pre+self.data+'Metallic.%s.png'%(id))[:,:,0:1]
        img=cv.imread(self.path+'LR_'+self.extra_pre+self.data+'PreTonemapHDRColor.%s.png'%(id),cv.IMREAD_UNCHANGED)
        label=cv.imread(self.path+self.extra_pre+self.data+'PreTonemapHDRColor.%s.png'%(id),cv.IMREAD_UNCHANGED)
        label=label.transpose([2,0,1]).astype(np.float32)
        cur_rgb=img.transpose([2,0,1]).astype(np.float32)
        cur_gbuffer=np.concatenate((depth,albedo,normal,roughness,metallic),axis=2).transpose([2,0,1]).astype(np.float32)
        return cur_rgb,cur_gbuffer,label

    def __len__(self):
        return self.totalNum
import onnxruntime
def display(modelPath,dataset):
    
    session = onnxruntime.InferenceSession(modelPath,
                                           providers=[
                                               ("CUDAExecutionProvider", {  # 使用GPU推理
                                                   "device_id": 0,
                                                   "arena_extend_strategy": "kNextPowerOfTwo",
                                                   "gpu_mem_limit": 16 * 1024 * 1024 * 1024,
                                                   "cudnn_conv_algo_search": "EXHAUSTIVE",
                                                   "do_copy_in_default_stream": True,
                                                    "cudnn_conv_use_max_workspace": "1"    # 在初始化阶段需要占用好几G的显存
                                               }),
                                            #    "CPUExecutionProvider"       # 使用CPU推理
                                           ]) 
    
    input_name0 = session.get_inputs()[0].name
    input_name1 = session.get_inputs()[1].name
    output_name = session.get_outputs()[0].name
    
    dataLoader = data.DataLoader(dataset,1,shuffle=False,num_workers=1, pin_memory=False)
    savepath=dataset.path+dataset.data+'res/'
    if not os.path.exists(savepath):
        os.mkdir(savepath)
    
    
    iter=0
    with torch.no_grad():
        for cur_rgb,cur_gbuffer,label in dataLoader:
            cur_rgb=cur_rgb.numpy()
            # label=label.cuda()
            cur_gbuffer=cur_gbuffer.numpy()
            
            pred=None
            pred = session.run([output_name], {input_name0: cur_rgb,input_name1:cur_gbuffer})[0]
            pred=pred.clip(0,255)
            cv.imwrite(savepath+"%s.png"%(dataset.start_n+iter),pred[0].transpose([1,2,0]))
            iter=iter+1
if __name__ =="__main__":
    path="G:/DisplaySet/RF_AA/"
    for i in range(130,401):
        img=cv.imread(path+"LR_aa_RedwoodForestPreTonemapHDRColor.%s.png"%(str(i).zfill(4)),cv.IMREAD_UNCHANGED)
        res=cv.resize(img,(0,0),fx=4,fy=4,interpolation=cv.INTER_LINEAR)
        cv.imwrite(path+"HR_aa_RedwoodForestPreTonemapHDRColor.%s.png"%(str(i).zfill(4)),res)
    # os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    # imgPath='G:/DisplaySet/videoSet/MD-2K/'
    # displaydataset=displayDataset(imgPath,start_n=0,end_n=297,data='MedievalDocks',extra_pre='')
    
    
    # display("./xiaoyimi_2K.onnx",displaydataset)