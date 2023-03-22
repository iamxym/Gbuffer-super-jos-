import os
os.environ["OPENCV_IO_ENABLE_OPENEXR"]="1"
from multiprocessing import synchronize
import cv2 as cv
import numpy as np
import torch
import torch.nn.functional as F
import time
import random
# import onnx
# import net_bak
# import lpips
# import torch.nn as nn
# import tinycudann as tcnn
#from Small_UnetGated import warp
# class MLP(nn.Module):

#     def __init__(self, in_dim, out_dim, hidden_list=[128,128,128]):
#         super().__init__()

#         self.network = tcnn.Network(in_dim, out_dim, network_config={
# 			"otype": "FullyFusedMLP",               # Component type.
# 			"activation": 'ReLU',               # Activation of hidden layers.
# 			"output_activation": 'None',   # Activation of the output layer.
# 			"n_neurons": hidden_list[0],           # Neurons in each hidden layer. # May only be 16, 32, 64, or 128.
# 			"n_hidden_layers": len(hidden_list),   # Number of hidden layers.
# 			"feedback_alignment": False  # Use feedback alignment # [Lillicrap et al. 2016].
# 		})

#     def forward(self, x):
#         shape = x.shape[:-1]
#         # x=self.network(x.view(-1, x.shape[-1]))
#         x = self.network(x.view(-1, x.shape[-1]))
#         return x.view(*shape, -1)
def bilinearSample(px,py,img):
    if (not px in range(0,img.shape[0])) or (not py in range(0,img.shape[1])):
        return 0
    x1,y1=int(px),int(py)
    x2,y2=x1,y1+1
    x3,y3=x1+1,y1
    x4,y4=x1+1,y1+1
    if x4>=img.shape[0] or y4>=img.shape[1]:
        return img[img.shape[0],img.shape[1],:]
    return img[x1,y1,:]*(y1+1-py)*(x1+1-px)+img[x2,y2,:]*(py-y1)*(x1+1-px)+img[x3,y3,:]*(y1+1-py)*(px-x1)+img[x4,y4,:]*(py-y1)*(px-x1)
def torchImgWrap(img1,motion2):
    n,c,h,w=img1.shape
    dx,dy=torch.linspace(-1,1,w).to(img1.device),torch.linspace(-1,1,h).to(img1.device)
    grid_y, grid_x = torch.meshgrid(dy, dx)
    
    grid_x = grid_x.repeat(n,1,1)-(2*motion2[:,1]/(w))
    grid_y = grid_y.repeat(n,1,1)+(2*motion2[:,0]/(h))
    coord = torch.stack([grid_x, grid_y], dim=-1)
    res=F.grid_sample(img1, coord, padding_mode='zeros',align_corners=True)
    return res
def backwarp(tenInput, tenFlow):
    #tenHorizontal = torch.linspace(-1.0, 1.0, tenFlow.shape[3]).view(1, 1, 1, tenFlow.shape[3]).expand(tenFlow.shape[0], -1, tenFlow.shape[2], -1)
    #tenVertical = torch.linspace(-1.0, 1.0, tenFlow.shape[2]).view(1, 1, tenFlow.shape[2], 1).expand(tenFlow.shape[0], -1, -1, tenFlow.shape[3])

    #backwarp_tenGrid = torch.cat([ tenHorizontal, tenVertical ], 1)
    #tenFlow = torch.cat([ tenFlow[:, 0:1, :, :] / (tenInput.shape[3] / 2.0), tenFlow[:, 1:2, :, :] / (tenInput.shape[2] / 2.0) ], 1)
    #print(tenFlow.shape,backwarp_tenGrid.shape)
    print(tenFlow.shape)
    tenHorizontal = torch.linspace(0, tenFlow.shape[3]-1, tenFlow.shape[3]).view(1, 1, 1, tenFlow.shape[3]).expand(tenFlow.shape[0], -1, tenFlow.shape[2], -1)
    tenVertical = torch.linspace(0, tenFlow.shape[2]-1, tenFlow.shape[2]).view(1, 1, tenFlow.shape[2], 1).expand(tenFlow.shape[0], -1, -1, tenFlow.shape[3])
    
    backwarp_tenGrid = torch.cat([ tenHorizontal, tenVertical ], 1)
    tenFlow = torch.cat([ -tenFlow[:, 1:2, :, :], tenFlow[:, 0:1, :, :] ], 1)

    coord=backwarp_tenGrid + tenFlow
    coord=torch.cat([coord[:,0:1,:,:]/(tenFlow.shape[3]/2.0)-1,coord[:,1:2,:,:]/(tenFlow.shape[2]/2.0)-1],1)

    return torch.nn.functional.grid_sample(input=tenInput, grid=coord.permute(0, 2, 3, 1), mode='bilinear', padding_mode='zeros', align_corners=True)

def stdAccWarping(prev_img,prev_mv,step):#return a ndarray 'cur_img'
    if abs(step)!=len(prev_mv):
        print("accWarping Error!")
        return None

    output=prev_img
    if step<0:
        for i in range(len(prev_mv)):
            output=torchImgWrap(output,prev_mv[i])
    else:    
        for i in range(len(prev_mv)-1,-1,-1):
            output=torchImgWrap(output,prev_mv[i])
    return output
def zeroUpsampling(img,scale):#x:(4 or 12) channels,and return a tensor batchsize*2*(h*3)*(w*3) 'y' 
    '''pool=nn.MaxPool2d(config.hs,stride=config.hs,return_indices=True)
    unpool=nn.MaxUnpool2d(config.hs,stride=config.hs)
    tmp=torch.rand(img.shape[0],img.shape[1],img.shape[2]*config.hs,img.shape[3]*config.ws).cuda()
    _,indices=pool(tmp)
    output=unpool(img,indices)
    return output'''
    b,c,h,w=img.shape
    tmp=torch.randn(b,c,h*scale,w*scale,device=img.device)
    up_img=F.interpolate(img,scale_factor=scale,mode='nearest')
    up_img=F.unfold(up_img,kernel_size=scale,stride=scale).transpose(1,2).reshape(b,h*w,c,scale**2)
    
    #return output
    id=F.unfold(tmp,kernel_size=scale,stride=scale).transpose(1,2).reshape(b,h*w,c,scale**2)
    value=torch.max(id,dim=-1)[0]
    value=value.unsqueeze(-1).expand(-1,-1,-1,scale**2)
    up_img[id!=value]=0
    #print(indices.shape,indices)
    
    #up_img[:,:,:,indices]=-1
    output=F.fold(up_img.reshape(b,h*w,c*scale**2).transpose(1,2),(h*scale,w*scale),kernel_size=scale,stride=scale)
    
    return output
import onnxruntime
def onnx_inference(model_path):
    """
    模型推理
    :param model_path: 
    :return: 
    """
    # data=np.load('G:/BK_Gbuffer/BK_test/Bunker.testdata.0000.npz')
    cur_rgb=np.random.randn(1,3,270,480).astype(np.float32)
    cur_gbuffer=np.random.randn(1,9,1080,1920).astype(np.float32)
    # 使用onnxruntime-gpu在GPU上进行推理
    session = onnxruntime.InferenceSession(model_path,
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
    # session = onnxruntime.InferenceSession(model_path)
 
    # 获取模型原始输入的字段名称
    input_name0 = session.get_inputs()[0].name
    input_name1 = session.get_inputs()[1].name
    output_name = session.get_outputs()[0].name
    list=session.get_inputs()
    # 以字典方式将数据输入到模型中
    outputs = session.run([output_name], {input_name0: cur_rgb,input_name1:cur_gbuffer})
    outputs = session.run([output_name], {input_name0: cur_rgb,input_name1:cur_gbuffer})
    t0=time.time()
    
    
    for i in range(100):
        outputs = session.run([output_name], {input_name0: cur_rgb,input_name1:cur_gbuffer})
    t1=time.time()
    print((t1-t0)/1000)
    # print(outputs)
def Gen_onnx(pthPath='../Models/MD-supernew-net-model.pth'):
    # model=net_bak.superNet()
    model=MLP(100,3)
    model=model.to(torch.device("cuda:0")).cuda()
    # modelPath='./Models/BK-super-net-model.pth'
    model_dict=torch.load(pthPath, map_location="cuda:0")
    model.load_state_dict(model_dict)
    ONNX_FILE_PATH="./tcnn.onnx"
    # data=np.load('G:/BK_Gbuffer/BK_test/Bunker.testdata.0000.npz')
    # label=torch.tensor(data['label']).float().unsqueeze(0).cuda()
    # cur_rgb=torch.tensor(data['cur_rgb']).float().unsqueeze(0).cuda()
    # cur_gbuffer=torch.tensor(data['cur_gbuffer']).float().unsqueeze(0).cuda()
    cur_rgb=torch.randn(1920*1080,100).cuda()
    # cur_gbuffer=torch.randn(1920*1080,3).cuda()
    # x1=torch.randn(1,16,1080,1920).cuda()
    # y1=torch.randn(1,16,1080,1920).cuda()
    # x2=torch.randn(1,32,540,960).cuda()
    # y2=torch.randn(1,32,540,960).cuda()
    # x3=torch.randn(1,64,270,480).cuda()
    # y3=torch.randn(1,64,270,480).cuda()
    # f1=torch.randn(1,32,1080,1920).cuda()
    # f2=torch.randn(1,64,540,960).cuda()
    # f3=torch.randn(1,128,270,480).cuda()
    torch.onnx.export(model,cur_rgb,ONNX_FILE_PATH,export_params=True,opset_version=11,verbose=True)

    onnx_model=onnx.load(ONNX_FILE_PATH)
    onnx.checker.check_model(onnx_model)
    return ONNX_FILE_PATH
def make_coord(shape, ranges=None, flatten=True):
    """ Make coordinates at grid centers.
    """
    coord_seqs = []
    for i, n in enumerate(shape):
        if ranges is None:
            v0, v1 = -1, 1
        else:
            v0, v1 = ranges[i]
        r = (v1 - v0) / (2 * n)
        seq = v0 + r + (2 * r) * torch.arange(n).float()
        coord_seqs.append(seq)
    ret = torch.stack(torch.meshgrid(*coord_seqs), dim=-1)
    if flatten:
        ret = ret.view(-1, ret.shape[-1])
    return ret
# model=MLP(100,3)
# model=model.to(torch.device("cuda:0"))
# torch.save(model.state_dict(), "./my_test.pth")
# path=Gen_onnx('./my_test.pth')
# print(path)

def myWarp(img,mv):
    img=torch.tensor(img.transpose([2,0,1])).unsqueeze(0)
    mv=torch.tensor(mv.transpose([2,0,1])).unsqueeze(0)
    res=torchImgWrap(img,mv)
    return res[0].numpy().transpose([1,2,0])


path='H:/Data/Demonstration/FE_Test_2/1440p/'
respath='H:/0-final14_all/Results/FE_Test_2_Shadow/'

warpfix='DemonstrationWarp.'
gtfix='DemonstrationGT.'
prefix='DemonstrationPreTonemapHDRColor.'
mvfix='DemonstrationMotionVector.'
for i in range(6,30):
    img0=cv.imread(path+prefix+str(i-1).zfill(4)+'.exr',cv.IMREAD_UNCHANGED)
    img1=cv.imread(path+prefix+str(i).zfill(4)+'.exr',cv.IMREAD_UNCHANGED)
    mv=cv.imread(path+mvfix+str(i).zfill(4)+'.exr',cv.IMREAD_UNCHANGED)[:,:,1:3]
    img0=myWarp(img0,mv)
    # print(img0)
    mask=cv.imread(path+'warp_res/'+warpfix+str(i).zfill(4)+'.1.exr',cv.IMREAD_UNCHANGED)[:,:,np.newaxis]
    res=(np.abs(img1-img0)*mask)
    # res=cv.GaussianBlur(res,(5,5),15)
    shadow_mask=res[...,0:1]+res[...,1:2]+res[...,2:3]
    cv.imwrite(respath+str(i)+'.exr',shadow_mask)
    
    # mv=cv.imread(path+)

# onnx_inference('./xiaoyimi_concat.onnx')

# model=net_bak.superNet()
# model=model.to(torch.device("cuda:0"))
# model._initialize_weights()
# torch.save(model.state_dict(), "./Models/my_test.pth")
# modelPath=Gen_onnx()
# modelPath="./xiaoyimi.onnx"
# onnx_inference(modelPath)
