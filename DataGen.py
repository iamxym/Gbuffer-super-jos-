import numpy as np
import os
import cv2 as cv
import config
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
    path='E:/Upsample_TrainData/DemoScene'
    list=[]
    for id in range(2,102):
        img=cv.imread(path+"PreTonemapHDRColor.%s.png"%(str(id).zfill(4)),cv.IMREAD_UNCHANGED)
        depth=cv.imread(path+"SceneDepth.%s.png"%(str(id).zfill(4)),cv.IMREAD_UNCHANGED)
        mv=cv.imread(path+"MotionVector.%s.exr"%(str(id).zfill(4)),cv.IMREAD_UNCHANGED)
        #print("MedievalDocksPreTonemapHDRColor.0%d.exr"%(id))
        if id==2:
            print(mv,depth)
        '''rgbdmv=np.concatenate((img[:,:,0:3],depth[:,:,0:1],mv[:,:,1:3]),axis=2)
        list.append(rgbdmv)
        if (len(list)>5):
            del list[0]
        if (len(list)==5):
            for i in range(3,-1,-1):
                rgbdmv=np.concatenate((rgbdmv,list[i]),axis=2)

            data=np.load(path+"MedievalDocks.traindata.{}.npy".format(idx))
            label=torch.tensor(data[:,:,0:3].copy().transpose([2,0,1]))
            data=cv.boxFilter(data,-1,(3,3))
            data=cv.resize(data,(data.shape[1]//3,data.shape[0]//3))
            data=data.transpose([2,0,1])
            #print(data.shape)
            cur_rgbd=torch.tensor(npToneSimple(data[0:4,:,:]))
            #cur_mv=torch.tensor((data[4:6,:,:]/3))
            prev_rgbd=torch.cat([torch.tensor(npToneSimple(data[6:10,:,:])),torch.tensor(npToneSimple(data[12:16,:,:])),torch.tensor(npToneSimple(data[18:22,:,:])),torch.tensor(npToneSimple(data[24:28,:,:]))],0).reshape(config.previous_frames,4,cur_rgbd.shape[1],cur_rgbd.shape[2])
            #prev_mv_list=[torch.tensor(data[10:12,:,:]/3),torch.tensor(data[16:18,:,:]/3),torch.tensor(data[22:24,:,:]/3),torch.tensor(data[28:30,:,:]/3)]
            mv=torch.cat([torch.tensor((data[4:6,:,:]/3)),torch.tensor(data[10:12,:,:]/3),torch.tensor(data[16:18,:,:]/3),torch.tensor(data[22:24,:,:]/3),torch.tensor(data[28:30,:,:]/3)],0)'''
           # np.save(savePath+"DemoScene.traindata.%s"%(str(id-6).zfill(4)),rgbdmv)
        #print(rgbd.shape)
        #print(rgbd.shape)
        #print(mv[540,960])
        #np.save(savePath+"DemoScene.rgbdmv.%s"%(str(id).zfill(4)),rgbdmv)
    '''for id in range(6,1509):
        data=np.load(savePath+"DemoScene.rgbdmv.0%d.npy"%(id))
        for i in range(1,5):
            data=np.concatenate((data,np.load("../rgbdmv/MedievalDocks.rgbdmv.0%d.npy"%(id-i))),axis=2)
        np.save(savePath+"MedievalDocks.traindata.%04d"%(total),data)
        total=total+1'''

    '''total=0
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
        total=total+1'''
def testDataGen():
    path="C:/train_set/DemoScene"
    savepath="C:/train_set/gbuffer_training/"
    list=[]
    for id in range(1,106):
        rgb=cv.imread(path+"PreTonemapHDRColor.%s.png"%(str(id).zfill(4)),cv.IMREAD_UNCHANGED)[:,:,0:3]
        depth=cv.imread(path+"SceneDepth.%s.png"%(str(id).zfill(4)),cv.IMREAD_UNCHANGED)[:,:,0:1]
        mv=cv.imread(path+"MotionVector.%s.exr"%(str(id).zfill(4)),cv.IMREAD_UNCHANGED)[:,:,1:3]
        albedo=cv.imread(path+"BaseColor.%s.png"%(str(id).zfill(4)),cv.IMREAD_UNCHANGED)[:,:,0:3]
        normal=cv.imread(path+"WorldNormal.%s.png"%(str(id).zfill(4)),cv.IMREAD_UNCHANGED)[:,:,0:3]
        roughness=cv.imread(path+"Roughness.%s.png"%(str(id).zfill(4)),cv.IMREAD_UNCHANGED)[:,:,0:1]
        metallic=cv.imread(path+"Metallic.%s.png"%(str(id).zfill(4)),cv.IMREAD_UNCHANGED)[:,:,0:1]
        data=np.concatenate((rgb,depth,mv,albedo,normal,roughness,metallic),axis=2)
        list.append(data[:,:,0:6])
        if (len(list)>5):
            del list[0]
        if (len(list)==5):
            label=list[4][:,:,0:3].copy().transpose([2,0,1])
            
            total=np.concatenate((data[:,:,0:3],list[3][:,:,0:4],list[2][:,:,0:4],list[1][:,:,0:4],list[0][:,:,0:4]),axis=2)
            mv=np.concatenate((data[:,:,4:6],list[3][:,:,4:6],list[2][:,:,4:6],list[1][:,:,4:6]),axis=2)
            cur_gbuffer=np.concatenate((data[:,:,3:4],data[:,:,6:]),axis=2)
            total=cv.boxFilter(total,-1,(config.hs,config.ws))
            total=cv.resize(total,(total.shape[1]//config.ws,total.shape[0]//config.hs))
            total=total.transpose([2,0,1])
            cur_rgb=total[0:3]
            prev_rgbd=total[3:]
            cur_gbuffer=cur_gbuffer.transpose([2,0,1])
            if id<9:
                print(cur_rgb.shape,prev_rgbd.shape)
            mv=cv.boxFilter(mv,-1,(config.hs,config.ws))
            mv=cv.resize(mv,(mv.shape[1]//config.ws,mv.shape[0]//config.hs))/config.hs
            mv=mv.transpose([2,0,1])
            #cur_gbuffer=cv.boxFilter(cur_gbuffer,-1,(config.hs,config.ws))
            #cur_gbuffer=cv.resize(cur_gbuffer,(cur_gbuffer.shape[1]//config.ws,cur_gbuffer.shape[0]//config.hs))
            #cur_gbuffer=cur_gbuffer.transpose([2,0,1])
            #print(id)
            np.savez(savepath+"DemoScene.testdata.%s"%(str(id-5).zfill(4)),label=label,cur_rgb=cur_rgb,mv=mv,
                                                                            cur_gbuffer=cur_gbuffer,prev_rgbd=prev_rgbd)
            #print(id)
            #np.savez(savepath+"DemoScene.testdata.%s.1"%(str(id-6).zfill(4)),label=label[:,360:,:640],cur_rgb=cur_rgb[:,90:,:160],mv=mv[:,90:,:160],
            #                                                                cur_gbuffer=cur_gbuffer[:,360:,:640],prev_rgbd=prev_rgbd[:,:90,:160])
            #np.savez(savepath+"DemoScene.testdata.%s.2"%(str(id-6).zfill(4)),label=label[:,:360,640:],cur_rgb=cur_rgb[:,:90,160:],mv=mv[:,:90,160:],
            #                                                                cur_gbuffer=cur_gbuffer[:,:360,640:],prev_rgbd=prev_rgbd[:,:90,:160])
            #np.savez(savepath+"DemoScene.testdata.%s.3"%(str(id-6).zfill(4)),label=label[:,360:,640:],cur_rgb=cur_rgb[:,90:,160:],mv=mv[:,90:,160:],
            #                                                                cur_gbuffer=cur_gbuffer[:,360:,640:],prev_rgbd=prev_rgbd[:,:90,:160])
def saveCompressedData(savepath):
    path='E:/Upsample_TrainData/uncompress_test/'
    list=[]
    for id in range(96):
        data=np.load(path+"DemoScene.traindata.{}.npy".format(str(id).zfill(4)))
        data=data.transpose([2,0,1])
        np.savez(savepath+"DemoScene.traindata.{}.npz".format(str(id).zfill(4)),label=data[0:3,:,:],rgbd=np.concatenate((data[0:4,:,:],data[6:10,:,:],data[12:16,:,:],data[18:22,:,:],data[24:28,:,:]),axis=0),mv=np.concatenate((data[4:6,:,:],data[10:12,:,:],data[16:18,:,:],data[22:24,:,:],data[28:30,:,:]),axis=0))
def RFDataGen():
    path="G:/RF_Gbuffer/RedwoodForest"
    otherpath="G:/RF_Gbuffer/LR_RedwoodForest"
    file_total=2979

    list=np.arange(file_total,dtype=np.int32)
    np.random.seed(20000506)
    np.random.shuffle(list)
    for i in range(0,file_total):
        if i%100==1:
            print(i)
        datatype=None
        num=0
        if i<2500:
            datatype='traindata'
            num=i
            savepath="G:/RF_Gbuffer/RF_TAA_train/"
        else:
            datatype='testdata'
            num=i-2500
            savepath="G:/RF_Gbuffer/RF_TAA_test/"
        id=list[i]+1
        cur_rgb=cv.imread(otherpath+"PreTonemapHDRColor.%s.png"%(str(id).zfill(4)),cv.IMREAD_UNCHANGED)[:,:,0:3]
        label=cv.imread(path+"PreTonemapHDRColor.%s.png"%(str(id).zfill(4)),cv.IMREAD_UNCHANGED)[:,:,0:3]
        depth=cv.imread(path+"SceneDepth.%s.png"%(str(id).zfill(4)),cv.IMREAD_UNCHANGED)[:,:,0:1]
        #mv=cv.imread(path+"MotionVector.%s.exr"%(str(id).zfill(4)),cv.IMREAD_UNCHANGED)[:,:,1:3]
        albedo=cv.imread(path+"BaseColor.%s.png"%(str(id).zfill(4)),cv.IMREAD_UNCHANGED)[:,:,0:3]
        normal=cv.imread(path+"WorldNormal.%s.png"%(str(id).zfill(4)),cv.IMREAD_UNCHANGED)[:,:,0:3]
        roughness=cv.imread(path+"Roughness.%s.png"%(str(id).zfill(4)),cv.IMREAD_UNCHANGED)[:,:,0:1]
        metallic=cv.imread(path+"Metallic.%s.png"%(str(id).zfill(4)),cv.IMREAD_UNCHANGED)[:,:,0:1]
        label=label.transpose([2,0,1])
        cur_rgb=cur_rgb.transpose([2,0,1])
        if i==1:
            print(cur_rgb.shape)
        cur_gbuffer=np.concatenate((depth,albedo,normal,roughness,metallic),axis=2).transpose([2,0,1])
        sw=640
        sh=360
        if i<2500:
            for j in range(3):
                for k in range(3):
                    np.savez(savepath+"RedwoodForest.%s.%s.%d"%(datatype,str(num).zfill(4),j*3+k),label=label[:,j*sh:(j+1)*sh,k*sw:(k+1)*sw],cur_rgb=cur_rgb[:,j*sh//config.hs:(j+1)*sh//config.hs,k*sw//config.ws:(k+1)*sw//config.ws],
                                                                            cur_gbuffer=cur_gbuffer[:,j*sh:(j+1)*sh,k*sw:(k+1)*sw])
        else:
            np.savez(savepath+"RedwoodForest.%s.%s"%(datatype,str(num).zfill(4)),label=label,cur_rgb=cur_rgb,
                                                                            cur_gbuffer=cur_gbuffer)
        #data=np.concatenate((rgb,depth,albedo,normal,roughness,metallic),axis=2)        
        #list.append(data[:,:,0:6])
        # if (len(list)>5):
        #     del list[0]
        # if (len(list)==5):
        #     label=list[4][:,:,0:3].copy().transpose([2,0,1])
            
        #     total=np.concatenate((data[:,:,0:3],list[3][:,:,0:4],list[2][:,:,0:4],list[1][:,:,0:4],list[0][:,:,0:4]),axis=2)
        #     mv=np.concatenate((data[:,:,4:6],list[3][:,:,4:6],list[2][:,:,4:6],list[1][:,:,4:6]),axis=2)
        #     cur_gbuffer=np.concatenate((data[:,:,3:4],data[:,:,6:]),axis=2)
        #     total=cv.boxFilter(total,-1,(config.hs,config.ws))
        #     total=cv.resize(total,(total.shape[1]//config.ws,total.shape[0]//config.hs))
        #     total=total.transpose([2,0,1])
        #     cur_rgb=total[0:3]
        #     prev_rgbd=total[3:]
        #     cur_gbuffer=cur_gbuffer.transpose([2,0,1])
        #     if id<9:
        #         print(cur_rgb.shape,prev_rgbd.shape)
        #     #mv=cv.boxFilter(mv,-1,(config.hs,config.ws))
        #     #mv=cv.resize(mv,(mv.shape[1]//config.ws,mv.shape[0]//config.hs))/config.hs
        #     #mv=mv.transpose([2,0,1])
        #     #cur_gbuffer=cv.boxFilter(cur_gbuffer,-1,(config.hs,config.ws))
        #     #cur_gbuffer=cv.resize(cur_gbuffer,(cur_gbuffer.shape[1]//config.ws,cur_gbuffer.shape[0]//config.hs))
        #     #cur_gbuffer=cur_gbuffer.transpose([2,0,1])
        #     np.savez(savepath+"DemoScene.traindata.%s.0"%(str(id-6).zfill(4)),label=label[:,:360,:640],cur_rgb=cur_rgb[:,:90,:160],mv=mv[:,:90,:160],
        #                                                                     cur_gbuffer=cur_gbuffer[:,:360,:640],prev_rgbd=prev_rgbd[:,:90,:160])
        #     np.savez(savepath+"DemoScene.traindata.%s.1"%(str(id-6).zfill(4)),label=label[:,360:,:640],cur_rgb=cur_rgb[:,90:,:160],mv=mv[:,90:,:160],
        #                                                                     cur_gbuffer=cur_gbuffer[:,360:,:640],prev_rgbd=prev_rgbd[:,:90,:160])
        #     np.savez(savepath+"DemoScene.traindata.%s.2"%(str(id-6).zfill(4)),label=label[:,:360,640:],cur_rgb=cur_rgb[:,:90,160:],mv=mv[:,:90,160:],
        #                                                                     cur_gbuffer=cur_gbuffer[:,:360,640:],prev_rgbd=prev_rgbd[:,:90,:160])
        #     np.savez(savepath+"DemoScene.traindata.%s.3"%(str(id-6).zfill(4)),label=label[:,360:,640:],cur_rgb=cur_rgb[:,90:,160:],mv=mv[:,90:,160:],
        #                                                                     cur_gbuffer=cur_gbuffer[:,360:,640:],prev_rgbd=prev_rgbd[:,:90,:160])
def BKDataGen():
    path="D:/UE4dataset_tmp/Bunker"
    #~开showflag.motionblur了吗？？？？
    otherpath="D:/UE4dataset_tmp/LR_Bunker"
    file_total=2999

    list=np.arange(file_total,dtype=np.int32)
    np.random.seed(20000506)
    np.random.shuffle(list)
    for i in range(0,file_total):
        if i%100==1:
            print(i)
        datatype=None
        num=0
        if i<2500:
            datatype='traindata'
            num=i
            savepath="G:/BK_Gbuffer/BK_train/"
        else:
            datatype='testdata'
            num=i-2500
            savepath="G:/BK_Gbuffer/BK_test/"
        id=list[i]+1
        cur_rgb=cv.imread(otherpath+"PreTonemapHDRColor.%s.png"%(str(id).zfill(4)),cv.IMREAD_UNCHANGED)[:,:,0:3]
        label=cv.imread(path+"PreTonemapHDRColor.%s.png"%(str(id).zfill(4)),cv.IMREAD_UNCHANGED)[:,:,0:3]
        depth=cv.imread(path+"SceneDepth.%s.png"%(str(id).zfill(4)),cv.IMREAD_UNCHANGED)[:,:,0:1]
        #mv=cv.imread(path+"MotionVector.%s.exr"%(str(id).zfill(4)),cv.IMREAD_UNCHANGED)[:,:,1:3]
        albedo=cv.imread(path+"BaseColor.%s.png"%(str(id).zfill(4)),cv.IMREAD_UNCHANGED)[:,:,0:3]
        normal=cv.imread(path+"WorldNormal.%s.png"%(str(id).zfill(4)),cv.IMREAD_UNCHANGED)[:,:,0:3]
        roughness=cv.imread(path+"Roughness.%s.png"%(str(id).zfill(4)),cv.IMREAD_UNCHANGED)[:,:,0:1]
        metallic=cv.imread(path+"Metallic.%s.png"%(str(id).zfill(4)),cv.IMREAD_UNCHANGED)[:,:,0:1]
        label=label.transpose([2,0,1])
        cur_rgb=cur_rgb.transpose([2,0,1])
        if i==1:
            print(cur_rgb.shape)
        cur_gbuffer=np.concatenate((depth,albedo,normal,roughness,metallic),axis=2).transpose([2,0,1])
        sw=640
        sh=360
        if i<2500:
            for j in range(3):
                for k in range(3):
                    np.savez(savepath+"Bunker.%s.%s.%d"%(datatype,str(num).zfill(4),j*3+k),label=label[:,j*sh:(j+1)*sh,k*sw:(k+1)*sw],cur_rgb=cur_rgb[:,j*sh//config.hs:(j+1)*sh//config.hs,k*sw//config.ws:(k+1)*sw//config.ws],
                                                                            cur_gbuffer=cur_gbuffer[:,j*sh:(j+1)*sh,k*sw:(k+1)*sw])
        else:
            np.savez(savepath+"Bunker.%s.%s"%(datatype,str(num).zfill(4)),label=label,cur_rgb=cur_rgb,
                                                                            cur_gbuffer=cur_gbuffer)
'''img1=cv.imread('D:/Python-Test/DemoSceneBaseColor.1508.exr',cv.IMREAD_UNCHANGED)
img2=cv.imread('D:/Python-Test/DemoSceneWorldNormal.1508.exr',cv.IMREAD_UNCHANGED)
print(img1.shape,img2.shape)
print(img1[:,:,0:3].max())'''
#testDataGen()
#testDataGen()
# BKDataGen()
'''path='G:/RF_Gbuffer/Gbuffer_RFtest/RedwoodForest.testdata'
for i in range(500):
    
    sw=640
    sh=360
    label=np.zeros((3,1080,1920),dtype=np.uint8)
    cur_rgb=np.zeros((3,270,480),dtype=np.uint8)
    cur_gbuffer=np.zeros((9,1080,1920),dtype=np.uint8)
    for j in range(3):
        for k in range(3):
            data=np.load(path+".%s.%d.npz"%(str(i).zfill(4),j*3+k))
            label[:,j*sh:(j+1)*sh,k*sw:(k+1)*sw]=data['label']
            cur_rgb[:,j*sh//config.hs:(j+1)*sh//config.hs,k*sw//config.ws:(k+1)*sw//config.ws]=data['cur_rgb']
            cur_gbuffer[:,j*sh:(j+1)*sh,k*sw:(k+1)*sw]=data['cur_gbuffer']
    np.savez(path+'.'+str(i).zfill(4),label=label,cur_rgb=cur_rgb,cur_gbuffer=cur_gbuffer)
'''            

path="G:/DisplaySet/RF_displayDataset/"
start_n=301
for i in range(60):
    old="LR3_RedwoodForestMotionVector.%s.exr"%(str(i+start_n).zfill(4))
    new="LR_3_RedwoodForestMotionVector.%s.exr"%(str(i+start_n).zfill(4))
    os.rename(path+old,path+new)
    old="LR3_RedwoodForestSceneDepth.%s.png"%(str(i+start_n).zfill(4))
    new="LR_3_RedwoodForestSceneDepth.%s.png"%(str(i+start_n).zfill(4))
    os.rename(path+old,path+new)
    old="LR3_RedwoodForestPreTonemapHDRColor.%s.png"%(str(i+start_n).zfill(4))
    new="LR_3_RedwoodForestPreTonemapHDRColor.%s.png"%(str(i+start_n).zfill(4))
    os.rename(path+old,path+new)