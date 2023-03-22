import torch
import torch.nn as nn
import math
import onnx
import onnxruntime
import numpy as np
import time
def make_model(args, parent=False):
    return RCAN(args)

## Channel Attention (CA) Layer
class CALayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(CALayer, self).__init__()
        # global average pooling: feature --> point
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # feature channel downscale and upscale --> channel weight
        self.conv_du = nn.Sequential(
                nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=True),
                nn.ReLU(inplace=True),
                nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=True),
                nn.Sigmoid()
        )

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv_du(y)
        return x * y

## Residual Channel Attention Block (RCAB)
class RCAB(nn.Module):
    def __init__(
        self, conv, n_feat, kernel_size, reduction,
        bias=True, bn=False, act=nn.ReLU(True), res_scale=1):

        super(RCAB, self).__init__()
        modules_body = []
        for i in range(2):
            modules_body.append(conv(n_feat, n_feat, kernel_size, bias=bias))
            if bn: modules_body.append(nn.BatchNorm2d(n_feat))
            if i == 0: modules_body.append(act)
        modules_body.append(CALayer(n_feat, reduction))
        self.body = nn.Sequential(*modules_body)
        self.res_scale = res_scale

    def forward(self, x):
        res = self.body(x)
        #res = self.body(x).mul(self.res_scale)
        res += x
        return res

## Residual Group (RG)
class ResidualGroup(nn.Module):
    def __init__(self, conv, n_feat, kernel_size, reduction, act, res_scale, n_resblocks):
        super(ResidualGroup, self).__init__()
        modules_body = []
        modules_body = [
            RCAB(
                conv, n_feat, kernel_size, reduction, bias=True, bn=False, act=nn.ReLU(True), res_scale=1) \
            for _ in range(n_resblocks)]
        modules_body.append(conv(n_feat, n_feat, kernel_size))
        self.body = nn.Sequential(*modules_body)

    def forward(self, x):
        res = self.body(x)
        res += x
        return res
def default_conv(in_channels, out_channels, kernel_size, bias=True):
    return nn.Conv2d(
        in_channels, out_channels, kernel_size,
        padding=(kernel_size//2), bias=bias)
class Upsampler(nn.Sequential):
    def __init__(self, conv, scale, n_feat, bn=False, act=False, bias=True):

        m = []
        if (scale & (scale - 1)) == 0:    # Is scale = 2^n?
            for _ in range(int(math.log(scale, 2))):
                m.append(conv(n_feat, 4 * n_feat, 3, bias))
                m.append(nn.PixelShuffle(2))
                if bn: m.append(nn.BatchNorm2d(n_feat))
                if act: m.append(act())
        elif scale == 3:
            m.append(conv(n_feat, 9 * n_feat, 3, bias))
            m.append(nn.PixelShuffle(3))
            if bn: m.append(nn.BatchNorm2d(n_feat))
            if act: m.append(act())
        else:
            raise NotImplementedError

        super(Upsampler, self).__init__(*m)
## Residual Channel Attention Network (RCAN)
class RCAN(nn.Module):
    def __init__(self, conv=default_conv):
        super(RCAN, self).__init__()
        
        n_resgroups = 10
        n_resblocks = 20
        n_feats = 64
        kernel_size = 3
        reduction = 16
        scale = 4
        act = nn.ReLU(True)        
        # define head module
        modules_head = [conv(3, n_feats, kernel_size)]

        # define body module
        modules_body = [
            ResidualGroup(
                conv, n_feats, kernel_size, reduction, act=act, res_scale=1, n_resblocks=n_resblocks) \
            for _ in range(n_resgroups)]

        modules_body.append(conv(n_feats, n_feats, kernel_size))

        # define tail module
        modules_tail = [
            Upsampler(conv, scale, n_feats, act=False),
            conv(n_feats, 3, kernel_size)]


        self.head = nn.Sequential(*modules_head)
        self.body = nn.Sequential(*modules_body)
        self.tail = nn.Sequential(*modules_tail)

    def forward(self, x):
        x = self.head(x)

        res = self.body(x)
        res += x

        x = self.tail(res)

        return x 

    def load_state_dict(self, state_dict, strict=False):
        own_state = self.state_dict()
        for name, param in state_dict.items():
            if name in own_state:
                if isinstance(param, nn.Parameter):
                    param = param.data
                try:
                    own_state[name].copy_(param)
                except Exception:
                    if name.find('tail') >= 0:
                        print('Replace pre-trained upsampler to new one...')
                    else:
                        raise RuntimeError('While copying the parameter named {}, '
                                           'whose dimensions in the model are {} and '
                                           'whose dimensions in the checkpoint are {}.'
                                           .format(name, own_state[name].size(), param.size()))
            elif strict:
                if name.find('tail') == -1:
                    raise KeyError('unexpected key "{}" in state_dict'
                                   .format(name))

        if strict:
            missing = set(own_state.keys()) - set(state_dict.keys())
            if len(missing) > 0:
                raise KeyError('missing keys in state_dict: "{}"'.format(missing))
def onnx_inference(model_path):
    """
    模型推理
    :param model_path: 
    :return: 
    """
    # data=np.load('G:/BK_Gbuffer/BK_test/Bunker.testdata.0000.npz')
    input=np.random.randn(1,3,270,480).astype(np.float32)
    # cur_gbuffer=np.random.randn(1,9,1080,1920).astype(np.float32)
    # 使用onnxruntime-gpu在GPU上进行推理
    session = onnxruntime.InferenceSession(model_path,providers=['CUDAExecutionProvider'])
                                        #    providers=[
                                        #        ("CUDAExecutionProvider", {  # 使用GPU推理
                                        #            "device_id": 0,
                                        #            "arena_extend_strategy": "kNextPowerOfTwo",
                                        #            "gpu_mem_limit": 20 * 1024 * 1024 * 1024,
                                        #            "cudnn_conv_algo_search": "EXHAUSTIVE",
                                        #            "do_copy_in_default_stream": True,
                                        #             "cudnn_conv_use_max_workspace": "1"    # 在初始化阶段需要占用好几G的显存
                                        #        }),
                                        #     #    "CPUExecutionProvider"       # 使用CPU推理
                                        #    ])
    # session = onnxruntime.InferenceSession(model_path)
    # 获取模型原始输入的字段名称
    # print(session.get_inputs())
    input_name0 = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name
    list=session.get_inputs()
    # 以字典方式将数据输入到模型中
    outputs = session.run([output_name], {input_name0: input})
    outputs = session.run([output_name], {input_name0: input})
    t0=time.time()
    for i in range(100):
        outputs = session.run([output_name], {input_name0: input})
    t1=time.time()
    print((t1-t0)/1000)
    # print(outputs)
model=RCAN()
model=model.eval()
model=model.to(torch.device("cuda:0"))
input=torch.randn(1,3,270,480).cuda()
ONNX_FILE_PATH='./rcan.onnx'
# torch.onnx.export(model,input,ONNX_FILE_PATH,opset_version=12)
# print('ok')
onnx_inference(ONNX_FILE_PATH)
# onnx_model=onnx.load(ONNX_FILE_PATH)
# onnx.checker.check_model(onnx_model)