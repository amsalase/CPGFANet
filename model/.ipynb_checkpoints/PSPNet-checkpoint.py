# coding:utf-8
# Email: 2443434059@qq.com

import torch
import torch.nn as nn
import torchvision.models as models
from torch.nn import functional as F
from torch.nn import Softmax
from thop import profile
################ FEAM #########################################
# Non-local Neural Networks

def INF(B,H,W):
    return -torch.diag(torch.tensor(float("inf"), device=3).repeat(H),0).unsqueeze(0).repeat(B*W,1,1)


class CrissCrossAttention(nn.Module):
    """ Criss-Cross Attention Module"""
    def __init__(self, in_dim):
        super(CrissCrossAttention,self).__init__()
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.query_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim//16, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim//16, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.softmax = Softmax(dim=3)
        self.relu1 = nn.ReLU()
        self.INF = INF
        self.gamma = nn.Parameter(torch.zeros(1))


    def forward(self, x):
        x1 = self.max_pool(x)
        m_batchsize, _, height, width = x1.size()
        proj_query = self.query_conv(x1)
        proj_query = self.relu1(proj_query)
        proj_query_H = proj_query.permute(0,3,1,2).contiguous().view(m_batchsize*width,-1,height).permute(0, 2, 1)
        proj_query_W = proj_query.permute(0,2,1,3).contiguous().view(m_batchsize*height,-1,width).permute(0, 2, 1)
        proj_key = self.key_conv(x1)
        proj_key = self.relu1(proj_key)
        proj_key_H = proj_key.permute(0,3,1,2).contiguous().view(m_batchsize*width,-1,height)
        proj_key_W = proj_key.permute(0,2,1,3).contiguous().view(m_batchsize*height,-1,width)
        proj_value = self.value_conv(x1)
        proj_value = self.relu1(proj_value)
        proj_value_H = proj_value.permute(0,3,1,2).contiguous().view(m_batchsize*width,-1,height)
        proj_value_W = proj_value.permute(0,2,1,3).contiguous().view(m_batchsize*height,-1,width)
        energy_H = (torch.matmul(proj_query_H, proj_key_H)+self.INF(m_batchsize, height, width)).view(m_batchsize,width,height,height).permute(0,2,1,3)
        energy_W = torch.matmul(proj_query_W, proj_key_W).view(m_batchsize,height,width,width)
        concate = self.softmax(torch.cat([energy_H, energy_W], 3))
        att_H = concate[:,:,:,0:height].permute(0,2,1,3).contiguous().view(m_batchsize*width,height,height)
        att_W = concate[:,:,:,height:height+width].contiguous().view(m_batchsize*height,width,width)
        out_H = torch.matmul(proj_value_H, att_H.permute(0, 2, 1)).view(m_batchsize,width,-1,height).permute(0,2,3,1)
        out_W = torch.matmul(proj_value_W, att_W.permute(0, 2, 1)).view(m_batchsize,height,-1,width).permute(0,2,1,3)
        return self.gamma*(out_H + out_W) + x
        #return self.gamma*(out_H + out_W)


class RCCAModule(nn.Module):
    def __init__(self, in_channels):
        super(RCCAModule, self).__init__()
        self.cca = CrissCrossAttention(in_channels)
              
    def forward(self, x, recurrence=1):
        output = x
        for i in range(recurrence):
            output = self.cca(output)
        return output


class PSPModule(nn.Module):
    def __init__(self, features, out_features=1024, sizes=(1, 2, 3, 6)):
        super().__init__()
        self.stages = []
        self.stages = nn.ModuleList([self._make_stage(features, size) for size in sizes])
        self.bottleneck = nn.Conv2d(features * (len(sizes) + 1), out_features, kernel_size=1)
        self.relu = nn.ReLU()

    def _make_stage(self, features, size):
        prior = nn.AdaptiveAvgPool2d(output_size=(size, size))
        conv = nn.Conv2d(features, features, kernel_size=1, bias=False)
        return nn.Sequential(prior, conv)

    def forward(self, feats):
        h, w = feats.size(2), feats.size(3)
        priors = [F.upsample(input=stage(feats), size=(h, w), mode='bilinear') for stage in self.stages] + [feats]
        bottle = self.bottleneck(torch.cat(priors, 1))
        return self.relu(bottle)


class PSPUpsample(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.PReLU()
        )

    def forward(self, x):
        h, w = 2 * x.size(2), 2 * x.size(3)
        p = F.upsample(input=x, size=(h, w), mode='bilinear')
        return self.conv(p)


class PSPNet(nn.Module):
    def __init__(self, n_classes=9, sizes=(1, 2, 3, 6), psp_size=2048, deep_features_size=1024):
        super().__init__()
        self.psp = PSPModule(psp_size, 1024, sizes)
        self.drop_1 = nn.Dropout2d(p=0.3)

        self.up_1 = PSPUpsample(1024, 512)
        self.up_2 = PSPUpsample(512, 256)
        self.up_3 = PSPUpsample(256, 128)
        self.up_4 = PSPUpsample(128, 64)
        self.up_5 = PSPUpsample(64, 64)

        self.drop_2 = nn.Dropout2d(p=0.15)
        self.final = nn.Sequential(
            nn.Conv2d(64, n_classes, kernel_size=1),
            nn.LogSoftmax()
        )

        self.classifier = nn.Sequential(
            nn.Linear(deep_features_size, 256),
            nn.ReLU(),
            nn.Linear(256, n_classes)
        )

    def forward(self, x):
        f = x 
        p = self.psp(f)
        p = self.drop_1(p)

        p = self.up_1(p)
        p = self.drop_2(p)

        p = self.up_2(p)
        p = self.drop_2(p)

        p = self.up_3(p)
        p = self.drop_2(p)
        
        p = self.up_4(p)
        p = self.drop_2(p)

        p = self.up_5(p)
        p = self.drop_2(p)


        return self.final(p)


class FEANet(nn.Module):

    def __init__(self, n_class):
        super(FEANet, self).__init__()

        self.num_resnet_layers = 152

        if self.num_resnet_layers == 50:
            resnet_raw_model1 = models.resnet50(pretrained=True)
            resnet_raw_model2 = models.resnet50(pretrained=True)
            self.inplanes = 2048
        elif self.num_resnet_layers == 101:
            resnet_raw_model1 = models.resnet101(pretrained=True)
            resnet_raw_model2 = models.resnet101(pretrained=True)
            self.inplanes = 2048
        elif self.num_resnet_layers == 152:
            resnet_raw_model1 = models.resnet152(pretrained=True)
            resnet_raw_model2 = models.resnet152(pretrained=True)
            self.inplanes = 2048
            
        ########  Thermal ENCODER  ########

        self.encoder_thermal_conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.encoder_thermal_conv1.weight.data = torch.unsqueeze(torch.mean(resnet_raw_model1.conv1.weight.data, dim=1),
                                                                 dim=1)
        self.encoder_thermal_bn1 = resnet_raw_model1.bn1
        self.encoder_thermal_relu = resnet_raw_model1.relu
        self.encoder_thermal_maxpool = resnet_raw_model1.maxpool
        self.encoder_thermal_layer1 = resnet_raw_model1.layer1
        self.encoder_thermal_layer2 = resnet_raw_model1.layer2
        self.encoder_thermal_layer3 = resnet_raw_model1.layer3
        self.encoder_thermal_layer4 = resnet_raw_model1.layer4
      
        self.atten_RCCAModule_0 = RCCAModule(64)
        self.atten_RCCAModule_1 = RCCAModule(256)
        self.atten_RCCAModule_2 = RCCAModule(512)
        self.atten_RCCAModule_3_1 = RCCAModule(1024)
        self.atten_RCCAModule_4_1 = RCCAModule(2048)
 
        
        ########  RGB ENCODER  ########
        self.encoder_rgb_conv1 = resnet_raw_model2.conv1
        self.encoder_rgb_bn1 = resnet_raw_model2.bn1
        self.encoder_rgb_relu = resnet_raw_model2.relu
        self.encoder_rgb_maxpool = resnet_raw_model2.maxpool
        self.encoder_rgb_layer1 = resnet_raw_model2.layer1
        self.encoder_rgb_layer2 = resnet_raw_model2.layer2
        self.encoder_rgb_layer3 = resnet_raw_model2.layer3
        self.encoder_rgb_layer4 = resnet_raw_model2.layer4

        ########  DECODER  ########

        self.deconv = PSPNet(n_classes=9, sizes=(1, 2, 3, 6), psp_size=2048, deep_features_size=1024)
         
    def forward(self, input):

        rgb = input[:, :3]
        thermal = input[:, 3:]

        verbose = False

        # encoder

        ######################################################################
        if verbose: print("rgb.size() original: ", rgb.size())  # (480, 640)
        if verbose: print("thermal.size() original: ", thermal.size())  # (480, 640)
        ######################################################################
        rgb = self.encoder_rgb_conv1(rgb)
        if verbose: print("rgb.size() after conv1: ", rgb.size())  # (240, 320)
        rgb = self.encoder_rgb_bn1(rgb)
        if verbose: print("rgb.size() after bn1: ", rgb.size())  # (240, 320)
        rgb = self.encoder_rgb_relu(rgb)
        if verbose: print("rgb.size() after relu: ", rgb.size())  # (240, 320)
        thermal = self.encoder_thermal_conv1(thermal)
        if verbose: print("thermal.size() after conv1: ", thermal.size())  # (240, 320)
        thermal = self.encoder_thermal_bn1(thermal)
        if verbose: print("thermal.size() after bn1: ", thermal.size())  # (240, 320)
        thermal = self.encoder_thermal_relu(thermal)
        if verbose: print("thermal.size() after relu: ", thermal.size())  # (240, 320)
        ######################################################################
        # Des-comentar para usar los bloques RCCAModule en imagen termica y RGB
        rgb  = self.atten_RCCAModule_0(rgb)
        temp = self.atten_RCCAModule_0(thermal)
        rgb_aux = torch.cat((rgb,temp),1)
        rgb = rgb + temp
        ######################################################################
        # rgb = rgb + thermal #comentar para usar los bloques RCCAModule en imagen termica y RGB
        ######################################################################
        rgb = self.encoder_rgb_maxpool(rgb)
        if verbose: print("rgb.size() after maxpool: ", rgb.size())  # (120, 160)
        thermal = self.encoder_thermal_maxpool(thermal)
        if verbose: print("thermal.size() after maxpool: ", thermal.size())  # (120, 160)
        ######################################################################
        rgb = self.encoder_rgb_layer1(rgb)
        if verbose: print("rgb.size() after layer1: ", rgb.size())  # (120, 160)
        thermal = self.encoder_thermal_layer1(thermal)
        if verbose: print("thermal.size() after layer1: ", thermal.size())  # (120, 160)
        ######################################################################
        rgb  = self.atten_RCCAModule_1(rgb)
        temp = self.atten_RCCAModule_1(thermal)
        rgb = rgb + temp

        ######################################################################
        rgb = self.encoder_rgb_layer2(rgb)
        if verbose: print("rgb.size() after layer2: ", rgb.size())  # (60, 80)
        thermal = self.encoder_thermal_layer2(thermal)
        if verbose: print("thermal.size() after layer2: ", thermal.size())  # (60, 80)
        ######################################################################
        rgb  = self.atten_RCCAModule_2(rgb)
        temp = self.atten_RCCAModule_2(thermal)
        rgb = rgb + temp
        ######################################################################
        rgb = self.encoder_rgb_layer3(rgb)
        if verbose: print("rgb.size() after layer3: ", rgb.size())  # (30, 40)
        thermal = self.encoder_thermal_layer3(thermal)
        if verbose: print("thermal.size() after layer3: ", thermal.size())  # (30, 40)
        ######################################################################
        rgb = self.atten_RCCAModule_3_1(rgb)
        temp = self.atten_RCCAModule_3_1(thermal)
        rgb = rgb + temp
        ######################################################################
        rgb = self.encoder_rgb_layer4(rgb)
        if verbose: print("rgb.size() after layer4: ", rgb.size())  # (15, 20)
        thermal = self.encoder_thermal_layer4(thermal)
        if verbose: print("thermal.size() after layer4: ", thermal.size())  # (15, 20)
        ######################################################################
        # Des-comentar para usar los bloques RCCAModule en imagen termica y RGB
        rgb = self.atten_RCCAModule_4_1(rgb)
        temp = self.atten_RCCAModule_4_1(thermal)
        fuse = rgb + temp
        ######################################################################
        # fuse = rgb + thermal #comentar para usar los bloques RCCAModule en imagen termica y RGB
        ######################################################################
        # decoder
        fuse = self.deconv(fuse)
        return fuse

def unit_test():
    net = FEANet(9).cuda(3)
    image = torch.randn(1, 4, 480, 640).cuda(3)
    with torch.no_grad():
        output = net.forward(image)
    flops, params = profile(net, inputs=(image, ))
    print(f"FLOPs: {flops}, Params: {params}")


# unit_test()

unit_test()


