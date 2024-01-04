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

class NLBlockNDAnd(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(NLBlockNDAnd, self).__init__()
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc1 = nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.in_planes = in_planes
        self.ratio = ratio
        self.W_z = nn.Sequential(
                     nn.Conv2d(in_planes // ratio, in_planes, kernel_size=1),
                     nn.BatchNorm2d(in_planes)
                 )
        nn.init.constant_(self.W_z[1].weight, 0)
        nn.init.constant_(self.W_z[1].bias, 0)
    
    def forward(self, x):
        batch_size = x.size(0)
        g = self.fc1(x)
        g = self.max_pool(g)
        theta = self.fc1(x)
        phi = self.fc1(x)
        phi = self.max_pool(phi)
        g_x = g.view(batch_size, self.in_planes // self.ratio, -1)
        g_x = g_x.permute(0, 2, 1)
        g_x = self.relu1(g_x)
        theta_x = theta.view(batch_size, self.in_planes // self.ratio, -1)
        phi_x = phi.view(batch_size, self.in_planes // self.ratio, -1)
        phi_x = self.relu1(phi_x)
        theta_x = theta_x.permute(0, 2, 1)
        theta_x = self.relu1(theta_x)
        f = torch.matmul(theta_x, phi_x)
        f_div_C = F.sigmoid(f)
        y = torch.matmul(f_div_C, g_x)
        y = y.permute(0, 2, 1).contiguous()
        y = y.view(batch_size, self.in_planes // self.ratio, *x.size()[2:])
        W_y = self.W_z(y)
        z= W_y + x
        return z

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
      
        self.atten_NLBlock_0 = NLBlockNDAnd(64)
        self.atten_NLBlock_1 = NLBlockNDAnd(256)
        self.atten_NLBlock_2 = NLBlockNDAnd(512)
        self.atten_NLBlock_3_1 = NLBlockNDAnd(1024)
        self.atten_NLBlock_4_1 = NLBlockNDAnd(2048)
 
        
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

        self.deconv1 = self._make_transpose_layer(TransBottleneck, self.inplanes // 2, 2,
                                                  stride=2)  # using // for python 3.6
        self.deconv2 = self._make_transpose_layer(TransBottleneck, self.inplanes // 2, 2,
                                                  stride=2)  # using // for python 3.6
        self.deconv3 = self._make_transpose_layer(TransBottleneck, self.inplanes // 2, 2,
                                                  stride=2)  # using // for python 3.6
        self.deconv4 = self._make_transpose_layer(TransBottleneck, self.inplanes // 2, 2,
                                                  stride=2)  # using // for python 3.6
        self.deconv5 = self._make_transpose_layer(TransBottleneck, n_class, 2, stride=2)

    def _make_transpose_layer(self, block, planes, blocks, stride=1):

        upsample = None
        if stride != 1:
            upsample = nn.Sequential(
                nn.ConvTranspose2d(self.inplanes, planes, kernel_size=2, stride=stride, padding=0, bias=False),
                nn.BatchNorm2d(planes),
            )
        elif self.inplanes != planes:
            upsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes, kernel_size=1, stride=stride, padding=0, bias=False),
                nn.BatchNorm2d(planes),
            )


        for m in upsample.modules():
            if isinstance(m, nn.ConvTranspose2d):
                nn.init.xavier_uniform_(m.weight.data)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

        layers = []

        for i in range(1, blocks):
            layers.append(block(self.inplanes, self.inplanes))

        layers.append(block(self.inplanes, planes, stride, upsample))
        self.inplanes = planes

        return nn.Sequential(*layers)

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
        rgb  = self.atten_NLBlock_0(rgb)
        temp = self.atten_NLBlock_0(thermal)
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
        rgb  = self.atten_NLBlock_1(rgb)
        temp = self.atten_NLBlock_1(thermal)
        rgb = rgb + temp

        ######################################################################
        rgb = self.encoder_rgb_layer2(rgb)
        if verbose: print("rgb.size() after layer2: ", rgb.size())  # (60, 80)
        thermal = self.encoder_thermal_layer2(thermal)
        if verbose: print("thermal.size() after layer2: ", thermal.size())  # (60, 80)
        ######################################################################
        rgb  = self.atten_NLBlock_2(rgb)
        temp = self.atten_NLBlock_2(thermal)
        rgb = rgb + temp
        ######################################################################
        rgb = self.encoder_rgb_layer3(rgb)
        if verbose: print("rgb.size() after layer3: ", rgb.size())  # (30, 40)
        thermal = self.encoder_thermal_layer3(thermal)
        if verbose: print("thermal.size() after layer3: ", thermal.size())  # (30, 40)
        ######################################################################
        rgb = self.atten_NLBlock_3_1(rgb)
        temp = self.atten_NLBlock_3_1(thermal)
        rgb = rgb + temp
        ######################################################################
        rgb = self.encoder_rgb_layer4(rgb)
        if verbose: print("rgb.size() after layer4: ", rgb.size())  # (15, 20)
        thermal = self.encoder_thermal_layer4(thermal)
        if verbose: print("thermal.size() after layer4: ", thermal.size())  # (15, 20)
        ######################################################################
        # Des-comentar para usar los bloques RCCAModule en imagen termica y RGB
        rgb = self.atten_NLBlock_4_1(rgb)
        temp = self.atten_NLBlock_4_1(thermal)
        fuse = rgb + temp
        ######################################################################
        # fuse = rgb + thermal #comentar para usar los bloques RCCAModule en imagen termica y RGB
        ######################################################################
        # decoder
        fuse = self.deconv1(fuse)
        if verbose: print("fuse after deconv1: ", fuse.size())  # (30, 40)
        fuse = self.deconv2(fuse)
        if verbose: print("fuse after deconv2: ", fuse.size())  # (60, 80)
        fuse = self.deconv3(fuse)
        if verbose: print("fuse after deconv3: ", fuse.size())  # (120, 160)
        fuse = self.deconv4(fuse)
        if verbose: print("fuse after deconv4: ", fuse.size())  # (240, 320)
        fuse = self.deconv5(fuse)
        if verbose: print("fuse after deconv5: ", fuse.size())  # (480, 640)
        return fuse

class TransBottleneck(nn.Module):

    def __init__(self, inplanes, planes, stride=1, upsample=None,reduction=4):
        super(TransBottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        if upsample is not None and stride != 1:
            self.conv3 = nn.ConvTranspose2d(planes, planes, kernel_size=2, stride=stride, padding=0, bias=False)
        else:
            self.conv3 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)

        self.bn3 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.upsample = upsample
        self.stride = stride
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight.data)
            elif isinstance(m, nn.ConvTranspose2d):
                nn.init.xavier_uniform_(m.weight.data)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.upsample is not None:
            residual = self.upsample(x)

        out += residual
        out = self.relu(out)

        return out


def unit_test():
    net = FEANet(9).cuda(2)
    image = torch.randn(1, 4, 480, 640).cuda(2)
    with torch.no_grad():
        output = net.forward(image)
    flops, params = profile(net, inputs=(image, ))
    print(f"FLOPs: {flops}, Params: {params}")


# unit_test()

unit_test()


