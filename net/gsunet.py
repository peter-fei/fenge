import cv2
import torch
from torchsummary import summary
import numpy as np
from net.unet3_plus import *
from torch.nn import Module
import torch.nn as nn
from torch.nn import functional as F
from torch.nn.modules.conv import _ConvNd
from torch.nn.modules.utils import _pair


class ASPP(Module):
    def __init__(self,in_channels,out_channels,rates=(6,12,18)):
        super(ASPP,self).__init__()
        self.convs=nn.ModuleList()
        for r in rates:
            self.convs.append(
                nn.Sequential(
                    nn.Conv2d(in_channels,out_channels,3,1,padding=r,dilation=r),
                    nn.BatchNorm2d(out_channels),
                    nn.ReLU(True)
                )
            )
        self.out_conv=nn.Sequential(
            nn.Conv2d(len(rates)*out_channels+in_channels,out_channels,3,1,1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(True)
        )
    def forward(self,x):
        out=x.clone()
        for f in self.convs:
            y=f(x)
            out=torch.cat((out,y),dim=1)
        out=self.out_conv(out)
        return out

class MSBlock(nn.Module):
    def __init__(self, c_in,c_out=21, rate=4):
        super(MSBlock, self).__init__()
        self.rate = rate

        self.conv = nn.Conv2d(c_in, 32, 3, stride=1, padding=1)
        self.relu = nn.ReLU(inplace=True)
        dilation = self.rate*1 if self.rate >= 1 else 1
        self.conv1 = nn.Conv2d(32, 32, 3, stride=1, dilation=dilation, padding=dilation)
        self.relu1 = nn.ReLU(inplace=True)
        dilation = self.rate*2 if self.rate >= 1 else 1
        self.conv2 = nn.Conv2d(32, 32, 3, stride=1, dilation=dilation, padding=dilation)
        self.relu2 = nn.ReLU(inplace=True)
        dilation = self.rate*3 if self.rate >= 1 else 1
        self.conv3 = nn.Conv2d(32, 32, 3, stride=1, dilation=dilation, padding=dilation)
        self.relu3 = nn.ReLU(inplace=True)
        self.out_conv=nn.Sequential(
            nn.Conv2d(32,c_out,3,1,1),
            nn.BatchNorm2d(c_out),
            nn.ReLU(True)
        )
        self._initialize_weights()

    def forward(self, x):
        o = self.relu(self.conv(x))
        o1 = self.relu1(self.conv1(o))
        o2 = self.relu2(self.conv2(o))
        o3 = self.relu3(self.conv3(o))
        out = o + o1 + o2 + o3
        out=self.out_conv(out)
        return out

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                m.weight.data.normal_(0, 0.01)
                if m.bias is not None:
                    m.bias.data.zero_()


class MSBlock_Res(nn.Module):
    def __init__(self, c_in,c_out=21, rate=4):
        super(MSBlock_Res, self).__init__()
        self.rate = rate

        self.conv = nn.Conv2d(c_in, 32, 3, stride=1, padding=1)
        self.relu = nn.ReLU(inplace=True)
        dilation = self.rate * 1 if self.rate >= 1 else 1
        self.conv1 = nn.Conv2d(32, 32, 3, stride=1, dilation=dilation, padding=dilation)
        self.relu1 = nn.ReLU(inplace=True)
        dilation = self.rate * 2 if self.rate >= 1 else 1
        self.conv2 = nn.Conv2d(32, 32, 3, stride=1, dilation=dilation, padding=dilation)
        self.relu2 = nn.ReLU(inplace=True)
        dilation = self.rate * 3 if self.rate >= 1 else 1
        self.conv3 = nn.Conv2d(32, 32, 3, stride=1, dilation=dilation, padding=dilation)
        self.relu3 = nn.ReLU(inplace=True)
        self.out_conv = nn.Sequential(
            nn.Conv2d(32, c_out, 3, 1, 1),
            nn.BatchNorm2d(c_out),
            nn.ReLU(True)
        )
#        self._initialize_weights()
    def forward(self, x):
        o = self.conv(x)
        o1 = self.conv1(o)
        o2 = self.conv2(o)
        o3 = self.conv3(o)
        out =self.relu(o + o1 + o2 + o3)
        out = self.out_conv(out)
        return out


class GatedSpatialConv2d(_ConvNd):
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1,
                 padding=0, dilation=1, groups=1, bias=False):

        kernel_size = _pair(kernel_size)
        stride = _pair(stride)
        padding = _pair(padding)
        dilation = _pair(dilation)
        super(GatedSpatialConv2d, self).__init__(
            in_channels, out_channels, kernel_size, stride, padding, dilation,
            False, _pair(0), groups, bias, 'zeros')

        self._gate_conv = nn.Sequential(
            # nn.BatchNorm2d(in_channels+1),
            nn.Conv2d(in_channels+1, in_channels+1, 1),
            nn.ReLU(),
            nn.Conv2d(in_channels+1, 1, 1),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )

    def forward(self, input_features, gating_features):
        """

        :param input_features:  [NxCxHxW]  featuers comming from the shape branch (canny branch).
        :param gating_features: [Nx1xHxW] features comming from the texture branch (resnet). Only one channel feature map.
        :return:
        """
        alphas = self._gate_conv(torch.cat([input_features, gating_features], dim=1))

        input_features = (input_features * (alphas + 1))
        return F.conv2d(input_features, self.weight, self.bias, self.stride,
                        self.padding, self.dilation, self.groups)


class RefUnet(nn.Module):
    def __init__(self,in_ch,inc_ch):
        super(RefUnet, self).__init__()

        self.conv0 = nn.Conv2d(in_ch,inc_ch,3,padding=1)

        self.conv1 = nn.Conv2d(inc_ch,64,3,padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu1 = nn.ReLU(inplace=True)

        self.pool1 = nn.MaxPool2d(2,2,ceil_mode=True)

        self.conv2 = nn.Conv2d(64,64,3,padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.relu2 = nn.ReLU(inplace=True)

        self.pool2 = nn.MaxPool2d(2,2,ceil_mode=True)

        self.conv3 = nn.Conv2d(64,64,3,padding=1)
        self.bn3 = nn.BatchNorm2d(64)
        self.relu3 = nn.ReLU(inplace=True)

        self.pool3 = nn.MaxPool2d(2,2,ceil_mode=True)

        self.conv4 = nn.Conv2d(64,64,3,padding=1)
        self.bn4 = nn.BatchNorm2d(64)
        self.relu4 = nn.ReLU(inplace=True)

        self.pool4 = nn.MaxPool2d(2,2,ceil_mode=True)

        #####

        self.conv5 = nn.Conv2d(64,64,3,padding=1)
        self.bn5 = nn.BatchNorm2d(64)
        self.relu5 = nn.ReLU(inplace=True)

        #####

        self.conv_d4 = nn.Conv2d(128,64,3,padding=1)
        self.bn_d4 = nn.BatchNorm2d(64)
        self.relu_d4 = nn.ReLU(inplace=True)

        self.conv_d3 = nn.Conv2d(128,64,3,padding=1)
        self.bn_d3 = nn.BatchNorm2d(64)
        self.relu_d3 = nn.ReLU(inplace=True)

        self.conv_d2 = nn.Conv2d(128,64,3,padding=1)
        self.bn_d2 = nn.BatchNorm2d(64)
        self.relu_d2 = nn.ReLU(inplace=True)

        self.conv_d1 = nn.Conv2d(128,64,3,padding=1)
        self.bn_d1 = nn.BatchNorm2d(64)
        self.relu_d1 = nn.ReLU(inplace=True)

        self.conv_d0 = nn.Conv2d(64,1,3,padding=1)

        self.upscore2 = nn.Upsample(scale_factor=2, mode='bilinear')


    def forward(self,x):

        hx = x
        hx = self.conv0(hx)

        hx1 = self.relu1(self.bn1(self.conv1(hx)))
        hx = self.pool1(hx1)

        hx2 = self.relu2(self.bn2(self.conv2(hx)))
        hx = self.pool2(hx2)

        hx3 = self.relu3(self.bn3(self.conv3(hx)))
        hx = self.pool3(hx3)

        hx4 = self.relu4(self.bn4(self.conv4(hx)))
        hx = self.pool4(hx4)

        hx5 = self.relu5(self.bn5(self.conv5(hx)))

        hx = self.upscore2(hx5)

        d4 = self.relu_d4(self.bn_d4(self.conv_d4(torch.cat((hx,hx4),1))))
        hx = self.upscore2(d4)

        d3 = self.relu_d3(self.bn_d3(self.conv_d3(torch.cat((hx,hx3),1))))
        hx = self.upscore2(d3)

        d2 = self.relu_d2(self.bn_d2(self.conv_d2(torch.cat((hx,hx2),1))))
        hx = self.upscore2(d2)

        d1 = self.relu_d1(self.bn_d1(self.conv_d1(torch.cat((hx,hx1),1))))

        residual = self.conv_d0(d1)
        residual=residual+x
        return residual

class Sobel_conv(Module):
    def __init__(self,in_channels,out_channels):
        super(Sobel_conv, self).__init__()
        self.conv1=nn.Sequential(
            nn.Conv2d(in_channels,in_channels,3,1,1),
            nn.BatchNorm2d(in_channels),
            nn.AdaptiveAvgPool2d(1),
        )
        self.out=nn.Sequential(
            nn.ReLU(True),
            nn.Conv2d(in_channels,out_channels,3,1,1),
            nn.BatchNorm2d(out_channels)
        )
        self.sigmoid=nn.Sigmoid()
    def forward(self,x):
        y=self.conv1(x)
        a=self.sigmoid(y)
        # out=x*a+x
        # out=self.out(out)
        return a




class GSUNet0(Module):
    def __init__(self,in_channels,out_channels,filters=(64,128,256,512,1024)):
        super(GSUNet0,self).__init__()
        c_out=21
        self.unet_model=Unet_3plus(in_channels,out_channels,filters)
        up_channel = 5 * filters[0]
        self.gate5=GatedSpatialConv2d(filters[4],filters[3])
        self.up5_4=nn.Upsample(scale_factor=2,mode='bilinear')
        self.msb4=MSBlock(up_channel,c_out)
        self.gate4=GatedSpatialConv2d(filters[3],filters[2])
        self.up4_3=nn.Upsample(scale_factor=2,mode='bilinear')
        self.msb3=MSBlock(up_channel,c_out)
        self.gate3=GatedSpatialConv2d(filters[2],filters[1])
        self.up3_2=nn.Upsample(scale_factor=2,mode='bilinear')
        self.msb2=MSBlock(up_channel,c_out)
        self.gate2=GatedSpatialConv2d(filters[1],filters[0])
        self.up2_1=nn.Upsample(scale_factor=2,mode='bilinear')
        self.msb1=MSBlock(filters[0],c_out)
        self.edge4=nn.Conv2d(c_out,1,1,1,0)
        self.edge3=nn.Conv2d(c_out,1,1,1,0)
        self.edge2=nn.Conv2d(c_out,1,1,1,0)
        self.edge1=nn.Conv2d(c_out,1,1,1,0)
        self.edge_out=nn.Sequential(
            nn.Conv2d(filters[0],out_channels,1,1,0),
            nn.BatchNorm2d(out_channels),
            nn.Conv2d(out_channels,out_channels,3,1,1)
        )
        # self.sobel_conv=Sobel_conv(3,1)
        self.hd1_conv=nn.Conv2d(filters[0],filters[0],3,1,1)
        self.edge_conv=nn.Conv2d(1,filters[0],3,1,1)
        self.aspp=ASPP(filters[0]*2,1)
        self.rfnet=RefUnet(1,64)

        # self.edge_=nn.Sequential(
        #     nn.Conv2d(2,out_channels,1,1,0),
        #     nn.BatchNorm2d(out_channels),
        #     nn.Conv2d(out_channels,out_channels,3,1,1)
        # )
        initialize_weights(self)

    def forward(self,x,train=False):
        outs=self.unet_model(x,res=False,res2=True)
        x5, hd4, hd3, hd2, hd1_=outs
        x5=self.up5_4(x5)
        hd4=self.msb4(hd4)
        hd4_edge=self.edge4(hd4)
        hd3 = self.msb3(hd3)
        hd3_edge = self.edge3(hd3)
        hd2 = self.msb2(hd2)
        hd2_edge = self.edge2(hd2)
        hd1=self.msb1(hd1_)
        hd1_edge=self.edge1(hd1)
        gate5_4=self.gate5(x5,hd4_edge)
        gate5_4=self.up4_3(gate5_4)
        gate4_3=self.gate4(gate5_4,hd3_edge)
        gate4_3=self.up3_2(gate4_3)
        gate3_2=self.gate3(gate4_3,hd2_edge)
        gate3_2=self.up2_1(gate3_2)
        gate2_1=self.gate2(gate3_2,hd1_edge)
        edge_out=self.edge_out(gate2_1)
        x_size=x.size()[2]
        edge_out4=F.interpolate(hd4_edge,x_size,mode='bilinear')
        edge_out3 = F.interpolate(hd3_edge, x_size, mode='bilinear')
        edge_out2 = F.interpolate(hd2_edge, x_size, mode='bilinear')
        edge_out1 = F.interpolate(hd1_edge, x_size, mode='bilinear')


        # im_arr = x.cpu().numpy().transpose((0, 2, 3, 1)).astype(np.uint8)
        # x_size = x.size()
        # sobel = np.zeros((x_size[0], x_size[2], x_size[3], 3))
        # for i in range(x_size[0]):
        #     sobel[i] = cv2.Sobel(im_arr[i], -1, 1, 0, ksize=3) + cv2.Sobel(im_arr[i], -1, 0, 1, ksize=3)
        # sobel = torch.from_numpy(sobel).transpose(1, -1).cuda().float()
        # sobel = self.sobel_conv(sobel)
        # edge_out = torch.cat((edge_out, sobel), dim=1)
        # edge_out=self.edge_(edge_out)

        hd1_asp = self.hd1_conv(hd1_.detach())
        edge_asp=self.edge_conv(edge_out.detach())
        hd1_aspp=self.aspp(torch.cat((hd1_asp,edge_asp),dim=1))
        body=self.rfnet(hd1_aspp)

        out=body+edge_out


        if train:
            return out,body,(edge_out1,edge_out2,edge_out3,edge_out4,edge_out)

        return out



class GSUNet(Module):
    def __init__(self,in_channels,out_channels,filters=(64,128,256,512,1024)):
        super(GSUNet,self).__init__()
        c_out=21
        self.unet_model=Unet_3plus(in_channels,out_channels,filters)
        up_channel = 5 * filters[0]
        self.gate5=GatedSpatialConv2d(filters[4],filters[3])
        self.up5_4=nn.Upsample(scale_factor=2,mode='bilinear')
        self.msb4=MSBlock(up_channel,c_out)
        self.gate4=GatedSpatialConv2d(filters[3],filters[2])
        self.up4_3=nn.Upsample(scale_factor=2,mode='bilinear')
        self.msb3=MSBlock(up_channel,c_out)
        self.gate3=GatedSpatialConv2d(filters[2],filters[1])
        self.up3_2=nn.Upsample(scale_factor=2,mode='bilinear')
        self.msb2=MSBlock(up_channel,c_out)
        self.gate2=GatedSpatialConv2d(filters[1],filters[0])
        self.up2_1=nn.Upsample(scale_factor=2,mode='bilinear')
        self.msb1=MSBlock(filters[0],c_out)
        self.edge4=nn.Conv2d(c_out,1,1,1,0)
        self.edge3=nn.Conv2d(c_out,1,1,1,0)
        self.edge2=nn.Conv2d(c_out,1,1,1,0)
        self.edge1=nn.Conv2d(c_out,1,1,1,0)
        self.edge_out=nn.Sequential(
            nn.Conv2d(filters[0],out_channels,1,1,0),
            nn.BatchNorm2d(out_channels),
            nn.Conv2d(out_channels,out_channels,3,1,1)
        )
        self.sobel_conv=Sobel_conv(1,1)
        self.hd1_conv=nn.Conv2d(filters[0],filters[0],3,1,1)
        self.edge_conv=nn.Sequential(
            nn.Conv2d(1, filters[0], 3, 1, 1),
            nn.BatchNorm2d(filters[0]),
            nn.ReLU(True)
        )

        self.aspp=ASPP(filters[0]*2,1)
        self.rfnet=RefUnet(1,64)

        self.edge_=nn.Sequential(
            nn.Conv2d(1,out_channels,1,1,0),
            nn.BatchNorm2d(out_channels),
            nn.Conv2d(out_channels,out_channels,3,1,1)
        )
        self.final_out=nn.Sequential(
            nn.Conv2d(out_channels,out_channels,1,1,0)
        )
        initialize_weights(self)

    def forward(self,x,train=False):
        outs=self.unet_model(x,res=False,res2=True)
        x5, hd4, hd3, hd2, hd1_=outs
        x5=self.up5_4(x5)
        hd4=self.msb4(hd4)
        hd4_edge=self.edge4(hd4)
        hd3 = self.msb3(hd3)
        hd3_edge = self.edge3(hd3)
        hd2 = self.msb2(hd2)
        hd2_edge = self.edge2(hd2)
        hd1=self.msb1(hd1_)
        hd1_edge=self.edge1(hd1)
        gate5_4=self.gate5(x5,hd4_edge)
        gate5_4=self.up4_3(gate5_4)
        gate4_3=self.gate4(gate5_4,hd3_edge)
        gate4_3=self.up3_2(gate4_3)
        gate3_2=self.gate3(gate4_3,hd2_edge)
        gate3_2=self.up2_1(gate3_2)
        gate2_1=self.gate2(gate3_2,hd1_edge)
        edge_out=self.edge_out(gate2_1)
        x_size=x.size()[2]
        edge_out4=F.interpolate(hd4_edge,x_size,mode='bilinear')
        edge_out3 = F.interpolate(hd3_edge, x_size, mode='bilinear')
        edge_out2 = F.interpolate(hd2_edge, x_size, mode='bilinear')
        edge_out1 = F.interpolate(hd1_edge, x_size, mode='bilinear')


        im_arr = x.cpu().numpy().transpose((0, 2, 3, 1)).astype(np.uint8)
        x_size = x.size()
        sobel = np.zeros((x_size[0], 1, x_size[2], x_size[3]))
        for i in range(x_size[0]):
            sobel[i]=cv2.Canny(im_arr[i],175,250)
            # sobel[i] = cv2.Sobel(im_arr[i], -1, 1, 0, ksize=3) + cv2.Sobel(im_arr[i], -1, 0, 1, ksize=3)
        # sobel = torch.from_numpy(sobel).transpose(1, -1).cuda().float()
        sobel=torch.from_numpy(sobel).cuda().float()
        # print(edge_out.shape,sobel.shape)
        # print(torch.max(sobel),torch.min(sobel),torch.max(edge_out))
        sobel = self.sobel_conv(sobel)
        edge_out=(edge_out+edge_out1+edge_out2+edge_out3+edge_out4)*sobel
        # edge_out = torch.cat((edge_out, sobel), dim=1)
        edge_out=self.edge_(edge_out)

        hd1_asp = self.hd1_conv(hd1_.detach())
        edge_asp=self.edge_conv(edge_out)
        hd1_aspp=self.aspp(torch.cat((hd1_asp,edge_asp),dim=1))
        body=self.rfnet(hd1_aspp)

        out=self.final_out(body+edge_out)


        if train:
            return (edge_out1,edge_out2,edge_out3,edge_out4,edge_out)

        return edge_out



class GSUNet2(Module):
    def __init__(self,in_channels,out_channels,filters=(64,128,256,512,1024)):
        super(GSUNet2,self).__init__()
        c_out=21
        self.unet_model=Unet_3plus(in_channels,out_channels,filters)
        up_channel = 5 * filters[0]
        self.gate5=GatedSpatialConv2d(filters[4],filters[3])
        self.up5_4=nn.Upsample(scale_factor=2,mode='bilinear')
        self.msb4=MSBlock(up_channel,c_out)
        self.gate4=GatedSpatialConv2d(filters[3],filters[2])
        self.up4_3=nn.Upsample(scale_factor=2,mode='bilinear')
        self.msb3=MSBlock(up_channel,c_out)
        self.gate3=GatedSpatialConv2d(filters[2],filters[1])
        self.up3_2=nn.Upsample(scale_factor=2,mode='bilinear')
        self.msb2=MSBlock(up_channel,c_out)
        self.gate2=GatedSpatialConv2d(filters[1],filters[0])
        self.up2_1=nn.Upsample(scale_factor=2,mode='bilinear')
        self.msb1=MSBlock(filters[0],c_out)
        self.edge4=nn.Conv2d(c_out,1,1,1,0)
        self.edge3=nn.Conv2d(c_out,1,1,1,0)
        self.edge2=nn.Conv2d(c_out,1,1,1,0)
        self.edge1=nn.Conv2d(c_out,1,1,1,0)
        self.edge_out=nn.Sequential(
            nn.Conv2d(filters[0],out_channels,1,1,0),
            nn.BatchNorm2d(out_channels),
            nn.Conv2d(out_channels,out_channels,3,1,1)
        )
        self.sobel_conv=Sobel_conv(1,1)
        self.hd1_conv=nn.Conv2d(filters[0],filters[0],3,1,1)
        self.edge_conv=nn.Sequential(
            nn.Conv2d(1, filters[0], 3, 1, 1),
            nn.BatchNorm2d(filters[0]),
            nn.ReLU(True)
        )

        self.aspp=ASPP(filters[0]*2,1)
        self.rfnet=RefUnet(1,64)

        self.edge_=nn.Sequential(
            nn.Conv2d(5,out_channels,1,1,0),
            nn.BatchNorm2d(out_channels),
            nn.Conv2d(out_channels,out_channels,3,1,1)
        )
        self.final_out=nn.Sequential(
            nn.Conv2d(1,out_channels,1,1,0)
        )
        self.edge_out4=nn.Sequential(
            nn.Conv2d(filters[3],out_channels,1,1,0),
            nn.BatchNorm2d(out_channels),
            nn.Conv2d(out_channels,out_channels,3,1,1),
        )
        self.edge_out3 = nn.Sequential(
            nn.Conv2d(filters[2], out_channels, 1, 1, 0),
            nn.BatchNorm2d(out_channels),
            nn.Conv2d(out_channels, out_channels, 3, 1, 1),
        )
        self.edge_out2 = nn.Sequential(
            nn.Conv2d(filters[1], out_channels, 1, 1, 0),
            nn.BatchNorm2d(out_channels),
            nn.Conv2d(out_channels, out_channels, 3, 1, 1),
        )
        self.up5_1=nn.Sequential(
            nn.Upsample(scale_factor=8,mode='bilinear'),
            nn.Conv2d(filters[4],filters[0],3,1,1),
            nn.BatchNorm2d(filters[0]),
            nn.ReLU(True)
        )
        initialize_weights(self)

    def forward(self,x,train=False):
        outs=self.unet_model(x,res=False,res2=True)
        x5, hd4, hd3, hd2, hd1_=outs
        x5=self.up5_4(x5)
        hd4=self.msb4(hd4)
        hd4_edge=self.edge4(hd4)
        hd3 = self.msb3(hd3)
        hd3_edge = self.edge3(hd3)
        hd2 = self.msb2(hd2)
        hd2_edge = self.edge2(hd2)

        gate5_4=self.gate5(x5,hd4_edge)
        gate5_4=self.up4_3(gate5_4)
        gate4_3=self.gate4(gate5_4,hd3_edge)
        gate4_3=self.up3_2(gate4_3)
        gate3_2=self.gate3(gate4_3,hd2_edge)
        gate3_2=self.up2_1(gate3_2)


        x_size=x.size()[2]

        gate_out4=F.interpolate(gate5_4,x_size,mode='bilinear')
        gate_out3=F.interpolate(gate4_3,x_size,mode='bilinear')
        gate_out2=F.interpolate(gate3_2,x_size,mode='bilinear')

        # edge_out4=self.edge_out4(gate_out4)
        # edge_out3 =self.edge_out3(gate_out3)
        # edge_out2=self.edge_out2(gate_out2)

        edge_out4 = F.interpolate(hd4_edge, x_size, mode='bilinear')
        edge_out3 = F.interpolate(hd3_edge, x_size, mode='bilinear')
        edge_out2 = F.interpolate(hd2_edge, x_size, mode='bilinear')
        # edge_out1 = F.interpolate(hd1_edge, x_size, mode='bilinear')
        x5_up=self.up5_1(x5)
        hd1_aspp=self.aspp(torch.cat((hd1_,x5_up),dim=1))
        body=self.rfnet(hd1_aspp)
        final2_1=body+edge_out2
        # final3_1=body+edge_out3.detach()
        # final4_1=body+edge_out4.detach()
        # final=torch.cat((final2_1,final3_1,final4_1),dim=1)
        final_out=self.final_out(final2_1)

        if train:
            return final_out,body,(edge_out4,edge_out3,edge_out2)
        return final_out




class GSUNet3(Module):
    def __init__(self,in_channels,out_channels,filters=(64,128,256,512,1024)):
        super(GSUNet3,self).__init__()
        c_out=21
        self.unet_model=Unet_3plus(in_channels,out_channels,filters)
        up_channel = 5 * filters[0]
        self.gate5=GatedSpatialConv2d(filters[4],filters[3])
        self.up5_4=nn.Upsample(scale_factor=2,mode='bilinear')
        self.msb4=MSBlock(up_channel,c_out)
        self.gate4=GatedSpatialConv2d(filters[3],filters[2])
        self.up4_3=nn.Upsample(scale_factor=2,mode='bilinear')
        self.msb3=MSBlock(up_channel,c_out)
        self.gate3=GatedSpatialConv2d(filters[2],filters[1])
        self.up3_2=nn.Upsample(scale_factor=2,mode='bilinear')
        self.msb2=MSBlock(up_channel,c_out)
        self.gate2=GatedSpatialConv2d(filters[1],filters[0])
        self.up2_1=nn.Upsample(scale_factor=2,mode='bilinear')
        self.msb1=MSBlock(filters[0],c_out)
        self.edge4=nn.Conv2d(c_out,1,1,1,0)
        self.edge3=nn.Conv2d(c_out,1,1,1,0)
        self.edge2=nn.Conv2d(c_out,1,1,1,0)
        self.edge1=nn.Conv2d(c_out,1,1,1,0)
        self.edge_out=nn.Sequential(
            nn.Conv2d(filters[0],out_channels,1,1,0),
            nn.BatchNorm2d(out_channels),
            nn.Conv2d(out_channels,out_channels,3,1,1)
        )
        self.sobel_conv=Sobel_conv(1,1)
        self.hd1_conv=nn.Conv2d(filters[0],filters[0],3,1,1)
        self.edge_conv=nn.Sequential(
            nn.Conv2d(1, filters[0], 3, 1, 1),
            nn.BatchNorm2d(filters[0]),
            nn.ReLU(True)
        )

        self.aspp=ASPP(filters[0]*2,1)
        self.rfnet=RefUnet(1,64)

        self.edge_=nn.Sequential(
            nn.Conv2d(5,out_channels,1,1,0),
            nn.BatchNorm2d(out_channels),
            nn.Conv2d(out_channels,out_channels,3,1,1)
        )
        self.final_out=nn.Sequential(
            nn.Conv2d(1,out_channels,1,1,0)
        )
        self.edge_out4=nn.Sequential(
            nn.Conv2d(filters[3],out_channels,1,1,0),
            nn.BatchNorm2d(out_channels),
            nn.Conv2d(out_channels,out_channels,3,1,1),
        )
        self.edge_out3 = nn.Sequential(
            nn.Conv2d(filters[2], out_channels, 1, 1, 0),
            nn.BatchNorm2d(out_channels),
            nn.Conv2d(out_channels, out_channels, 3, 1, 1),
        )
        self.edge_out2 = nn.Sequential(
            nn.Conv2d(filters[1], out_channels, 1, 1, 0),
            nn.BatchNorm2d(out_channels),
            nn.Conv2d(out_channels, out_channels, 3, 1, 1),
        )
        self.up5_1=nn.Sequential(
            nn.Upsample(scale_factor=8,mode='bilinear'),
            nn.Conv2d(filters[4],filters[0],3,1,1),
            nn.BatchNorm2d(filters[0]),
            nn.ReLU(True)
        )
        initialize_weights(self)

    def forward(self,x,train=False):
        outs=self.unet_model(x,res=False,res2=True)
        x5, hd4, hd3, hd2, hd1_=outs
        x5=self.up5_4(x5)
        hd4=self.msb4(hd4)
        hd4_edge=self.edge4(hd4)
        hd3 = self.msb3(hd3)
        hd3_edge = self.edge3(hd3)
        hd2 = self.msb2(hd2)
        hd2_edge = self.edge2(hd2)

        gate5_4=self.gate5(x5,hd4_edge)
        gate5_4=self.up4_3(gate5_4)
        gate4_3=self.gate4(gate5_4,hd3_edge)
        gate4_3=self.up3_2(gate4_3)
        gate3_2=self.gate3(gate4_3,hd2_edge)
        gate3_2=self.up2_1(gate3_2)


        x_size=x.size()[2]

        gate_out4=F.interpolate(gate5_4,x_size,mode='bilinear')
        gate_out3=F.interpolate(gate4_3,x_size,mode='bilinear')
        gate_out2=F.interpolate(gate3_2,x_size,mode='bilinear')

        edge_out4=self.edge_out4(gate_out4)
        edge_out3 =self.edge_out3(gate_out3)
        edge_out2=self.edge_out2(gate_out2)

        # edge_out4 = F.interpolate(hd4_edge, x_size, mode='bilinear')
        # edge_out3 = F.interpolate(hd3_edge, x_size, mode='bilinear')
        # edge_out2 = F.interpolate(hd2_edge, x_size, mode='bilinear')
        # edge_out1 = F.interpolate(hd1_edge, x_size, mode='bilinear')
        x5_up=self.up5_1(x5)
        hd1_aspp=self.aspp(torch.cat((hd1_,x5_up),dim=1))
        body=self.rfnet(hd1_aspp)
        final2_1=body+edge_out2
        # final3_1=body+edge_out3.detach()
        # final4_1=body+edge_out4.detach()
        # final=torch.cat((final2_1,final3_1,final4_1),dim=1)
        final_out=self.final_out(final2_1)

        if train:
            return edge_out2
        return final_out


class Att(Module):
    def __init__(self,in_channels,planes=64):
        super(Att, self).__init__()
        self.conv1=nn.Sequential(
            nn.Conv2d(1,planes,3,1,1,bias=False),
            nn.BatchNorm2d(planes),
            nn.ReLU(True),
            nn.Conv2d(planes,1,3,1,1,bias=False)
        )
        self.conv2=nn.Sequential(
            nn.Conv2d(in_channels,in_channels,3,1,1,bias=False),
        )
        self.sigmoid=nn.Sigmoid()
        self.relu=nn.ReLU(True)
    def forward(self,x,edge):
        edge_=self.conv1(edge)
        x=self.conv2(x)
        x=self.relu((2-self.sigmoid(edge_))*x)
        return x


class GSUNet4(Module):
    def __init__(self,in_channels,out_channels,filters=(64,128,256,512,1024)):
        super(GSUNet4, self).__init__()
        cat_channels=filters[0]
        up_channels=filters[0]*5
        unet=UNet_3Plus_DeepSup(in_channels,out_channels,filters=filters)
        self.conv1=unet.conv1
        self.conv2=unet.conv2
        self.conv3=unet.conv3
        self.conv4=unet.conv4
        self.conv5=unet.conv5
        self.maxpool2=nn.MaxPool2d(2,2)
        self.maxpool4=nn.MaxPool2d(4,4)
        self.maxpool8=nn.MaxPool2d(8,8)
        self.maxpool16=nn.MaxPool2d(16,16)
        self.bi_up2=nn.Upsample(scale_factor=2,mode='bilinear')
        self.bi_up4=nn.Upsample(scale_factor=4,mode='bilinear')
        self.bi_up8=nn.Upsample(scale_factor=8,mode='bilinear')
        self.bi_up16=nn.Upsample(scale_factor=16,mode='bilinear')
        self.h1_d4=nn.Sequential(
            nn.Conv2d(filters[0],cat_channels,3,1,1),
            nn.BatchNorm2d(cat_channels),
            nn.ReLU(True)
        )
        self.h2_d4=nn.Sequential(
            nn.Conv2d(filters[1],cat_channels,3,1,1),
            nn.BatchNorm2d(cat_channels),
            nn.ReLU(True)
        )
        self.h3_d4=nn.Sequential(
            nn.Conv2d(filters[2],cat_channels,3,1,1),
            nn.BatchNorm2d(cat_channels),
            nn.ReLU(True)
        )
        self.h4_d4=nn.Sequential(
            nn.Conv2d(filters[3],cat_channels,3,1,1),
            nn.BatchNorm2d(cat_channels),
            nn.ReLU(True)
        )
        self.d5_d4=nn.Sequential(
            nn.Conv2d(filters[4],cat_channels,3,1,1),
            nn.BatchNorm2d(cat_channels),
            nn.ReLU(True)
        )
        self.convd4=nn.Sequential(
            nn.Conv2d(up_channels,up_channels,3,1,1),
            nn.BatchNorm2d(up_channels),
            nn.ReLU(True)
        )

        self.h1_d3=nn.Sequential(
            nn.Conv2d(filters[0],cat_channels,3,1,1),
            nn.BatchNorm2d(cat_channels),
            nn.ReLU(True)
        )
        self.h2_d3=nn.Sequential(
            nn.Conv2d(filters[1],cat_channels,3,1,1),
            nn.BatchNorm2d(cat_channels),
            nn.ReLU(True)
        )
        self.h3_d3=nn.Sequential(
            nn.Conv2d(filters[2],cat_channels,3,1,1),
            nn.BatchNorm2d(cat_channels),
            nn.ReLU(True)
        )
        self.d4_d3=nn.Sequential(
            nn.Conv2d(up_channels,cat_channels,3,1,1),
            nn.BatchNorm2d(cat_channels),
            nn.ReLU(True)
        )
        self.d5_d3=nn.Sequential(
            nn.Conv2d(filters[4],cat_channels,3,1,1),
            nn.BatchNorm2d(cat_channels),
            nn.ReLU(True)
        )
        self.convd3=nn.Sequential(
            nn.Conv2d(up_channels,up_channels,3,1,1),
            nn.BatchNorm2d(up_channels),
            nn.ReLU(True)
        )

        self.h1_d2=nn.Sequential(
            nn.Conv2d(filters[0],cat_channels,3,1,1),
            nn.BatchNorm2d(cat_channels),
            nn.ReLU(True)
        )
        self.h2_d2=nn.Sequential(
            nn.Conv2d(filters[1],cat_channels,3,1,1),
            nn.BatchNorm2d(cat_channels),
            nn.ReLU(True)
        )
        self.d3_d2=nn.Sequential(
            nn.Conv2d(up_channels,cat_channels,3,1,1),
            nn.BatchNorm2d(cat_channels),
            nn.ReLU(True)
        )
        self.d4_d2=nn.Sequential(
            nn.Conv2d(up_channels,cat_channels,3,1,1),
            nn.BatchNorm2d(cat_channels),
            nn.ReLU(True)
        )
        self.d5_d2=nn.Sequential(
            nn.Conv2d(filters[4],cat_channels,3,1,1),
            nn.BatchNorm2d(cat_channels),
            nn.ReLU(True)
        )
        self.convd2=nn.Sequential(
            nn.Conv2d(up_channels,up_channels,3,1,1),
            nn.BatchNorm2d(up_channels),
            nn.ReLU(True)
        )

        self.h1_d1=nn.Sequential(
            nn.Conv2d(filters[0],cat_channels,3,1,1),
            nn.BatchNorm2d(cat_channels),
            nn.ReLU(True)
        )
        self.d2_d1=nn.Sequential(
            nn.Conv2d(up_channels,cat_channels,3,1,1),
            nn.BatchNorm2d(cat_channels),
            nn.ReLU(True)
        )
        self.d3_d1=nn.Sequential(
            nn.Conv2d(up_channels,cat_channels,3,1,1),
            nn.BatchNorm2d(cat_channels),
            nn.ReLU(True)
        )
        self.d4_d1=nn.Sequential(
            nn.Conv2d(up_channels,cat_channels,3,1,1),
            nn.BatchNorm2d(cat_channels),
            nn.ReLU(True)
        )
        self.d5_d1=nn.Sequential(
            nn.Conv2d(filters[4],cat_channels,3,1,1),
            nn.BatchNorm2d(cat_channels),
            nn.ReLU(True)
        )
        self.convd1=nn.Sequential(
            nn.Conv2d(up_channels,filters[0],3,1,1),
            nn.BatchNorm2d(filters[0]),
            nn.ReLU(True)

        )
        self.attblock=Att(filters[0],planes=64)
        self.aspp=ASPP(filters[0],1)

        self.relu=nn.ReLU(True)

        self.msb5=MSBlock(filters[4],up_channels)
        self.msb4=MSBlock(up_channels,up_channels)
        self.msb3=MSBlock(up_channels,up_channels)
        self.msb2=MSBlock(up_channels,up_channels)
        self.edge_conv2x1=nn.Conv2d(up_channels,1,1,1,0)
        self.edge_conv3x1=nn.Conv2d(up_channels,1,1,1,0)
        self.edge_conv4x1=nn.Conv2d(up_channels,1,1,1,0)
        self.gate4=GatedSpatialConv2d(up_channels,up_channels)
        self.gate3=GatedSpatialConv2d(up_channels,up_channels)
        self.gate2=GatedSpatialConv2d(up_channels,up_channels)
        self.edge_out2=nn.Sequential(
            nn.Conv2d(up_channels, up_channels, 3, 1, 1),
            nn.BatchNorm2d(up_channels),
            nn.ReLU(True),
            nn.Conv2d(up_channels,1,1,1,0)
        )
        self.edge_out3 = nn.Sequential(
            nn.Conv2d(up_channels, up_channels, 3, 1, 1),
            nn.BatchNorm2d(up_channels),
            nn.ReLU(True),
            nn.Conv2d(up_channels, 1, 1, 1, 0)
        )
        self.edge_out4 = nn.Sequential(
            nn.Conv2d(up_channels, up_channels, 3, 1, 1),
            nn.BatchNorm2d(up_channels),
            nn.ReLU(True),
            nn.Conv2d(up_channels, 1, 1, 1, 0)
        )
        self.edge_conv=nn.Sequential(
            nn.Conv2d(3,1,3,1,1),
            nn.BatchNorm2d(1),
            nn.ReLU(True)
        )
        self.rfnet=RefUnet(1,64)
        self.final_conv=nn.Conv2d(1,1,3,1,1)
        self.body_out=nn.Conv2d(filters[0],1,1,1,0)

        self.hd1_conv=nn.Sequential(
            nn.Conv2d(filters[0],1,1,1,0),
            nn.BatchNorm2d(1),
            nn.ReLU(True),
            nn.Conv2d(1,1,3,1,1)
        )
        self.gate2_=nn.Sequential(
            nn.Conv2d(up_channels,1,3,1,1)
        )
        initialize_weights(self)

    def forward(self,x,train=False):
        x1=self.conv1(x)
        x2=self.conv2(self.maxpool2(x1))
        x3=self.conv3(self.maxpool2(x2))
        x4=self.conv4(self.maxpool2(x3))
        x5=self.conv5(self.maxpool2(x4))

        h1_d4=self.h1_d4(self.maxpool8(x1))
        h2_d4=self.h2_d4(self.maxpool4(x2))
        h3_d4=self.h3_d4(self.maxpool2(x3))
        h4_d4=self.h4_d4(x4)
        d5_d4=self.d5_d4(self.bi_up2(x5))
        hd4=self.convd4(torch.cat((h1_d4,h2_d4,h3_d4,h4_d4,d5_d4),dim=1))

        h1_d3=self.h1_d3(self.maxpool4(x1))
        h2_d3=self.h2_d3(self.maxpool2(x2))
        h3_d3=self.h3_d3(x3)
        d4_d3=self.d4_d3(self.bi_up2(hd4))
        d5_d3=self.d5_d3(self.bi_up4(x5))
        hd3=self.convd3(torch.cat((h1_d3,h2_d3,h3_d3,d4_d3,d5_d3),dim=1))

        h1_d2=self.h1_d2(self.maxpool2(x1))
        h2_d2=self.h2_d2(x2)
        d3_d2=self.d3_d2(self.bi_up2(hd3))
        d4_d2=self.d4_d2(self.bi_up4(hd4))
        d5_d2=self.d5_d2(self.bi_up8(x5))
        hd2=self.convd2(torch.cat((h1_d2,h2_d2,d3_d2,d4_d2,d5_d2),dim=1))

        msb5=self.msb5(self.bi_up2(x5))
        msb4=self.msb4(hd4)
        edge4=self.edge_conv4x1(msb4)
        gate4=self.gate4(msb5,edge4)
        gate4_3=self.bi_up2(gate4)
        msb3=self.msb3(hd3)
        edge3=self.edge_conv3x1(msb3)
        gate3=self.gate3(gate4_3,edge3)
        gate3_2=self.bi_up2(gate3)
        msb2=self.msb2(hd2)
        edge2=self.edge_conv2x1(msb2)
        gate2=self.gate2(gate3_2,edge2)
        # edge2_out=self.edge_out2(self.bi_up2(gate2))
        # edge3_out=self.edge_out3(self.bi_up4(gate3))
        # edge4_out=self.edge_out4(self.bi_up8(gate4))
        edge2_out = self.bi_up2(edge2)
        edge3_out = self.bi_up4(edge3)
        edge4_out = self.bi_up8(edge4)
        edge_=self.edge_conv(torch.cat((self.bi_up2(edge2),self.bi_up4(edge3),self.bi_up8(edge4)),dim=1))
        h1_d1=self.h1_d1(x1)
        d2_d1=self.d2_d1(self.bi_up2(hd2))
        d3_d1=self.d3_d1(self.bi_up4(hd3))
        d4_d1=self.d4_d1(self.bi_up8(hd4))
        d5_d1=self.d5_d1(self.bi_up16(x5))
        # gate2_=self.gate2_(self.bi_up2(gate2.detach()))
        hd1=self.convd1(torch.cat((h1_d1,d2_d1,d3_d1,d4_d1,d5_d1),dim=1))
        hd1=self.attblock(hd1,edge_)
        # hd1=abs(self.relu(self.convd1(torch.cat((h1_d1,d2_d1,d3_d1,d4_d1,d5_d1),dim=1))-edge_))
        # hd1=self.aspp(hd1)
        hd1=self.hd1_conv(hd1)
        body=self.rfnet(hd1)

        final=self.final_conv(body+edge2_out)
        if train:
            return final,body,(edge3_out,edge4_out,edge2_out,edge_)
        return final




        


class GSUNet5(Module):
    def __init__(self,in_channels,out_channels,filters=(64,128,256,512,1024)):
        super(GSUNet5, self).__init__()
        cat_channels=filters[0]
        up_channels=filters[0]*5
        unet=UNet_3Plus_DeepSup(in_channels,out_channels,filters=filters)
        self.conv1=unet.conv1
        self.conv2=unet.conv2
        self.conv3=unet.conv3
        self.conv4=unet.conv4
        self.conv5=unet.conv5
        self.maxpool2=nn.MaxPool2d(2,2)
        self.maxpool4=nn.MaxPool2d(4,4)
        self.maxpool8=nn.MaxPool2d(8,8)
        self.maxpool16=nn.MaxPool2d(16,16)
        self.bi_up2=nn.Upsample(scale_factor=2,mode='bilinear')
        self.bi_up4=nn.Upsample(scale_factor=4,mode='bilinear')
        self.bi_up8=nn.Upsample(scale_factor=8,mode='bilinear')
        self.bi_up16=nn.Upsample(scale_factor=16,mode='bilinear')
        self.h1_d4=nn.Sequential(
            nn.Conv2d(filters[0],cat_channels,3,1,1),
            nn.BatchNorm2d(cat_channels),
            nn.ReLU(True)
        )
        self.h2_d4=nn.Sequential(
            nn.Conv2d(filters[1],cat_channels,3,1,1),
            nn.BatchNorm2d(cat_channels),
            nn.ReLU(True)
        )
        self.h3_d4=nn.Sequential(
            nn.Conv2d(filters[2],cat_channels,3,1,1),
            nn.BatchNorm2d(cat_channels),
            nn.ReLU(True)
        )
        self.h4_d4=nn.Sequential(
            nn.Conv2d(filters[3],cat_channels,3,1,1),
            nn.BatchNorm2d(cat_channels),
            nn.ReLU(True)
        )
        self.d5_d4=nn.Sequential(
            nn.Conv2d(filters[4],cat_channels,3,1,1),
            nn.BatchNorm2d(cat_channels),
            nn.ReLU(True)
        )
        self.convd4=nn.Sequential(
            nn.Conv2d(up_channels,up_channels,3,1,1),
            nn.BatchNorm2d(up_channels),
            nn.ReLU(True)
        )

        self.h1_d3=nn.Sequential(
            nn.Conv2d(filters[0],cat_channels,3,1,1),
            nn.BatchNorm2d(cat_channels),
            nn.ReLU(True)
        )
        self.h2_d3=nn.Sequential(
            nn.Conv2d(filters[1],cat_channels,3,1,1),
            nn.BatchNorm2d(cat_channels),
            nn.ReLU(True)
        )
        self.h3_d3=nn.Sequential(
            nn.Conv2d(filters[2],cat_channels,3,1,1),
            nn.BatchNorm2d(cat_channels),
            nn.ReLU(True)
        )
        self.d4_d3=nn.Sequential(
            nn.Conv2d(up_channels,cat_channels,3,1,1),
            nn.BatchNorm2d(cat_channels),
            nn.ReLU(True)
        )
        self.d5_d3=nn.Sequential(
            nn.Conv2d(filters[4],cat_channels,3,1,1),
            nn.BatchNorm2d(cat_channels),
            nn.ReLU(True)
        )
        self.convd3=nn.Sequential(
            nn.Conv2d(up_channels,up_channels,3,1,1),
            nn.BatchNorm2d(up_channels),
            nn.ReLU(True)
        )

        self.h1_d2=nn.Sequential(
            nn.Conv2d(filters[0],cat_channels,3,1,1),
            nn.BatchNorm2d(cat_channels),
            nn.ReLU(True)
        )
        self.h2_d2=nn.Sequential(
            nn.Conv2d(filters[1],cat_channels,3,1,1),
            nn.BatchNorm2d(cat_channels),
            nn.ReLU(True)
        )
        self.d3_d2=nn.Sequential(
            nn.Conv2d(up_channels,cat_channels,3,1,1),
            nn.BatchNorm2d(cat_channels),
            nn.ReLU(True)
        )
        self.d4_d2=nn.Sequential(
            nn.Conv2d(up_channels,cat_channels,3,1,1),
            nn.BatchNorm2d(cat_channels),
            nn.ReLU(True)
        )
        self.d5_d2=nn.Sequential(
            nn.Conv2d(filters[4],cat_channels,3,1,1),
            nn.BatchNorm2d(cat_channels),
            nn.ReLU(True)
        )
        self.convd2=nn.Sequential(
            nn.Conv2d(up_channels,up_channels,3,1,1),
            nn.BatchNorm2d(up_channels),
            nn.ReLU(True)
        )

        self.h1_d1=nn.Sequential(
            nn.Conv2d(filters[0],cat_channels,3,1,1),
            nn.BatchNorm2d(cat_channels),
            nn.ReLU(True)
        )
        self.d2_d1=nn.Sequential(
            nn.Conv2d(up_channels,cat_channels,3,1,1),
            nn.BatchNorm2d(cat_channels),
            nn.ReLU(True)
        )
        self.d3_d1=nn.Sequential(
            nn.Conv2d(up_channels,cat_channels,3,1,1),
            nn.BatchNorm2d(cat_channels),
            nn.ReLU(True)
        )
        self.d4_d1=nn.Sequential(
            nn.Conv2d(up_channels,cat_channels,3,1,1),
            nn.BatchNorm2d(cat_channels),
            nn.ReLU(True)
        )
        self.d5_d1=nn.Sequential(
            nn.Conv2d(filters[4],cat_channels,3,1,1),
            nn.BatchNorm2d(cat_channels),
            nn.ReLU(True)
        )
        self.convd1=nn.Sequential(
            nn.Conv2d(up_channels,filters[0],3,1,1),
            nn.BatchNorm2d(filters[0]),
            nn.ReLU(True)

        )
        self.attblock=Att(filters[0],planes=64)
        self.aspp=ASPP(filters[0],1)

        self.relu=nn.ReLU(True)

        self.msb5=MSBlock(filters[4],up_channels)
        self.msb4=MSBlock(up_channels,up_channels)
        self.msb3=MSBlock(up_channels,up_channels)
        self.msb2=MSBlock(up_channels,up_channels)
        self.msb1=MSBlock(cat_channels, up_channels)
        self.edge_conv1x1=nn.Conv2d(up_channels, 1, 1, 1, 0)
        self.edge_conv2x1=nn.Conv2d(up_channels,1,1,1,0)
        self.edge_conv3x1=nn.Conv2d(up_channels,1,1,1,0)
        self.edge_conv4x1=nn.Conv2d(up_channels,1,1,1,0)
        self.gate4=GatedSpatialConv2d(cat_channels,up_channels)
        self.gate3=GatedSpatialConv2d(cat_channels,up_channels)
        self.gate2=GatedSpatialConv2d(cat_channels,up_channels)
        self.gate1=GatedSpatialConv2d(cat_channels,up_channels)
        self.edge_out2=nn.Sequential(
            nn.Conv2d(up_channels, up_channels, 3, 1, 1),
            nn.BatchNorm2d(up_channels),
            nn.ReLU(True),
            nn.Conv2d(up_channels,1,1,1,0)
        )
        self.edge_out3 = nn.Sequential(
            nn.Conv2d(up_channels, up_channels, 3, 1, 1),
            nn.BatchNorm2d(up_channels),
            nn.ReLU(True),
            nn.Conv2d(up_channels, 1, 1, 1, 0)
        )
        self.edge_out4 = nn.Sequential(
            nn.Conv2d(up_channels, up_channels, 3, 1, 1),
            nn.BatchNorm2d(up_channels),
            nn.ReLU(True),
            nn.Conv2d(up_channels, 1, 1, 1, 0)
        )
        self.edge_conv=nn.Sequential(
            nn.Conv2d(3,1,3,1,1),
            nn.BatchNorm2d(1),
            nn.ReLU(True)
        )
        self.rfnet=RefUnet(1,64)
        self.final_conv=nn.Conv2d(1,1,3,1,1)
        self.body_out=nn.Conv2d(filters[0],1,1,1,0)

        self.hd1_conv=nn.Sequential(
            nn.Conv2d(cat_channels,1,1,1,0),
            nn.BatchNorm2d(1),
            nn.ReLU(True),
            nn.Conv2d(1,1,3,1,1)
        )
        self.gate_out1=nn.Sequential(
            nn.Conv2d(up_channels,1,1,1,0),
            nn.Conv2d(1,1,3,1,1)
        )
        self.final=nn.Sequential(
            nn.Conv2d(2,2,3,1,1),
            nn.Conv2d(2,1,1,1,0)
        )
        initialize_weights(self)

    def forward(self,x,train=False):
        x1=self.conv1(x)
        x2=self.conv2(self.maxpool2(x1))
        x3=self.conv3(self.maxpool2(x2))
        x4=self.conv4(self.maxpool2(x3))
        x5=self.conv5(self.maxpool2(x4))

        h1_d4=self.h1_d4(self.maxpool8(x1))
        h2_d4=self.h2_d4(self.maxpool4(x2))
        h3_d4=self.h3_d4(self.maxpool2(x3))
        h4_d4=self.h4_d4(x4)
        d5_d4=self.d5_d4(self.bi_up2(x5))
        hd4=self.convd4(torch.cat((h1_d4,h2_d4,h3_d4,h4_d4,d5_d4),dim=1))
        msb4=self.msb4(hd4)
        msb4_edge=self.edge_conv4x1(msb4)
        gate4=self.gate4(d5_d4,msb4_edge)


        h1_d3=self.h1_d3(self.maxpool4(x1))
        h2_d3=self.h2_d3(self.maxpool2(x2))
        h3_d3=self.h3_d3(x3)
        s4_d3=self.d4_d3(self.bi_up2(gate4))
        d5_d3=self.d5_d3(self.bi_up4(x5))
        hd3=self.convd3(torch.cat((h1_d3,h2_d3,h3_d3,s4_d3,d5_d3),dim=1))
        msb3=self.msb3(hd3)
        msb3_edge=self.edge_conv3x1(msb3)
        gate3=self.gate3(s4_d3,msb3_edge)

        h1_d2=self.h1_d2(self.maxpool2(x1))
        h2_d2=self.h2_d2(x2)
        s3_d2=self.d3_d2(self.bi_up2(gate3))
        s4_d2=self.d4_d2(self.bi_up4(gate4))
        d5_d2=self.d5_d2(self.bi_up8(x5))
        hd2=self.convd2(torch.cat((h1_d2,h2_d2,s3_d2,s4_d2,d5_d2),dim=1))
        msb2=self.msb2(hd2)
        msb2_edge=self.edge_conv2x1(msb2)
        gate2=self.gate2(s3_d2,msb2_edge)

        h1_d1=self.h1_d1(x1)
        s2_d1=self.d2_d1(self.bi_up2(gate2))
        s3_d1=self.d3_d1(self.bi_up4(gate3))
        s4_d1=self.d4_d1(self.bi_up8(gate4))
        d5_d1=self.d5_d1(self.bi_up16(x5))
        hd1=self.convd1(torch.cat((h1_d1,s2_d1,s3_d1,s4_d1,d5_d1),dim=1))
        msb1=self.msb1(hd1)
        msb1_edge=self.edge_conv1x1(msb1)
        gate1=self.gate1(s2_d1,msb1_edge)
        edge1=self.gate_out1(gate1)
        out=self.hd1_conv(hd1)
        final=self.final(torch.cat((out,edge1),dim=1))


        if train:
            return final,msb1_edge
        return final



class GSUNet6(Module):
    def __init__(self,in_channels,out_channels,filters=(64,128,256,512,1024)):
        super(GSUNet6, self).__init__()
        cat_channels=filters[0]
        up_channels=filters[0]*5
        unet=UNet_3Plus_DeepSup(in_channels,out_channels,filters=filters)
        self.conv1=unet.conv1
        self.conv2=unet.conv2
        self.conv3=unet.conv3
        self.conv4=unet.conv4
        self.conv5=unet.conv5
        self.maxpool2=nn.MaxPool2d(2,2)
        self.maxpool4=nn.MaxPool2d(4,4)
        self.maxpool8=nn.MaxPool2d(8,8)
        self.maxpool16=nn.MaxPool2d(16,16)
        self.bi_up2=nn.Upsample(scale_factor=2,mode='bilinear')
        self.bi_up4=nn.Upsample(scale_factor=4,mode='bilinear')
        self.bi_up8=nn.Upsample(scale_factor=8,mode='bilinear')
        self.bi_up16=nn.Upsample(scale_factor=16,mode='bilinear')
        self.h1_d4=nn.Sequential(
            nn.Conv2d(filters[0],cat_channels,3,1,1),
            nn.BatchNorm2d(cat_channels),
            nn.ReLU(True)
        )
        self.h2_d4=nn.Sequential(
            nn.Conv2d(filters[1],cat_channels,3,1,1),
            nn.BatchNorm2d(cat_channels),
            nn.ReLU(True)
        )
        self.h3_d4=nn.Sequential(
            nn.Conv2d(filters[2],cat_channels,3,1,1),
            nn.BatchNorm2d(cat_channels),
            nn.ReLU(True)
        )
        self.h4_d4=nn.Sequential(
            nn.Conv2d(filters[3],cat_channels,3,1,1),
            nn.BatchNorm2d(cat_channels),
            nn.ReLU(True)
        )
        self.d5_d4=nn.Sequential(
            nn.Conv2d(filters[4],cat_channels,3,1,1),
            nn.BatchNorm2d(cat_channels),
            nn.ReLU(True)
        )
        self.convd4=nn.Sequential(
            nn.Conv2d(up_channels,up_channels,3,1,1),
            nn.BatchNorm2d(up_channels),
            nn.ReLU(True)
        )

        self.h1_d3=nn.Sequential(
            nn.Conv2d(filters[0],cat_channels,3,1,1),
            nn.BatchNorm2d(cat_channels),
            nn.ReLU(True)
        )
        self.h2_d3=nn.Sequential(
            nn.Conv2d(filters[1],cat_channels,3,1,1),
            nn.BatchNorm2d(cat_channels),
            nn.ReLU(True)
        )
        self.h3_d3=nn.Sequential(
            nn.Conv2d(filters[2],cat_channels,3,1,1),
            nn.BatchNorm2d(cat_channels),
            nn.ReLU(True)
        )
        self.d4_d3=nn.Sequential(
            nn.Conv2d(up_channels,cat_channels,3,1,1),
            nn.BatchNorm2d(cat_channels),
            nn.ReLU(True)
        )
        self.d5_d3=nn.Sequential(
            nn.Conv2d(filters[4],cat_channels,3,1,1),
            nn.BatchNorm2d(cat_channels),
            nn.ReLU(True)
        )
        self.convd3=nn.Sequential(
            nn.Conv2d(up_channels,up_channels,3,1,1),
            nn.BatchNorm2d(up_channels),
            nn.ReLU(True)
        )

        self.h1_d2=nn.Sequential(
            nn.Conv2d(filters[0],cat_channels,3,1,1),
            nn.BatchNorm2d(cat_channels),
            nn.ReLU(True)
        )
        self.h2_d2=nn.Sequential(
            nn.Conv2d(filters[1],cat_channels,3,1,1),
            nn.BatchNorm2d(cat_channels),
            nn.ReLU(True)
        )
        self.d3_d2=nn.Sequential(
            nn.Conv2d(up_channels,cat_channels,3,1,1),
            nn.BatchNorm2d(cat_channels),
            nn.ReLU(True)
        )
        self.d4_d2=nn.Sequential(
            nn.Conv2d(up_channels,cat_channels,3,1,1),
            nn.BatchNorm2d(cat_channels),
            nn.ReLU(True)
        )
        self.d5_d2=nn.Sequential(
            nn.Conv2d(filters[4],cat_channels,3,1,1),
            nn.BatchNorm2d(cat_channels),
            nn.ReLU(True)
        )
        self.convd2=nn.Sequential(
            nn.Conv2d(up_channels,up_channels,3,1,1),
            nn.BatchNorm2d(up_channels),
            nn.ReLU(True)
        )

        self.h1_d1=nn.Sequential(
            nn.Conv2d(filters[0],cat_channels,3,1,1),
            nn.BatchNorm2d(cat_channels),
            nn.ReLU(True)
        )
        self.d2_d1=nn.Sequential(
            nn.Conv2d(up_channels,cat_channels,3,1,1),
            nn.BatchNorm2d(cat_channels),
            nn.ReLU(True)
        )
        self.d3_d1=nn.Sequential(
            nn.Conv2d(up_channels,cat_channels,3,1,1),
            nn.BatchNorm2d(cat_channels),
            nn.ReLU(True)
        )
        self.d4_d1=nn.Sequential(
            nn.Conv2d(up_channels,cat_channels,3,1,1),
            nn.BatchNorm2d(cat_channels),
            nn.ReLU(True)
        )
        self.d5_d1=nn.Sequential(
            nn.Conv2d(filters[4],cat_channels,3,1,1),
            nn.BatchNorm2d(cat_channels),
            nn.ReLU(True)
        )
        self.convd1=nn.Sequential(
            nn.Conv2d(up_channels,filters[0],3,1,1),
            nn.BatchNorm2d(filters[0]),
            nn.ReLU(True)

        )
        self.attblock=Att(filters[0],planes=64)
        self.aspp=ASPP(filters[0],1)

        self.relu=nn.ReLU(True)

        self.msb5=MSBlock(filters[4],up_channels)
        self.msb4=MSBlock(filters[3],up_channels)
        self.msb3=MSBlock(filters[2],up_channels)
        self.msb2=MSBlock(filters[1],up_channels)
        self.msb1=MSBlock(filters[0], up_channels)
        self.edge_conv1x1=nn.Conv2d(up_channels, 1, 1, 1, 0)
        self.edge_conv2x1=nn.Conv2d(up_channels,1,1,1,0)
        self.edge_conv3x1=nn.Conv2d(up_channels,1,1,1,0)
        self.edge_conv4x1=nn.Conv2d(up_channels,1,1,1,0)
        self.gate4=GatedSpatialConv2d(cat_channels,up_channels)
        self.gate3=GatedSpatialConv2d(cat_channels,up_channels)
        self.gate2=GatedSpatialConv2d(cat_channels,up_channels)
        self.gate1=GatedSpatialConv2d(cat_channels,up_channels)

        self.edge_out2=nn.Sequential(
            nn.Conv2d(up_channels, up_channels, 3, 1, 1),
            nn.BatchNorm2d(up_channels),
            nn.ReLU(True),
            nn.Conv2d(up_channels,1,1,1,0)
        )
        self.edge_out3 = nn.Sequential(
            nn.Conv2d(up_channels, up_channels, 3, 1, 1),
            nn.BatchNorm2d(up_channels),
            nn.ReLU(True),
            nn.Conv2d(up_channels, 1, 1, 1, 0)
        )
        self.edge_out4 = nn.Sequential(
            nn.Conv2d(up_channels, up_channels, 3, 1, 1),
            nn.BatchNorm2d(up_channels),
            nn.ReLU(True),
            nn.Conv2d(up_channels, 1, 1, 1, 0)
        )
        self.edge_conv=nn.Sequential(
            nn.Conv2d(3,1,1,1,0),
            nn.BatchNorm2d(1),
            nn.ReLU(True),
            nn.Conv2d(1,1,3,1,1)
        )
        self.rfnet=RefUnet(1,64)
        self.final_conv=nn.Conv2d(1,1,3,1,1)
        self.body_out=nn.Conv2d(filters[0],1,1,1,0)

        self.hd1_conv=nn.Sequential(
            nn.Conv2d(up_channels,1,1,1,0),
            nn.BatchNorm2d(1),
            nn.ReLU(True),
            nn.Conv2d(1,1,3,1,1)
        )
        self.gate_out1=nn.Sequential(
            nn.Conv2d(up_channels,1,1,1,0),
            nn.Conv2d(1,1,3,1,1)
        )
        self.final=nn.Sequential(
            nn.Conv2d(3,3,3,1,1),
            nn.BatchNorm2d(3),
            nn.ReLU(True),
            nn.Conv2d(3,1,1,1,0)
        )
        self.out2=nn.Sequential(
            nn.Conv2d(2,1,1,1,0),
            nn.BatchNorm2d(1),
            nn.ReLU(True)
        )
        self.out3 = nn.Sequential(
            nn.Conv2d(2, 1, 1, 1, 0),
            nn.BatchNorm2d(1),
            nn.ReLU(True)
        )
        self.out4 = nn.Sequential(
            nn.Conv2d(2, 1, 1, 1, 0),
            nn.BatchNorm2d(1),
            nn.ReLU(True)
        )

        initialize_weights(self)

    def forward(self,x,train=False):
        x1=self.conv1(x)
        x2=self.conv2(self.maxpool2(x1))
        x3=self.conv3(self.maxpool2(x2))
        x4=self.conv4(self.maxpool2(x3))
        x5=self.conv5(self.maxpool2(x4))

        # h1_d4=self.h1_d4(self.maxpool8(x1))
        # h2_d4=self.h2_d4(self.maxpool4(x2))
        # h3_d4=self.h3_d4(self.maxpool2(x3))
        # h4_d4=self.h4_d4(x4)
        d5_d4=self.d5_d4(self.bi_up2(x5))
        msb4 = self.msb4(x4)
        msb4_edge = self.edge_conv4x1(msb4)
        # hd4=self.convd4(torch.cat((h1_d4,h2_d4,h3_d4,h4_d4,d5_d4),dim=1))

        gate4=self.gate4(d5_d4,msb4_edge)


        # h1_d3=self.h1_d3(self.maxpool4(x1))
        # h2_d3=self.h2_d3(self.maxpool2(x2))
        # h3_d3=self.h3_d3(x3)
        s4_d3=self.d4_d3(self.bi_up2(gate4))
        # d5_d3=self.d5_d3(self.bi_up4(x5))
        # hd3=self.convd3(torch.cat((h1_d3,h2_d3,h3_d3,s4_d3,d5_d3),dim=1))
        msb3=self.msb3(x3)
        msb3_edge=self.edge_conv3x1(msb3)
        gate3=self.gate3(s4_d3,msb3_edge)

        # h1_d2=self.h1_d2(self.maxpool2(x1))
        # h2_d2=self.h2_d2(x2)
        s3_d2=self.d3_d2(self.bi_up2(gate3))
        # s4_d2=self.d4_d2(self.bi_up4(gate4))
        # d5_d2=self.d5_d2(self.bi_up8(x5))
        # hd2=self.convd2(torch.cat((h1_d2,h2_d2,s3_d2,s4_d2,d5_d2),dim=1))
        msb2=self.msb2(x2)
        msb2_edge=self.edge_conv2x1(msb2)
        gate2=self.gate2(s3_d2,msb2_edge)

        # h1_d1=self.h1_d1(x1)
        s2_d1=self.d2_d1(self.bi_up2(gate2))
        # s3_d1=self.d3_d1(self.bi_up4(gate3))
        # s4_d1=self.d4_d1(self.bi_up8(gate4))
        # d5_d1=self.d5_d1(self.bi_up16(x5))
        # hd1=self.convd1(torch.cat((h1_d1,s2_d1,s3_d1,s4_d1,d5_d1),dim=1))
        edge_=self.edge_conv(torch.cat((self.bi_up2(msb2_edge),self.bi_up4(msb3_edge),self.bi_up8(msb4_edge)),dim=1))
        msb1=self.msb1(x1)
        msb1_edge=self.edge_conv1x1(msb1)
        gate1=self.gate1(s2_d1,msb1_edge)
        # edge1=self.gate_out1(gate1)
        out=self.hd1_conv(gate1)
        out2=self.out2(torch.cat((out,self.bi_up2(msb2_edge)),dim=1))
        out3=self.out3(torch.cat((out,self.bi_up4(msb3_edge)),dim=1))
        out4= self.out3(torch.cat((out, self.bi_up8(msb4_edge)),dim=1))

        final=self.final(torch.cat((out2,out3,out4),dim=1))


        if train:
            return final,(out2,out3,out4),(self.bi_up2(msb2_edge),self.bi_up4(msb3_edge),self.bi_up8(msb4_edge))
        return final


class GSUNet7(Module):
    def __init__(self,in_channels,out_channels,filters=(64,128,256,512,1024)):
        super(GSUNet7,self).__init__()
        c_out=21
        self.unet_model=Unet_3plus(in_channels,out_channels,filters)
        up_channel = 5 * filters[0]
        self.gate5=GatedSpatialConv2d(up_channel,up_channel)
        self.edge5=nn.Sequential(
            nn.Conv2d(filters[4],1,3,1,1),
            nn.BatchNorm2d(1),
            nn.ReLU(True),
            nn.Conv2d(1,1,3,1,1),
            nn.ReLU(True)
        )
        self.msb4=MSBlock_Res(up_channel,c_out)
        self.gate4=GatedSpatialConv2d(up_channel,up_channel)

        self.msb3=MSBlock_Res(up_channel,c_out)
        self.gate3=GatedSpatialConv2d(up_channel,up_channel)

        self.msb2=MSBlock_Res(up_channel,c_out)
        self.gate2=GatedSpatialConv2d(up_channel,up_channel)

        self.msb1=MSBlock_Res(filters[0],c_out)
        self.edge4=nn.Conv2d(c_out,1,1,1,0)
        self.edge3=nn.Conv2d(c_out,1,1,1,0)
        self.edge2=nn.Conv2d(c_out,1,1,1,0)
        self.edge1=nn.Conv2d(c_out,1,1,1,0)


        self.aspp=ASPP(filters[0]*2,1)
        self.rfnet=RefUnet(1,64)

        self.conv_hd2=nn.Sequential(
            nn.Conv2d(filters[4],up_channel,3,1,1),
            nn.BatchNorm2d(up_channel),
            nn.ReLU(True)
        )
        self.hd2_edge=nn.Sequential(
            nn.Conv2d(up_channel,1,3,1,1),
            nn.BatchNorm2d(1),
            nn.ReLU(True),
            nn.Conv2d(1,1,1,1,0)
        )
        self.conv_2_1=nn.Sequential(
            nn.Conv2d(up_channel,filters[0],3,1,1),
            nn.BatchNorm2d(filters[0]),
            nn.ReLU(True)
        )
        self.final=nn.Conv2d(1,1,3,1,1)
        initialize_weights(self)

    def forward(self,x,train=False):
        outs=self.unet_model(x,res=False,res2=True)
        x5, hd4, hd3, hd2, hd1=outs
        x_size=x.size()[2:]
        hd4_up=F.interpolate(hd4,x_size,mode='bilinear')
        hd3_up=F.interpolate(hd3,x_size,mode='bilinear')
        hd2_up=F.interpolate(hd2,x_size,mode='bilinear')
        hd5_up=F.interpolate(x5,x_size,mode='bilinear')

        msb4=self.msb4(hd4_up)
        msb4_edge=self.edge4(msb4)
        msb3 = self.msb3(hd3_up)
        msb3_edge = self.edge3(msb3)
        msb2=self.msb2(hd2_up)
        msb2_edge=self.edge2(msb2)
        msb1 = self.msb1(hd1)
        msb1_edge = self.edge2(msb1)
        hd5_up=self.conv_hd2(hd5_up)
        # hd5_edge=self.edge5(hd5_up)

        hd2_=self.gate5(hd5_up,msb4_edge)
        hd2_=self.gate4(hd2_,msb3_edge)
        hd2_=self.gate3(hd2_,msb2_edge)
        hd2_=self.gate2(hd2_,msb1_edge)
        hd2_edge=self.hd2_edge(hd2_)
        hd2_conv=self.conv_2_1(hd2_)

        hd1_=self.aspp(torch.cat((hd2_conv,hd1),dim=1))
        body=self.rfnet(hd1_)
        final=self.final(hd2_edge+body)
        if train:
            return final,body,(msb4_edge,msb3_edge,msb2_edge,msb1_edge,hd2_edge)

        return final


def initialize_weights(*models):
   for model in models:
        for module in model.modules():
            if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
                nn.init.kaiming_normal(module.weight)
                if module.bias is not None:
                    module.bias.data.zero_()
            elif isinstance(module, nn.BatchNorm2d):
                module.weight.data.fill_(1)
                module.bias.data.zero_()
               
               
class GSUNet8(Module):
    def __init__(self,in_channels,out_channels,filters=(64,128,256,512,1024)):
        super(GSUNet8, self).__init__()
        cat_channels = filters[0]
        up_channels = filters[0] * 5
        unet = UNet_3Plus_DeepSup(in_channels, out_channels, filters=filters)
        self.conv1 = unet.conv1
        self.conv2 = unet.conv2
        self.conv3 = unet.conv3
        self.conv4 = unet.conv4
        self.conv5 = unet.conv5
        self.maxpool2 = nn.MaxPool2d(2, 2)
        self.maxpool4 = nn.MaxPool2d(4, 4)
        self.maxpool8 = nn.MaxPool2d(8, 8)
        self.maxpool16 = nn.MaxPool2d(16, 16)
        self.bi_up2 = nn.Upsample(scale_factor=2, mode='bilinear')
        self.bi_up4 = nn.Upsample(scale_factor=4, mode='bilinear')
        self.bi_up8 = nn.Upsample(scale_factor=8, mode='bilinear')
        self.bi_up16 = nn.Upsample(scale_factor=16, mode='bilinear')

        self.h1_d4 = nn.Sequential(
            nn.Conv2d(filters[0], cat_channels, 3, 1, 1),
            nn.BatchNorm2d(cat_channels),
            nn.ReLU(True)
        )
        self.h2_d4 = nn.Sequential(
            nn.Conv2d(filters[1], cat_channels, 3, 1, 1),
            nn.BatchNorm2d(cat_channels),
            nn.ReLU(True)
        )
        self.h3_d4 = nn.Sequential(
            nn.Conv2d(filters[2], cat_channels, 3, 1, 1),
            nn.BatchNorm2d(cat_channels),
            nn.ReLU(True)
        )
        self.h4_d4 = nn.Sequential(
            nn.Conv2d(filters[3], cat_channels, 3, 1, 1),
            nn.BatchNorm2d(cat_channels),
            nn.ReLU(True)
        )
        self.d5_d4 = nn.Sequential(
            nn.Conv2d(filters[4], cat_channels, 3, 1, 1),
            nn.BatchNorm2d(cat_channels),
            nn.ReLU(True)
        )
        self.convd4 = nn.Sequential(
            nn.Conv2d(cat_channels*2, up_channels, 3, 1, 1),
            nn.BatchNorm2d(up_channels),
            nn.ReLU(True)
        )

        self.h1_d3 = nn.Sequential(
            nn.Conv2d(filters[0], cat_channels, 3, 1, 1),
            nn.BatchNorm2d(cat_channels),
            nn.ReLU(True)
        )
        self.h2_d3 = nn.Sequential(
            nn.Conv2d(filters[1], cat_channels, 3, 1, 1),
            nn.BatchNorm2d(cat_channels),
            nn.ReLU(True)
        )
        self.h3_d3 = nn.Sequential(
            nn.Conv2d(filters[2], cat_channels, 3, 1, 1),
            nn.BatchNorm2d(cat_channels),
            nn.ReLU(True)
        )
        self.d4_d3 = nn.Sequential(
            nn.Conv2d(up_channels, cat_channels, 3, 1, 1),
            nn.BatchNorm2d(cat_channels),
            nn.ReLU(True)
        )
        self.d5_d3 = nn.Sequential(
            nn.Conv2d(filters[4], cat_channels, 3, 1, 1),
            nn.BatchNorm2d(cat_channels),
            nn.ReLU(True)
        )
        self.convd3 = nn.Sequential(
            nn.Conv2d(cat_channels*3, up_channels, 3, 1, 1),
            nn.BatchNorm2d(up_channels),
            nn.ReLU(True)
        )

        self.h1_d2 = nn.Sequential(
            nn.Conv2d(filters[0], cat_channels, 3, 1, 1),
            nn.BatchNorm2d(cat_channels),
            nn.ReLU(True)
        )
        self.h2_d2 = nn.Sequential(
            nn.Conv2d(filters[1], cat_channels, 3, 1, 1),
            nn.BatchNorm2d(cat_channels),
            nn.ReLU(True)
        )
        self.d3_d2 = nn.Sequential(
            nn.Conv2d(up_channels, cat_channels, 3, 1, 1),
            nn.BatchNorm2d(cat_channels),
            nn.ReLU(True)
        )
        self.d4_d2 = nn.Sequential(
            nn.Conv2d(up_channels, cat_channels, 3, 1, 1),
            nn.BatchNorm2d(cat_channels),
            nn.ReLU(True)
        )
        self.d5_d2 = nn.Sequential(
            nn.Conv2d(filters[4], cat_channels, 3, 1, 1),
            nn.BatchNorm2d(cat_channels),
            nn.ReLU(True)
        )
        self.convd2 = nn.Sequential(
            nn.Conv2d(cat_channels*4, up_channels, 3, 1, 1),
            nn.BatchNorm2d(up_channels),
            nn.ReLU(True)
        )

        self.h1_d1 = nn.Sequential(
            nn.Conv2d(filters[0], cat_channels, 3, 1, 1),
            nn.BatchNorm2d(cat_channels),
            nn.ReLU(True)
        )
        self.d2_d1 = nn.Sequential(
            nn.Conv2d(up_channels, cat_channels, 3, 1, 1),
            nn.BatchNorm2d(cat_channels),
            nn.ReLU(True)
        )
        self.d3_d1 = nn.Sequential(
            nn.Conv2d(up_channels, cat_channels, 3, 1, 1),
            nn.BatchNorm2d(cat_channels),
            nn.ReLU(True)
        )
        self.d4_d1 = nn.Sequential(
            nn.Conv2d(up_channels, cat_channels, 3, 1, 1),
            nn.BatchNorm2d(cat_channels),
            nn.ReLU(True)
        )
        self.d5_d1 = nn.Sequential(
            nn.Conv2d(filters[4], cat_channels, 3, 1, 1),
            nn.BatchNorm2d(cat_channels),
            nn.ReLU(True)
        )
        self.convd1 = nn.Sequential(
            nn.Conv2d(up_channels, filters[0], 3, 1, 1),
            # nn.BatchNorm2d(filters[0]),
            # nn.ReLU(True)
        )

        self.aspp = ASPP(filters[0], 1)
        # self.rfnet=RefUnet(1,64)
        self.relu = nn.ReLU(True)

        self.msb5 = MSBlock_Res(up_channels, up_channels)
        self.msb4 = MSBlock_Res(up_channels, up_channels)
        self.msb3 = MSBlock_Res(up_channels, up_channels)
        self.msb2 = MSBlock_Res(up_channels, up_channels)
        self.msb1 = MSBlock_Res(up_channels, up_channels)
        self.edge_conv1x1 = nn.Conv2d(up_channels, 1, 1, 1, 0)
        self.edge_conv2x1 = nn.Conv2d(up_channels, 1, 1, 1, 0)
        self.edge_conv3x1 = nn.Conv2d(up_channels, 1, 1, 1, 0)
        self.edge_conv4x1 = nn.Conv2d(up_channels, 1, 1, 1, 0)
        self.gate4 = Gate(cat_channels, up_channels)
        self.gate3 = Gate(cat_channels, up_channels)
        self.gate2 = Gate(cat_channels, up_channels)
        self.gate1 = Gate(cat_channels, up_channels)
        self.edge2=nn.Conv2d(1,1,3,1,1)
        self.edge3=nn.Conv2d(1,1,3,1,1)
        self.edge4=nn.Conv2d(1,1,3,1,1)
        self.body=nn.Sequential(
            nn.Conv2d(filters[0],filters[0],3,1,1),
            nn.Conv2d(filters[0],1,1,1,0)
        )
        self.final=nn.Sequential(
            nn.Conv2d(2,out_channels,3,1,1)
        )
        initialize_weights(self)

    def forward(self, x, train=False):
        x1 = self.conv1(x)
        x2 = self.conv2(self.maxpool2(x1))
        x3 = self.conv3(self.maxpool2(x2))
        x4 = self.conv4(self.maxpool2(x3))
        x5 = self.conv5(self.maxpool2(x4))

        # h1_d4 = self.h1_d4(self.maxpool8(x1))
        # h2_d4 = self.h2_d4(self.maxpool4(x2))
        # h3_d4 = self.h3_d4(self.maxpool2(x3))
        h4_d4 = self.h4_d4(x4)
        d5_d4 = self.d5_d4(self.bi_up2(x5))

        # hd4=self.convd4(torch.cat((h1_d4,h2_d4,h3_d4,h4_d4,d5_d4),dim=1))
        hd4 = self.convd4(torch.cat((h4_d4, d5_d4), dim=1))
        msb4 = self.msb4(hd4)
        msb4_edge = self.edge_conv4x1(msb4)
        gate4 = self.gate4(d5_d4, msb4_edge)

        # h1_d3 = self.h1_d3(self.maxpool4(x1))
        # h2_d3 = self.h2_d3(self.maxpool2(x2))
        h3_d3 = self.h3_d3(x3)
        s4_d3 = self.d4_d3(self.bi_up2(hd4))
        d5_d3 = self.d5_d3(self.bi_up4(x5))
        # hd3=self.convd3(torch.cat((h1_d3,h2_d3,h3_d3,s4_d3,d5_d3),dim=1))
        hd3 = self.convd3(torch.cat((h3_d3, s4_d3, d5_d3), dim=1))
        msb3 = self.msb3(hd3)
        msb3_edge = self.edge_conv3x1(msb3)
        gate3 = self.gate3(s4_d3, msb3_edge)

        # h1_d2 = self.h1_d2(self.maxpool2(x1))
        h2_d2 = self.h2_d2(x2)
        s3_d2 = self.d3_d2(self.bi_up2(hd3))
        s4_d2 = self.d4_d2(self.bi_up4(hd4))
        d5_d2 = self.d5_d2(self.bi_up8(x5))
        # hd2=self.convd2(torch.cat((h1_d2,h2_d2,s3_d2,s4_d2,d5_d2),dim=1))
        hd2 = self.convd2(torch.cat((h2_d2, s3_d2, s4_d2, d5_d2), dim=1))
        msb2 = self.msb2(hd2)
        msb2_edge = self.edge_conv2x1(msb2)
        gate2 = self.gate2(s3_d2, msb2_edge)

        h1_d1 = self.h1_d1(x1)
        s2_d1 = self.d2_d1(self.bi_up2(gate2))
        s3_d1 = self.d3_d1(self.bi_up4(gate3))
        s4_d1 = self.d4_d1(self.bi_up8(gate4))
        d5_d1 = self.d5_d1(self.bi_up16(x5))
        hd1=self.relu(self.convd1(torch.cat((h1_d1,s2_d1,s3_d1,s4_d1,d5_d1),dim=1)))
        out=self.aspp(hd1)
        # out=self.rfnet(hd1)
        # out=self.body(hd1)
        edge2=self.edge2(self.bi_up2(msb2_edge))
        edge3=self.edge3(self.bi_up4(msb3_edge))
        edge4=self.edge4(self.bi_up8(msb4_edge))

        final = self.final(torch.cat((out,edge2),dim=1))

        if train:
            return final, out,(edge4,edge3,edge2)
        return final



class Gate(Module):
    def __init__(self,in_channels,out_channels):
        super(Gate, self).__init__()
        self.conv1=nn.Sequential(
            nn.Conv2d(1,in_channels,3,1,1),
            nn.BatchNorm2d(in_channels)
        )
        self.conv2=nn.Sequential(
            nn.Conv2d(in_channels,in_channels,3,1,1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(True),
            nn.Conv2d(in_channels,1,1,1,0),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        self.out=nn.Sequential(
            nn.Conv2d(in_channels,out_channels,3,1,1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(True)
        )
        self.relu=nn.ReLU(True)


    def forward(self,x,y):
        y=self.conv1(y)
        z=self.relu(x-y)
        gate=self.conv2(z)
        out=x*(1+gate)
        out=self.out(out)
        return out

class GSUNets(Module):
    def __init__(self,in_channels,out_channels,filters=(64,128,256,512,1024)):
        super(GSUNets,self).__init__()
        c_out=21
        self.unet_model=Unet_3plus(in_channels,out_channels,filters)
        up_channel = 5 * filters[0]
        self.gate5=GatedSpatialConv2d(filters[4],filters[3])
        self.up5_4=nn.Upsample(scale_factor=2,mode='bilinear')
        self.msb4=MSBlock(up_channel,c_out)
        self.gate4=GatedSpatialConv2d(filters[3],filters[2])
        self.up4_3=nn.Upsample(scale_factor=2,mode='bilinear')
        self.msb3=MSBlock(up_channel,c_out)
        self.gate3=GatedSpatialConv2d(filters[2],filters[1])
        self.up3_2=nn.Upsample(scale_factor=2,mode='bilinear')
        self.msb2=MSBlock(up_channel,c_out)
        self.gate2=GatedSpatialConv2d(filters[1],filters[0])
        self.up2_1=nn.Upsample(scale_factor=2,mode='bilinear')
        self.msb1=MSBlock(filters[0],c_out)
        self.edge4=nn.Conv2d(c_out,1,1,1,0)
        self.edge3=nn.Conv2d(c_out,1,1,1,0)
        self.edge2=nn.Conv2d(c_out,1,1,1,0)
        self.edge1=nn.Conv2d(c_out,1,1,1,0)
        self.edge_out=nn.Sequential(
            nn.Conv2d(filters[0],out_channels,1,1,0),
            nn.BatchNorm2d(out_channels),
            nn.Conv2d(out_channels,out_channels,3,1,1)
        )
        # self.sobel_conv=Sobel_conv(3,1)
        self.hd1_conv=nn.Conv2d(filters[0],filters[0],3,1,1)
        self.edge_conv=nn.Conv2d(1,filters[0],3,1,1)
        self.aspp=ASPP(filters[0]*2,1)
        self.rfnet=RefUnet(1,64)

        # self.edge_=nn.Sequential(
        #     nn.Conv2d(2,out_channels,1,1,0),
        #     nn.BatchNorm2d(out_channels),
        #     nn.Conv2d(out_channels,out_channels,3,1,1)
        # )
        initialize_weights(self)

    def forward(self,x,train=False):
        outs=self.unet_model(x,res=False,train=True)
        x5, hd4, hd3, hd2, hd1_=outs
        x5=self.up5_4(x5)
        hd4=self.msb4(hd4)
        hd4_edge=self.edge4(hd4)
        hd3 = self.msb3(hd3)
        hd3_edge = self.edge3(hd3)
        hd2 = self.msb2(hd2)
        hd2_edge = self.edge2(hd2)
        hd1=self.msb1(hd1_)
        hd1_edge=self.edge1(hd1)
        gate5_4=self.gate5(x5,hd4_edge)
        gate5_4=self.up4_3(gate5_4)
        gate4_3=self.gate4(gate5_4,hd3_edge)
        gate4_3=self.up3_2(gate4_3)
        gate3_2=self.gate3(gate4_3,hd2_edge)
        gate3_2=self.up2_1(gate3_2)
        gate2_1=self.gate2(gate3_2,hd1_edge)
        edge_out=self.edge_out(gate2_1)
        x_size=x.size()[2]
        edge_out4=F.interpolate(hd4_edge,x_size,mode='bilinear')
        edge_out3 = F.interpolate(hd3_edge, x_size, mode='bilinear')
        edge_out2 = F.interpolate(hd2_edge, x_size, mode='bilinear')
        edge_out1 = F.interpolate(hd1_edge, x_size, mode='bilinear')


        # im_arr = x.cpu().numpy().transpose((0, 2, 3, 1)).astype(np.uint8)
        # x_size = x.size()
        # sobel = np.zeros((x_size[0], x_size[2], x_size[3], 3))
        # for i in range(x_size[0]):
        #     sobel[i] = cv2.Sobel(im_arr[i], -1, 1, 0, ksize=3) + cv2.Sobel(im_arr[i], -1, 0, 1, ksize=3)
        # sobel = torch.from_numpy(sobel).transpose(1, -1).cuda().float()
        # sobel = self.sobel_conv(sobel)
        # edge_out = torch.cat((edge_out, sobel), dim=1)
        # edge_out=self.edge_(edge_out)

        hd1_asp = self.hd1_conv(hd1_.detach())
        edge_asp=self.edge_conv(edge_out)
        hd1_aspp=self.aspp(torch.cat((hd1_asp,edge_asp),dim=1))
        body=self.rfnet(hd1_aspp)

        out=body+edge_out.detach()


        if train:
            return out,body,(edge_out1,edge_out2,edge_out3,edge_out4,edge_out)

        return out

if __name__=='__main__':
    model=GSUNet6(3,1,(4,8,16,32,64)).cuda()
    summary(model,(3,32,32))




