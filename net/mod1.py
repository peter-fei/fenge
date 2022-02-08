import webbrowser

import torch
from torch import nn
from torch.nn import functional as F
from torch.nn import Module
from torchsummary import summary

from net.unet3_plus import *
import cv2

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

class EDge(Module):
    def __init__(self,in_channels,out_channels,nums):
        super(EDge, self).__init__()
        convs=[]
        for i in range(1,nums+1):
            conv=nn.Sequential(
                nn.MaxPool2d(2**i,2**i),
                nn.Conv2d(in_channels,in_channels,3,1,1),
                nn.BatchNorm2d(in_channels),
                nn.Upsample(scale_factor=2**i,mode='bilinear')
            )
            convs.append(conv)
            self.convs=nn.ModuleList(convs)
            self.out_conv=nn.Sequential(
                nn.Conv2d(in_channels*(nums+1),out_channels,3,1,1)
            )
    def forward(self,x):
        out=[]
        for conv in self.convs:
            out.append(x-conv(x))
        out=torch.cat((out),dim=1)
        out = torch.cat((out,x), dim=1)
        out=self.out_conv(out)
        return out

class Body(Module):
    def __init__(self,in_channels,out_channels):
        super(Body, self).__init__()
        self.conv=nn.Sequential(
            nn.Conv2d(in_channels,in_channels,3,1,1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(True),
            nn.Conv2d(in_channels,1,1,1,0),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        self.relu=nn.ReLU(True)

    def forward(self,x):
        alpha=self.conv(x)
        y=x*(1+alpha)
        return y

class M1(Module):
    def __init__(self,in_channels,out_channels,filters):
        super(M1, self).__init__()
        up_channels=filters[0]*5
        cat_channels=filters[0]
        self.unet=UNet_3Plus_DeepSup(in_channels,out_channels,filters=filters)
        self.conv1_1 = nn.Sequential(
            nn.Conv2d(1, up_channels, 3, 1, 1),
            nn.BatchNorm2d(up_channels),
            nn.ReLU(True)
        )
        self.conv2_1=nn.Sequential(
            nn.Conv2d(1,up_channels,3,1,1),
            nn.BatchNorm2d(up_channels),
            nn.ReLU(True)
        )
        self.conv3_1 = nn.Sequential(
            nn.Conv2d(1, up_channels, 3, 1, 1),
            nn.BatchNorm2d(up_channels),
            nn.ReLU(True)
        )
        self.conv4_1 = nn.Sequential(
            nn.Conv2d(1, up_channels, 3, 1, 1),
            nn.BatchNorm2d(up_channels),
            nn.ReLU(True)
        )
        self.conv5_1 = nn.Sequential(
            nn.Conv2d(1, up_channels, 3, 1, 1),
            nn.BatchNorm2d(up_channels),
            nn.ReLU(True)
        )
        self.conv1_d1=nn.Sequential(
            nn.Conv2d(up_channels,up_channels,3,1,1),
            nn.BatchNorm2d(up_channels),
            # nn.ReLU(True)
        )
        self.conv1_d2=nn.Sequential(
            # nn.Upsample(scale_factor=2,mode='bilinear'),
            nn.Conv2d(up_channels,up_channels,3,1,4,dilation=4),
            nn.BatchNorm2d(up_channels),
            # nn.ReLU(True)
        )
        self.convd2=nn.Sequential(
            nn.Conv2d(up_channels*2,up_channels,3,1,1),
            nn.BatchNorm2d(up_channels),
            # nn.ReLU(True)
        )
        self.conv1_d3=nn.Sequential(
            # nn.Upsample(scale_factor=4,mode='bilinear'),
            nn.Conv2d(up_channels,up_channels,3,1,8,dilation=8),
            nn.BatchNorm2d(up_channels),
            nn.ReLU(True)
        )
        self.convd3=nn.Sequential(
            nn.Conv2d(up_channels*2,up_channels,3,1,1),
            nn.BatchNorm2d(up_channels),
            # nn.ReLU(True)
        )
        self.conv1_d4=nn.Sequential(
            # nn.Upsample(scale_factor=8,mode='bilinear'),
            nn.Conv2d(up_channels,up_channels,3,1,12,dilation=12),
            nn.BatchNorm2d(up_channels),
            nn.ReLU(True)
        )
        self.convd4=nn.Sequential(
            nn.Conv2d(up_channels*2,up_channels,3,1,1),
            nn.BatchNorm2d(up_channels),
            # nn.ReLU(True)
        )
        self.conv1_d5=nn.Sequential(
            # nn.Upsample(scale_factor=16,mode='bilinear'),
            nn.Conv2d(up_channels,up_channels,3,1,16,dilation=16),
            nn.BatchNorm2d(up_channels),
            nn.ReLU(True)
        )
        self.convd5=nn.Sequential(
            nn.Conv2d(up_channels*2,up_channels,3,1,1),
            nn.BatchNorm2d(up_channels),
            # nn.ReLU(True)
        )
        self.relu=nn.ReLU(True)

        self.out = nn.Sequential(
            nn.Conv2d(up_channels, up_channels, 3, 1, 1),
            nn.BatchNorm2d(up_channels),
            nn.ReLU(True),
            nn.Conv2d(up_channels, 1, 1, 1, 0)
        )
        self.edge1=EDge(up_channels,1,4)
        self.edge2=EDge(up_channels,1,3)
        self.edge3=EDge(up_channels,1,2)
        self.edge4=EDge(up_channels,1,1)
        self.gate4=Gate(up_channels,up_channels)
        self.gate3=Gate(up_channels,up_channels)
        self.gate2=Gate(up_channels,up_channels)
        self.edge_out4=nn.Sequential(
            nn.Conv2d(up_channels,up_channels,3,1,1),
            nn.BatchNorm2d(up_channels),
            nn.ReLU(True),
            nn.Conv2d(up_channels,out_channels,1,1,0)
        )
        self.edge_out3 = nn.Sequential(
            nn.Conv2d(up_channels, up_channels, 3, 1, 1),
            nn.BatchNorm2d(up_channels),
            nn.ReLU(True),
            nn.Conv2d(up_channels, out_channels, 1, 1, 0)
        )
        self.edge_out2 = nn.Sequential(
            nn.Conv2d(up_channels, up_channels, 3, 1, 1),
            nn.BatchNorm2d(up_channels),
            nn.ReLU(True),
            nn.Conv2d(up_channels, out_channels, 1, 1, 0)
        )
        self.final=nn.Sequential(
            nn.Conv2d(2,2,3,1,1),
            nn.BatchNorm2d(2),
            nn.ReLU(True),
            nn.Conv2d(2,1,1,1,0)
        )
        self.up_2=nn.Upsample(scale_factor=2,mode='bilinear')
        self.up_4 = nn.Upsample(scale_factor=4, mode='bilinear')
        self.up_8 = nn.Upsample(scale_factor=8, mode='bilinear')
        self.up_16 = nn.Upsample(scale_factor=16, mode='bilinear')

    def forward(self,x,train=False):
        outs=self.unet(x,train=True)
        hd5,hd4,hd3,hd2,hd1=outs
        hd1=self.conv1_1(hd1)
        hd2=self.conv2_1(hd2)
        hd3=self.conv3_1(hd3)
        hd4=self.conv4_1(hd4)
        hd5=self.conv5_1(hd5)
        h1_1=self.conv1_d1(hd1)
        hd1_2=self.conv1_d2(hd1)
        hd1_3=self.conv1_d3(hd1)
        hd1_4=self.conv1_d4(hd1)
        hd1_5=self.conv1_d5(hd1)
        # print(hd2.size())

        # print(hd1.size(),hd2.size(),hd3.size(),hd4.size(),hd5.size())
        # print(hd1_2.size(),hd1_3.size(),hd1_4.size(),hd1_5.size())
        h1_2=torch.cat((hd1_2,self.up_2(hd2)),dim=1)
        h1_2=self.convd2(h1_2)
        h1_3=torch.cat((hd1_3,self.up_4(hd3)),dim=1)
        h1_3=self.convd3(h1_3)
        h1_4=torch.cat((hd1_4,self.up_8(hd4)),dim=1)
        h1_4=self.convd4(h1_4)
        h1_5=torch.cat((hd1_5,self.up_16(hd5)),dim=1)
        h1_5=self.convd5(h1_5)
        h1_d1=self.relu(h1_1+h1_2+h1_3+h1_4+h1_5)

        edge4=self.edge4(hd4)
        g5_4=self.up_2(hd5)
        # print(edge4.size(),g5_4.size())
        gate4=self.gate4(g5_4,edge4)
        edge3=self.edge3(hd3)
        g4_3=self.up_2(gate4)
        gate3=self.gate3(g4_3,edge3)
        # print(hd2.size(),edge3.size(),gate3.size())
        edge2=self.edge2(hd2)
        g3_2=self.up_2(gate3)
        gate2=self.gate2(g3_2,edge2)
        # g2_1=self.up_2(gate2)
        x_size=x.size()[2:]
        # print(x_size)
        edge_out4=self.edge_out4(F.interpolate(gate4,x_size,mode='bilinear'))
        edge_out3=self.edge_out3(F.interpolate(gate3,x_size,mode='bilinear'))
        edge_out2=self.edge_out2(F.interpolate(gate2,x_size,mode='bilinear'))

        body_out=self.out(h1_d1)
        # print(body_out.size(),edge_out2.size())
        h1_out = torch.cat((body_out, edge_out2), dim=1)
        final=self.final(h1_out)
        # out=self.edge1(h1_d1)

        if train:
            return final,body_out,(edge_out4,edge_out3,edge_out2)
        # print(out.size())
        return final






from net.init_weights import init_weights
# from torch.utils.tensorboard import SummaryWriter


class DoubleConv(nn.Module):
    def __init__(self,in_channels,out_channels):
        super(DoubleConv,self).__init__()
        self.conv1=nn.Sequential(nn.Conv2d(in_channels,out_channels,kernel_size=3,stride=1,padding=1),
                                 nn.BatchNorm2d(out_channels),
                                 nn.ReLU())
        self.conv2=nn.Sequential(nn.Conv2d(out_channels,out_channels,kernel_size=3,stride=1,padding=1),
                                 nn.BatchNorm2d(out_channels),
                                 nn.ReLU(),
                                 )

    def forward(self,x):
        # print(x.shape)
        x=self.conv1(x)
        # print(1)
        y=self.conv2(x)
        return y

class M2(Module):
    def __init__(self,in_channel,out_channel,filters=(64,128,248,512,1024)):
        super(M2, self).__init__()
        cat_channel=filters[0]
        cat_blocks=5
        up_channel=cat_blocks*cat_channel

        self.conv1=DoubleConv(in_channel,filters[0])
        self.conv2=DoubleConv(filters[0],filters[1])
        self.conv3=DoubleConv(filters[1],filters[2])
        self.conv4=DoubleConv(filters[2],filters[3])
        self.conv5=DoubleConv(filters[3],filters[4])
        self.maxpool=nn.MaxPool2d(2,2)

        self.dconv4=nn.Sequential(
            nn.Conv2d(filters[3],filters[3],3,1,1),
            nn.BatchNorm2d(filters[3]),
            nn.ReLU(True),
            nn.Conv2d(filters[3],up_channel,1,1,0),
            nn.BatchNorm2d(up_channel),
            nn.ReLU(True)
        )
        self.edge4=EDge(up_channel,1,2)
        self.body4=Body(up_channel,up_channel)
        self.gate4=Gate(up_channel,up_channel)

        self.dconv3 = nn.Sequential(
            nn.Conv2d(filters[2], filters[2], 3, 1, 1),
            nn.BatchNorm2d(filters[2]),
            nn.ReLU(True),
            nn.Conv2d(filters[2], up_channel, 1, 1, 0),
            nn.BatchNorm2d(up_channel),
            nn.ReLU(True)
        )
        self.edge3=EDge(up_channel,1,3)
        self.body3=Body(up_channel,up_channel)
        self.gate3=Gate(up_channel,up_channel)

        self.dconv2 = nn.Sequential(
            nn.Conv2d(filters[1], filters[1], 3, 1, 1),
            nn.BatchNorm2d(filters[1]),
            nn.ReLU(True),
            nn.Conv2d(filters[1], up_channel, 1, 1, 0),
            nn.BatchNorm2d(up_channel),
            nn.ReLU(True)
        )
        self.edge2=EDge(up_channel,1,4)
        self.body2=Body(up_channel,up_channel)
        self.gate2=Gate(up_channel,up_channel)

        self.dconv1 = nn.Sequential(
            nn.Conv2d(filters[0], filters[0], 3, 1, 1),
            nn.BatchNorm2d(filters[0]),
            nn.ReLU(True),
            nn.Conv2d(filters[0], up_channel, 1, 1, 0),
            nn.BatchNorm2d(up_channel),
            nn.ReLU(True)
        )
        self.edge1 = EDge(up_channel, 1,5)
        self.body1 = Body(up_channel, up_channel)
        self.gate1 = Gate(up_channel, up_channel)

        self.h5_d4_conv=nn.Sequential(
            nn.Upsample(scale_factor=2,mode='bilinear'),
            nn.Conv2d(filters[4],up_channel,1,1,0),
            nn.BatchNorm2d(up_channel),
            nn.ReLU()
        )
        self.hconv4=nn.Sequential(
            nn.Conv2d(up_channel*2,up_channel,3,1,1),
            nn.BatchNorm2d(up_channel),
            nn.ReLU(True)
        )
        self.h4_d3_conv = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear'),
            nn.Conv2d(up_channel, up_channel, 3,1,1),
            nn.BatchNorm2d(up_channel),
            nn.ReLU()
        )
        self.hconv3=nn.Sequential(
            nn.Conv2d(up_channel*2,up_channel,3,1,1),
            nn.BatchNorm2d(up_channel),
            nn.ReLU(True)
        )
        self.h3_d2_conv = nn.Sequential(
            nn.Upsample(scale_factor=2,mode='bilinear'),
            nn.Conv2d(up_channel, up_channel, 3,1,1),
            nn.BatchNorm2d(up_channel),
            nn.ReLU()
        )
        self.hconv2=nn.Sequential(
            nn.Conv2d(up_channel*2,up_channel,3,1,1),
            nn.BatchNorm2d(up_channel),
            nn.ReLU(True)
        )
        self.h2_d1_conv = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear'),
            nn.Conv2d(up_channel, up_channel, 3,1,1),
            nn.BatchNorm2d(up_channel),
            nn.ReLU()
        )
        self.hconv1=nn.Sequential(
            nn.Conv2d(up_channel*2,up_channel,3,1,1),
            nn.BatchNorm2d(up_channel),
            nn.ReLU(True)
        )
        self.out=nn.Sequential(
            nn.Conv2d(up_channel,up_channel,3,1,1),
            nn.BatchNorm2d(up_channel),
            nn.ReLU(True),
            nn.Conv2d(up_channel,out_channel,1,1,0)
        )
        self.up_2=nn.Upsample(scale_factor=2,mode='bilinear')
        self.up_4=nn.Upsample(scale_factor=4,mode='bilinear')
        self.up_8=nn.Upsample(scale_factor=8,mode='bilinear')
        self.bodyout4=nn.Conv2d(up_channel,out_channel,3,1,1)
        self.edgeout4=nn.Conv2d(up_channel,out_channel,3,1,1)
        self.bodyout3=nn.Conv2d(up_channel,out_channel,3,1,1)
        self.edgeout3=nn.Conv2d(up_channel,out_channel,3,1,1)
        self.bodyout2=nn.Conv2d(up_channel,out_channel,3,1,1)
        self.edgeout2=nn.Conv2d(up_channel,out_channel,3,1,1)
        self.bodyout1=nn.Conv2d(up_channel,out_channel,3,1,1)
        self.edgeout1=nn.Conv2d(up_channel,out_channel,3,1,1)
        # init_weights(self)

    def forward(self,x,train=False):
        h1=self.conv1(x)
        h2=self.maxpool(h1)
        h2=self.conv2(h2)
        h3=self.maxpool(h2)
        h3=self.conv3(h3)
        h4=self.maxpool(h3)
        h4=self.conv4(h4)
        h5=self.maxpool(h4)
        hd5=self.conv5(h5)

        h5_4=self.h5_d4_conv(hd5)
        d4=self.dconv4(h4)
        edge4=self.edge4(d4)
        body4=self.body4(d4)
        gate4=self.gate4(body4,edge4)
        hd4=self.hconv4(torch.cat((body4,h5_4),dim=1))
        hd4=hd4+gate4

        h4_3=self.h4_d3_conv(hd4)
        d3 = self.dconv3(h3)
        edge3 = self.edge3(d3)
        body3 = self.body3(d3)
        # print(d3.size(),edge3.size(),body3.size(),h4_3.size())
        gate3 = self.gate3(body3, edge3)
        hd3 = self.hconv3(torch.cat((body3, h4_3), dim=1))
        hd3 = hd3 + gate3

        h3_2 = self.h3_d2_conv(hd3)
        d2 = self.dconv2(h2)
        edge2 = self.edge2(d2)
        body2 = self.body2(d2)
        gate2 = self.gate2(body2, edge2)
        hd2 = self.hconv2(torch.cat((body2, h3_2), dim=1))
        hd2 = hd2 + gate2

        h2_1 = self.h2_d1_conv(hd2)
        d1 = self.dconv1(h1)
        edge1 = self.edge1(d1)
        body1 = self.body1(d1)
        gate1 = self.gate1(body1, edge1)
        hd1 = self.hconv4(torch.cat((body1, h2_1), dim=1))
        hd1 = hd1 + gate1

        bodyout4=self.up_8(self.bodyout4(body4))
        edgeout4=self.up_8(self.edgeout4(gate4))

        bodyout3=self.up_4(self.bodyout3(body3))
        edgeout3=self.up_4(self.edgeout3(gate3))

        bodyout2=self.up_2(self.bodyout2(gate2))
        edgeout2=self.up_2(self.edgeout2(gate2))

        bodyout1=self.bodyout1(body1)
        edgeout1=self.edgeout1(gate1)
        out=self.out(hd1)

        # print(out.size(),bodyout4.size(),bodyout3.size(),bodyout2.size(),bodyout1.size())
        # print(edgeout4.size(),edgeout3.size(),edgeout2.size(),edgeout1.size())
        if train:
            return out,(bodyout4,bodyout3,bodyout2,bodyout1),(edgeout4,edgeout3,edgeout2,edgeout1)
        return out


class DownConv(Module):
    def __init__(self,in_channels,out_channels,stride):
        super(DownConv, self).__init__()
        self.conv1=nn.Sequential(
            nn.MaxPool2d(stride,stride),
            nn.Conv2d(in_channels,out_channels,3,1,1),
            nn.BatchNorm2d(in_channels)
        )
        self.conv2=nn.Sequential(
            nn.MaxPool2d(stride,stride),
            nn.Conv2d(in_channels,out_channels,1,1,0),
            nn.BatchNorm2d(out_channels)
        )
        self.relu=nn.ReLU(True)
    def forward(self,x):
        out=self.conv1(x)
        x=self.conv2(x)
        out=self.relu(x+out)
        return out


class UPConv(Module):
    def __init__(self,in_channels,out_channels,factor):
        super(UPConv, self).__init__()
        self.conv1=nn.Sequential(
            nn.Upsample(scale_factor=factor),
            nn.Conv2d(in_channels,out_channels,3,1,1),
            nn.BatchNorm2d(in_channels)
        )
        self.relu=nn.ReLU(True)
    def forward(self,x):
        out=self.conv1(x)
        return out


class M3(Module):
    def __init__(self,in_channels,out_cannels,filters):
        super(M3, self).__init__()
        self.unet=UNet_3Plus_DeepSup(in_channels,out_cannels,filters=filters)
        up_channels=filters[0]*5
        self.convd5=nn.Sequential(
            nn.Conv2d(filters[4],up_channels,3,1,1),
            nn.BatchNorm2d(up_channels),
            nn.ReLU(True)
        )
        self.h1_5=DownConv(up_channels,up_channels,16)
        self.h2_5=DownConv(up_channels,up_channels,8)
        self.h3_5=DownConv(up_channels,up_channels,4)
        self.h4_5=DownConv(up_channels,up_channels,2)
        self.conv5=nn.Sequential(
            nn.Conv2d(up_channels*5,up_channels,3,1,1),
            nn.BatchNorm2d(up_channels),
            nn.ReLU(True)
        )
        self.upconv5_4=nn.Sequential(
            nn.Conv2d(up_channels,up_channels,3,1,1),
            nn.BatchNorm2d(up_channels),
            nn.ReLU(True)
        )

        self.h1_4=DownConv(up_channels,up_channels,8)
        self.h2_4=DownConv(up_channels,up_channels,4)
        self.h3_4=DownConv(up_channels,up_channels,2)
        self.conv4 = nn.Sequential(
            nn.Conv2d(up_channels * 4, up_channels, 3, 1, 1),
            nn.BatchNorm2d(up_channels),
            nn.ReLU(True)
        )
        self.upconv4_3=nn.Sequential(
            nn.Conv2d(up_channels,up_channels,3,1,1),
            nn.BatchNorm2d(up_channels),
            nn.ReLU(True)
        )
        self.h1_3=DownConv(up_channels,up_channels,4)
        self.h2_3=DownConv(up_channels,up_channels,2)
        self.conv3 = nn.Sequential(
            nn.Conv2d(up_channels * 3, up_channels, 3, 1, 1),
            nn.BatchNorm2d(up_channels),
            nn.ReLU(True)
        )
        self.upconv3_2=nn.Sequential(
            nn.Conv2d(up_channels,up_channels,3,1,1),
            nn.BatchNorm2d(up_channels),
            nn.ReLU(True)
        )
        self.h1_2=DownConv(up_channels,up_channels,2)
        self.conv2 = nn.Sequential(
            nn.Conv2d(up_channels * 2, up_channels, 3, 1, 1),
            nn.BatchNorm2d(up_channels),
            nn.ReLU(True)
        )
        self.upconv2_1=nn.Sequential(
            nn.Conv2d(up_channels,up_channels,3,1,1),
            nn.BatchNorm2d(up_channels),
            nn.ReLU(True)
        )

        self.convd1_2=nn.Sequential(
            nn.Conv2d(up_channels,up_channels,3,1,padding=4,dilation=4),
            nn.BatchNorm2d(up_channels)
        )
        self.convd1_3 = nn.Sequential(
            nn.Conv2d(up_channels, up_channels, 3, 1, padding=8, dilation=8),
            nn.BatchNorm2d(up_channels)
        )
        self.convd1_4 = nn.Sequential(
            nn.Conv2d(up_channels, up_channels, 3, 1, padding=12, dilation=12),
            nn.BatchNorm2d(up_channels)
        )
        self.convd1_5 = nn.Sequential(
            nn.Conv2d(up_channels, up_channels, 3, 1, padding=16, dilation=16),
            nn.BatchNorm2d(up_channels)
        )
        self.up5_1=UPConv(up_channels,up_channels,16)
        self.up4_1 = UPConv(up_channels, up_channels, 8)
        self.up3_1 = UPConv(up_channels, up_channels, 4)
        self.up2_1 = UPConv(up_channels, up_channels, 2)
        self.conv1=nn.Sequential(
            nn.Conv2d(up_channels*5,up_channels,3,1,1),
            nn.BatchNorm2d(up_channels),
            nn.ReLU(True)
        )

        self.edge4=EDge(up_channels,1,1)
        self.edge3=EDge(up_channels,1,2)
        self.edge2=EDge(up_channels,1,3)
        self.edge1=EDge(up_channels,1,4)

        self.gate4 =  Gate(up_channels,up_channels)
        self.gate3 = Gate(up_channels, up_channels)
        self.gate2 = Gate(up_channels, up_channels)
        self.gate1 = Gate(up_channels, up_channels)
        self.out=nn.Sequential(
            nn.Conv2d(up_channels*2,up_channels,3,1,1),
            nn.BatchNorm2d(up_channels),
            nn.ReLU(True),
            nn.Conv2d(up_channels, up_channels, 3, 1, 1),
            nn.BatchNorm2d(up_channels),
            nn.ReLU(True),
            nn.Conv2d(up_channels,out_cannels,1,1,0)
        )
        self.edge_out1=nn.Sequential(
            nn.Conv2d(up_channels,up_channels,3,1,1),
            nn.BatchNorm2d(up_channels),
            nn.ReLU(True),
            nn.Conv2d(up_channels,out_cannels,1,1,0)
        )
        self.edge_out2 = nn.Sequential(
            nn.Conv2d(up_channels, up_channels, 3, 1, 1),
            nn.BatchNorm2d(up_channels),
            nn.ReLU(True),
            nn.Conv2d(up_channels, out_cannels, 1, 1, 0)
        )
        self.edge_out3 = nn.Sequential(
            nn.Conv2d(up_channels, up_channels, 3, 1, 1),
            nn.BatchNorm2d(up_channels),
            nn.ReLU(True),
            nn.Conv2d(up_channels, out_cannels, 1, 1, 0)
        )
        self.edge_out4 = nn.Sequential(
            nn.Conv2d(up_channels, up_channels, 3, 1, 1),
            nn.BatchNorm2d(up_channels),
            nn.ReLU(True),
            nn.Conv2d(up_channels, out_cannels, 1, 1, 0)
        )
        self.up2=nn.Upsample(scale_factor=2,mode='bilinear')
        self.up4=nn.Upsample(scale_factor=4, mode='bilinear')
        self.up8=nn.Upsample(scale_factor=8, mode='bilinear')
        self.relu=nn.ReLU(True)
        self.body_out=nn.Sequential(
            nn.Conv2d(up_channels,out_cannels,3,1,1)
        )

    def forward(self,x,train=False):
        h5,h4,h3,h2,h1=self.unet(x,train=True)
        h5=self.convd5(h5)
        # print(h5.size(),h4.size(),h3.size(),h2.size(),h1.size())
        # h1_5=self.h1_5(h1)
        # h2_5=self.h2_5(h2)
        # h3_5=self.h3_5(h3)
        # h4_5=self.h4_5(h4)
        # hd5=torch.cat((h5,h4_5,h3_5,h2_5,h1_5),dim=1)
        # hd5_4=self.upconv5_4(self.up2(self.conv5(hd5)))
        #
        # h1_4=self.h1_4(h1)
        # h2_4=self.h2_4(h2)
        # h3_4=self.h3_4(h3)
        # hd4=torch.cat((h4,h3_4,h2_4,h1_4),dim=1)
        # hd4=self.conv4(hd4)
        # edge4=self.edge4(hd4)
        # gate4=self.gate4(hd5_4,edge4)
        # hd4_3=self.upconv4_3(self.up2(gate4))
        #
        # h1_3=self.h1_3(h1)
        # h2_3=self.h2_3(h2)
        # hd3=torch.cat((h1_3,h2_3,h3),dim=1)
        # hd3=self.conv3(hd3)
        # edge3=self.edge3(hd3)
        # gate3=self.gate3(hd4_3,edge3)
        # hd3_2=self.upconv4_3(self.up2(gate3))

        # h1_2=self.h1_2(h1)
        # hd2=torch.cat((h1_2,h2),dim=1)
        # hd2=self.conv2(hd2)
        # edge2=self.edge2(hd2)
        # gate2=self.gate2(hd3_2,edge2)
        # hd2_1=self.upconv2_1(self.up2(gate2))

        hd1_2=self.convd1_2(h1)
        hd1_3=self.convd1_3(h1)
        hd1_4=self.convd1_4(h1)
        hd1_5=self.convd1_5(h1)
        up5_1=self.up5_1(h5)
        up4_1=self.up4_1(h4)
        up3_1=self.up3_1(h3)
        up2_1=self.up2_1(h2)
        hd1_2=self.relu(hd1_2+up2_1)
        hd1_3=self.relu(hd1_3+up3_1)
        hd1_4=self.relu(hd1_4+up4_1)
        hd1_5=self.relu(hd1_5+up5_1)
        hd1=torch.cat((h1,hd1_2,hd1_3,hd1_4,hd1_5),dim=1)
        hd1=self.conv1(hd1)
        # edge1=self.edge1(hd1)
        # gate1=self.gate1(hd2_1,edge1)

        out=torch.cat((h1,hd1),dim=1)
        out=self.out(out)
        # edge_out=self.edge_out1(gate1)
        # edge_out2=self.up2(self.edge_out2(gate2))
        # edge_out3=self.up4(self.edge_out3(gate3))
        # edgeout4=self.up8(self.edge_out4(gate4))
        # body=self.body_out(h1)
        # print(edge_out.size(),edge_ou2.size(),edge_ou3.size(),edge_ou3.size(),edgeout4.size())
        # (edge_out2, edge_out3, edgeout4, edge_out)
        # if train:
        #     return out,body,edge_out
        return out

class GCT(nn.Module):

    def __init__(self, num_channels, epsilon=1e-5, mode='l2', after_relu=False):
        super(GCT, self).__init__()

        self.alpha = nn.Parameter(torch.ones(1, num_channels, 1, 1))
        self.gamma = nn.Parameter(torch.zeros(1, num_channels, 1, 1))
        self.beta = nn.Parameter(torch.zeros(1, num_channels, 1, 1))
        self.epsilon = epsilon
        self.mode = mode
        self.after_relu = after_relu

    def forward(self, x):

        if self.mode == 'l2':
            embedding = (x.pow(2).sum((2, 3), keepdim=True) +
                         self.epsilon).pow(0.5) * self.alpha
            norm = self.gamma / \
                (embedding.pow(2).mean(dim=1, keepdim=True) + self.epsilon).pow(0.5)

        elif self.mode == 'l1':
            if not self.after_relu:
                _x = torch.abs(x)
            else:
                _x = x
            embedding = _x.sum((2, 3), keepdim=True) * self.alpha
            norm = self.gamma / \
                (torch.abs(embedding).mean(dim=1, keepdim=True) + self.epsilon)

        gate = 1. + torch.tanh(embedding * norm + self.beta)

        return x * gate


class M4(Module):
    def __init__(self,in_channels,out_cannels,filters):
        super(M4, self).__init__()
        self.unet=UNet_3Plus_DeepSup(in_channels,out_cannels,filters=filters)
        up_channels=filters[0]*5
        self.convd5=nn.Sequential(
            nn.Conv2d(filters[4],up_channels,3,1,1),
            nn.BatchNorm2d(up_channels),
            nn.ReLU(True)
        )
        self.h1_5=DownConv(up_channels,up_channels,16)
        self.h2_5=DownConv(up_channels,up_channels,8)
        self.h3_5=DownConv(up_channels,up_channels,4)
        self.h4_5=DownConv(up_channels,up_channels,2)
        self.conv5=nn.Sequential(
            nn.Conv2d(up_channels*5,up_channels,3,1,1),
            nn.BatchNorm2d(up_channels),
            nn.ReLU(True)
        )
        self.upconv5_4=nn.Sequential(
            nn.Conv2d(up_channels,up_channels,3,1,1),
            nn.BatchNorm2d(up_channels),
            nn.ReLU(True)
        )

        self.h1_4=DownConv(up_channels,up_channels,8)
        self.h2_4=DownConv(up_channels,up_channels,4)
        self.h3_4=DownConv(up_channels,up_channels,2)
        self.conv4 = nn.Sequential(
            nn.Conv2d(up_channels * 4, up_channels, 3, 1, 1),
            nn.BatchNorm2d(up_channels),
            nn.ReLU(True)
        )
        self.upconv4_3=nn.Sequential(
            nn.Conv2d(up_channels,up_channels,3,1,1),
            nn.BatchNorm2d(up_channels),
            nn.ReLU(True)
        )
        self.h1_3=DownConv(up_channels,up_channels,4)
        self.h2_3=DownConv(up_channels,up_channels,2)
        self.conv3 = nn.Sequential(
            nn.Conv2d(up_channels * 3, up_channels, 3, 1, 1),
            nn.BatchNorm2d(up_channels),
            nn.ReLU(True)
        )
        self.upconv3_2=nn.Sequential(
            nn.Conv2d(up_channels,up_channels,3,1,1),
            nn.BatchNorm2d(up_channels),
            nn.ReLU(True)
        )
        self.h1_2=DownConv(up_channels,up_channels,2)
        self.conv2 = nn.Sequential(
            nn.Conv2d(up_channels * 2, up_channels, 3, 1, 1),
            nn.BatchNorm2d(up_channels),
            nn.ReLU(True)
        )
        # self.upconv2_1=nn.Sequential(
        #     nn.Conv2d(up_channels,up_channels,3,1,1),
        #     nn.BatchNorm2d(up_channels),
        #     nn.ReLU(True)
        # )

        self.convd1_2=nn.Sequential(
            nn.Conv2d(up_channels,up_channels,3,1,padding=4,dilation=4),
            nn.BatchNorm2d(up_channels)
        )
        self.convd1_3 = nn.Sequential(
            nn.Conv2d(up_channels, up_channels, 3, 1, padding=8, dilation=8),
            nn.BatchNorm2d(up_channels)
        )
        self.convd1_4 = nn.Sequential(
            nn.Conv2d(up_channels, up_channels, 3, 1, padding=12, dilation=12),
            nn.BatchNorm2d(up_channels)
        )
        self.convd1_5 = nn.Sequential(
            nn.Conv2d(up_channels, up_channels, 3, 1, padding=16, dilation=16),
            nn.BatchNorm2d(up_channels)
        )
        self.up5_1=UPConv(up_channels,up_channels,16)
        self.up4_1 = UPConv(up_channels, up_channels, 8)
        self.up3_1 = UPConv(up_channels, up_channels, 4)
        self.up2_1 = UPConv(up_channels, up_channels, 2)
        self.conv1=nn.Sequential(
            nn.Conv2d(up_channels*5,up_channels,3,1,1),
            nn.BatchNorm2d(up_channels),
            nn.ReLU(True)
        )

        self.edge4=EDge(up_channels,1,1)
        self.edge3=EDge(up_channels,1,2)
        self.edge2=EDge(up_channels,1,3)
        # self.edge1=EDge(up_channels,1,4)

        self.gate4 = Gate(up_channels,up_channels)
        self.gate3 = Gate(up_channels, up_channels)
        self.gate2 = Gate(up_channels, up_channels)
        # self.gate1 = Gate(up_channels, up_channels)
        self.out=nn.Sequential(
            nn.Conv2d(up_channels*3,up_channels,3,1,1),
            nn.BatchNorm2d(up_channels),
            nn.ReLU(True),
            nn.Conv2d(up_channels, up_channels, 3, 1, 1),
            nn.BatchNorm2d(up_channels),
            nn.ReLU(True),
            nn.Conv2d(up_channels,out_cannels,1,1,0)
        )
        # self.edge_out1=nn.Sequential(
        #     nn.Conv2d(up_channels,up_channels,3,1,1),
        #     nn.BatchNorm2d(up_channels),
        #     nn.ReLU(True),
        #     nn.Conv2d(up_channels,out_cannels,1,1,0)
        # )
        self.edge_out2 = nn.Sequential(
            nn.Conv2d(up_channels, up_channels, 3, 1, 1),
            nn.BatchNorm2d(up_channels),
            nn.ReLU(True),
            nn.Conv2d(up_channels, out_cannels, 1, 1, 0)
        )
        self.edge_out3 = nn.Sequential(
            nn.Conv2d(up_channels, up_channels, 3, 1, 1),
            nn.BatchNorm2d(up_channels),
            nn.ReLU(True),
            nn.Conv2d(up_channels, out_cannels, 1, 1, 0)
        )
        self.edge_out4 = nn.Sequential(
            nn.Conv2d(up_channels, up_channels, 3, 1, 1),
            nn.BatchNorm2d(up_channels),
            nn.ReLU(True),
            nn.Conv2d(up_channels, out_cannels, 1, 1, 0)
        )
        self.up2=nn.Upsample(scale_factor=2,mode='bilinear')
        self.up4=nn.Upsample(scale_factor=4, mode='bilinear')
        self.up8=nn.Upsample(scale_factor=8, mode='bilinear')
        self.relu=nn.ReLU(True)
        self.body_out=nn.Sequential(
            nn.Conv2d(up_channels,up_channels,3,1,1),
            nn.BatchNorm2d(up_channels),
            nn.ReLU(True),
            nn.Conv2d(up_channels,out_cannels,1,1,0)
        )

    def forward(self,x,train=False):
        h5,h4,h3,h2,h1=self.unet(x,train=True)
        h5=self.convd5(h5)
        # print(h5.size(),h4.size(),h3.size(),h2.size(),h1.size())
        h1_5=self.h1_5(h1)
        h2_5=self.h2_5(h2)
        h3_5=self.h3_5(h3)
        h4_5=self.h4_5(h4)
        hd5=torch.cat((h5,h4_5,h3_5,h2_5,h1_5),dim=1)
        hd5_4=self.upconv5_4(self.up2(self.conv5(hd5)))

        h1_4=self.h1_4(h1)
        h2_4=self.h2_4(h2)
        h3_4=self.h3_4(h3)
        hd4=torch.cat((h4,h3_4,h2_4,h1_4),dim=1)
        hd4=self.conv4(hd4)
        edge4=self.edge4(hd4)
        gate4=self.gate4(hd5_4,edge4)
        hd4_3=self.upconv4_3(self.up2(gate4))

        h1_3=self.h1_3(h1)
        h2_3=self.h2_3(h2)
        hd3=torch.cat((h1_3,h2_3,h3),dim=1)
        hd3=self.conv3(hd3)
        edge3=self.edge3(hd3)
        gate3=self.gate3(hd4_3,edge3)
        hd3_2=self.upconv4_3(self.up2(gate3))

        h1_2=self.h1_2(h1)
        hd2=torch.cat((h1_2,h2),dim=1)
        hd2=self.conv2(hd2)
        edge2=self.edge2(hd2)
        gate2=self.gate2(hd3_2,edge2)
        # hd2_1=self.upconv2_1(self.up2(gate2))

        hd1_2=self.convd1_2(h1)
        hd1_3=self.convd1_3(h1)
        hd1_4=self.convd1_4(h1)
        hd1_5=self.convd1_5(h1)
        up5_1=self.up5_1(h5)
        up4_1=self.up4_1(h4)
        up3_1=self.up3_1(h3)
        up2_1=self.up2_1(h2)
        hd1_2=self.relu(hd1_2+up2_1)
        hd1_3=self.relu(hd1_3+up3_1)
        hd1_4=self.relu(hd1_4+up4_1)
        hd1_5=self.relu(hd1_5+up5_1)
        hd1=torch.cat((h1,hd1_2,hd1_3,hd1_4,hd1_5),dim=1)
        hd1=self.conv1(hd1)
        # edge1=self.edge1(hd1)
        # gate1=self.gate1(hd2_1,edge1)

        out=torch.cat((self.up2(gate2),hd1,h1),dim=1)
        out=self.out(out)
        # edge_out=self.edge_out1(gate1)
        edge_out2=self.up2(self.edge_out2(gate2))
        edge_out3=self.up4(self.edge_out3(gate3))
        edge_out4=self.up8(self.edge_out4(gate4))
        body=self.body_out(hd1)
        # print(edge_out.size(),edge_ou2.size(),edge_ou3.size(),edge_ou3.size(),edgeout4.size())
        # (edge_out2, edge_out3, edgeout4, edge_out)
        if train:
            return out,body,(edge_out4,edge_out3,edge_out2)
        return out




class M5(Module):
    def __init__(self,in_channels,out_cannels,filters):
        super(M5, self).__init__()
        self.unet=UNet_3Plus_DeepSup(in_channels,out_cannels,filters=filters)
        up_channels=filters[0]*5
        self.convd5=nn.Sequential(
            nn.Conv2d(filters[4],up_channels,3,1,1),
            nn.BatchNorm2d(up_channels),
            nn.ReLU(True)
        )
        self.h1_5=DownConv(up_channels,up_channels,16)
        self.h2_5=DownConv(up_channels,up_channels,8)
        self.h3_5=DownConv(up_channels,up_channels,4)
        self.h4_5=DownConv(up_channels,up_channels,2)
        self.conv5=nn.Sequential(
            nn.Conv2d(up_channels*5,up_channels,3,1,1),
            nn.BatchNorm2d(up_channels),
            nn.ReLU(True)
        )
        self.upconv5_4=nn.Sequential(
            nn.Conv2d(up_channels,up_channels,3,1,1),
            nn.BatchNorm2d(up_channels),
            nn.ReLU(True)
        )

        self.h1_4=DownConv(up_channels,up_channels,8)
        self.h2_4=DownConv(up_channels,up_channels,4)
        self.h3_4=DownConv(up_channels,up_channels,2)
        self.conv4 = nn.Sequential(
            nn.Conv2d(up_channels * 4, up_channels, 3, 1, 1),
            nn.BatchNorm2d(up_channels),
            nn.ReLU(True)
        )
        self.upconv4_3=nn.Sequential(
            nn.Conv2d(up_channels,up_channels,3,1,1),
            nn.BatchNorm2d(up_channels),
            nn.ReLU(True)
        )
        self.h1_3=DownConv(up_channels,up_channels,4)
        self.h2_3=DownConv(up_channels,up_channels,2)
        self.conv3 = nn.Sequential(
            nn.Conv2d(up_channels * 3, up_channels, 3, 1, 1),
            nn.BatchNorm2d(up_channels),
            nn.ReLU(True)
        )
        self.upconv3_2=nn.Sequential(
            nn.Conv2d(up_channels,up_channels,3,1,1),
            nn.BatchNorm2d(up_channels),
            nn.ReLU(True)
        )
        self.h1_2=DownConv(up_channels,up_channels,2)
        self.conv2 = nn.Sequential(
            nn.Conv2d(up_channels * 2, up_channels, 3, 1, 1),
            nn.BatchNorm2d(up_channels),
            nn.ReLU(True)
        )
        self.upconv2_1=nn.Sequential(
            nn.Conv2d(up_channels,up_channels,3,1,1),
            nn.BatchNorm2d(up_channels),
            nn.ReLU(True)
        )

        self.convd1_2=nn.Sequential(
            nn.Conv2d(up_channels,up_channels,3,1,padding=4,dilation=4),
            nn.BatchNorm2d(up_channels)
        )
        self.convd1_3 = nn.Sequential(
            nn.Conv2d(up_channels, up_channels, 3, 1, padding=8, dilation=8),
            nn.BatchNorm2d(up_channels)
        )
        self.convd1_4 = nn.Sequential(
            nn.Conv2d(up_channels, up_channels, 3, 1, padding=12, dilation=12),
            nn.BatchNorm2d(up_channels)
        )
        self.convd1_5 = nn.Sequential(
            nn.Conv2d(up_channels, up_channels, 3, 1, padding=16, dilation=16),
            nn.BatchNorm2d(up_channels)
        )
        self.up5_1=UPConv(up_channels,up_channels,16)
        self.up4_1 = UPConv(up_channels, up_channels, 8)
        self.up3_1 = UPConv(up_channels, up_channels, 4)
        self.up2_1 = UPConv(up_channels, up_channels, 2)
        self.conv1=nn.Sequential(
            nn.Conv2d(up_channels*5,up_channels,3,1,1),
            nn.BatchNorm2d(up_channels),
            nn.ReLU(True)
        )

        self.edge4=EDge(up_channels,1,1)
        self.edge3=EDge(up_channels,1,2)
        self.edge2=EDge(up_channels,1,3)
        self.edge1=EDge(up_channels,1,4)

        self.gate4 =  Gate(up_channels,up_channels)
        self.gate3 = Gate(up_channels, up_channels)
        self.gate2 = Gate(up_channels, up_channels)
        self.gate1 = Gate(up_channels, up_channels)
        self.out=nn.Sequential(
            nn.Conv2d(up_channels*3,up_channels,3,1,1),
            nn.BatchNorm2d(up_channels),
            nn.ReLU(True),
            nn.Conv2d(up_channels, up_channels, 3, 1, 1),
            nn.BatchNorm2d(up_channels),
            nn.ReLU(True),
            nn.Conv2d(up_channels,out_cannels,1,1,0)
        )
        self.edge_out1=nn.Sequential(
            nn.Conv2d(up_channels,up_channels,3,1,1),
            nn.BatchNorm2d(up_channels),
            nn.ReLU(True),
            nn.Conv2d(up_channels,out_cannels,1,1,0)
        )
        self.edge_out2 = nn.Sequential(
            nn.Conv2d(up_channels, up_channels, 3, 1, 1),
            nn.BatchNorm2d(up_channels),
            nn.ReLU(True),
            nn.Conv2d(up_channels, out_cannels, 1, 1, 0)
        )
        self.edge_out3 = nn.Sequential(
            nn.Conv2d(up_channels, up_channels, 3, 1, 1),
            nn.BatchNorm2d(up_channels),
            nn.ReLU(True),
            nn.Conv2d(up_channels, out_cannels, 1, 1, 0)
        )
        self.edge_out4 = nn.Sequential(
            nn.Conv2d(up_channels, up_channels, 3, 1, 1),
            nn.BatchNorm2d(up_channels),
            nn.ReLU(True),
            nn.Conv2d(up_channels, out_cannels, 1, 1, 0)
        )
        self.up2=nn.Upsample(scale_factor=2,mode='bilinear')
        self.up4=nn.Upsample(scale_factor=4, mode='bilinear')
        self.up8=nn.Upsample(scale_factor=8, mode='bilinear')
        self.relu=nn.ReLU(True)
        self.body_out=nn.Sequential(
            nn.Conv2d(up_channels,out_cannels,3,1,1)
        )
    def forward(self,x,train=False):
        h5,h4,h3,h2,h1=self.unet(x,train=True)
        h5=self.convd5(h5)
        # print(h5.size(),h4.size(),h3.size(),h2.size(),h1.size())
        h1_5=self.h1_5(h1)
        h2_5=self.h2_5(h2)
        h3_5=self.h3_5(h3)
        h4_5=self.h4_5(h4)
        hd5=torch.cat((h5,h4_5,h3_5,h2_5,h1_5),dim=1)
        hd5_4=self.upconv5_4(self.up2(self.conv5(hd5)))

        h1_4=self.h1_4(h1)
        h2_4=self.h2_4(h2)
        h3_4=self.h3_4(h3)
        hd4=torch.cat((h4,h3_4,h2_4,h1_4),dim=1)
        hd4=self.conv4(hd4)
        edge4=self.edge4(hd4)
        gate4=self.gate4(hd5_4,edge4)
        hd4_3=self.upconv4_3(self.up2(gate4))

        h1_3=self.h1_3(h1)
        h2_3=self.h2_3(h2)
        hd3=torch.cat((h1_3,h2_3,h3),dim=1)
        hd3=self.conv3(hd3)
        edge3=self.edge3(hd3)
        gate3=self.gate3(hd4_3,edge3)
        hd3_2=self.upconv4_3(self.up2(gate3))

        h1_2=self.h1_2(h1)
        hd2=torch.cat((h1_2,h2),dim=1)
        hd2=self.conv2(hd2)
        edge2=self.edge2(hd2)
        gate2=self.gate2(hd3_2,edge2)
        hd2_1=self.upconv2_1(self.up2(gate2))

        hd1_2=self.convd1_2(h1)
        hd1_3=self.convd1_3(h1)
        hd1_4=self.convd1_4(h1)
        hd1_5=self.convd1_5(h1)
        up5_1=self.up5_1(h5)
        up4_1=self.up4_1(h4)
        up3_1=self.up3_1(h3)
        up2_1=self.up2_1(h2)
        hd1_2=self.relu(hd1_2+up2_1)
        hd1_3=self.relu(hd1_3+up3_1)
        hd1_4=self.relu(hd1_4+up4_1)
        hd1_5=self.relu(hd1_5+up5_1)
        hd1=torch.cat((h1,hd1_2,hd1_3,hd1_4,hd1_5),dim=1)
        hd1=self.conv1(hd1)
        edge1=self.edge1(hd1)
        gate1=self.gate1(hd2_1,edge1)

        out=torch.cat((h1,gate1,hd1),dim=1)
        out=self.out(out)
        edge_out=self.edge_out1(gate1)
        edge_out2=self.up2(self.edge_out2(gate2))
        edge_out3=self.up4(self.edge_out3(gate3))
        edgeout4=self.up8(self.edge_out4(gate4))
        body=self.body_out(h1)
        # print(edge_out.size(),edge_ou2.size(),edge_ou3.size(),edge_ou3.size(),edgeout4.size())
        # (edge_ou2, edge_ou3, edgeout4, edge_out)
        if train:
            return out,body,(edge_out2, edge_out3, edgeout4, edge_out)
        return out
if __name__=='__main__':
    # mo=Gate(3,5).cuda()
    # x=torch.rand(1,3,32,32).cuda()
    # y = torch.rand(1, 1, 32, 32).cuda()
    # mo(x,y)
    # print(1)
    net=M3(3,1,(2,4,6,8,10)).cuda()
    summary(net,(3,32,32))