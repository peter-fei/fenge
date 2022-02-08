import torch
from torch.nn import Module
import torch.nn as nn
# from torchsummary import summary
from functools import reduce


class Net(Module):
    def __init__(self,in_channel=3,out_channel=2):
        super(Net, self).__init__()
        self.conv1=nn.Sequential(
            nn.Conv2d(in_channel,1,1,1,0),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
    def forward(self,x):
        x1=self.conv1(x)
        print(x1.shape,x.shape)
        x2=x1*x
        print(torch.max(x2), torch.min(x2))
        print(x2.shape)
        return x2


import numpy as np
import torch
import torch.nn as nn


class Attention_block(nn.Module):
    def __init__(self, F_g, F_l, F_int):
        super(Attention_block, self).__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )

        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )

        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)

        return x * psi


class conv_block(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(conv_block, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch_out, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)


class up_conv(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(up_conv, self).__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.up(x)


class AttU_Net(nn.Module):
    def __init__(self, n_channels=3, n_classes=1, scale_factor=1):
        super(AttU_Net, self).__init__()
        filters = np.array([64, 128, 256, 512, 1024])
        filters = filters // scale_factor
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.scale_factor = scale_factor
        self.Maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.Conv1 = conv_block(ch_in=n_channels, ch_out=filters[0])
        self.Conv2 = conv_block(ch_in=filters[0], ch_out=filters[1])
        self.Conv3 = conv_block(ch_in=filters[1], ch_out=filters[2])
        self.Conv4 = conv_block(ch_in=filters[2], ch_out=filters[3])
        self.Conv5 = conv_block(ch_in=filters[3], ch_out=filters[4])

        self.Up5 = up_conv(ch_in=filters[4], ch_out=filters[3])
        self.Att5 = Attention_block(F_g=filters[3], F_l=filters[3], F_int=filters[2])
        self.Up_conv5 = conv_block(ch_in=filters[4], ch_out=filters[3])

        self.Up4 = up_conv(ch_in=filters[3], ch_out=filters[2])
        self.Att4 = Attention_block(F_g=filters[2], F_l=filters[2], F_int=filters[1])
        self.Up_conv4 = conv_block(ch_in=filters[3], ch_out=filters[2])

        self.Up3 = up_conv(ch_in=filters[2], ch_out=filters[1])
        self.Att3 = Attention_block(F_g=filters[1], F_l=filters[1], F_int=filters[0])
        self.Up_conv3 = conv_block(ch_in=filters[2], ch_out=filters[1])

        self.Up2 = up_conv(ch_in=filters[1], ch_out=filters[0])
        self.Att2 = Attention_block(F_g=filters[0], F_l=filters[0], F_int=filters[0] // 2)
        self.Up_conv2 = conv_block(ch_in=filters[1], ch_out=filters[0])

        self.Conv_1x1 = nn.Conv2d(filters[0], n_classes, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        # encoding path
        x1 = self.Conv1(x)

        x2 = self.Maxpool(x1)
        x2 = self.Conv2(x2)

        x3 = self.Maxpool(x2)
        x3 = self.Conv3(x3)

        x4 = self.Maxpool(x3)
        x4 = self.Conv4(x4)

        x5 = self.Maxpool(x4)
        x5 = self.Conv5(x5)

        # decoding + concat path
        d5 = self.Up5(x5)
        x4 = self.Att5(g=d5, x=x4)
        d5 = torch.cat((x4, d5), dim=1)
        d5 = self.Up_conv5(d5)

        d4 = self.Up4(d5)
        x3 = self.Att4(g=d4, x=x3)
        d4 = torch.cat((x3, d4), dim=1)
        d4 = self.Up_conv4(d4)

        d3 = self.Up3(d4)
        x2 = self.Att3(g=d3, x=x2)
        d3 = torch.cat((x2, d3), dim=1)
        d3 = self.Up_conv3(d3)

        d2 = self.Up2(d3)
        x1 = self.Att2(g=d2, x=x1)
        d2 = torch.cat((x1, d2), dim=1)
        d2 = self.Up_conv2(d2)

        d1 = self.Conv_1x1(d2)

        return d1







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

class Atten_block(Module):
    def __init__(self,in_channels,out_channel,M=5):
        super(Atten_block, self).__init__()
        self.M=M
        # self.index=index
        self.convs=nn.ModuleList()
        for i in range(M):
            self.convs.append(
                nn.Sequential(
                nn.Conv2d(in_channels[i], out_channel, 1, 1, 0),
                nn.BatchNorm2d(out_channel),
                nn.ReLU(True)
                )
            )
        self.sigmoid=nn.Sequential(
            nn.Conv2d(out_channel,1,1,1,0),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
    def forward(self,u,v,w,x,y):
        out=[]
        inputs=[u,v,w,x,y]
        for i ,conv in enumerate(self.convs):
            out.append(conv(inputs[i]))
        u1,v1,w1,x1,y1=out
        res=u1+v1+w1+x1+y1
        # res=torch.zeros(out[0].shape)
        # res=[res+out[i] for i in range(len(out))]
        # res=reduce(lambda u,v,w,x,y:u+v+w+x+y,out)
        res=self.sigmoid(res)
        out=res*u
        return out






class Att_Unet_3plus(Module):
    def __init__(self,in_channel,out_channel,filters=(64,128,248,512,1024)):
        cat=filters[0]
        self.filters=filters
        super(Att_Unet_3plus,self).__init__()
        self.conv1=DoubleConv(in_channel,filters[0])
        self.conv2=DoubleConv(filters[0],filters[1])
        self.conv3=DoubleConv(filters[1],filters[2])
        self.conv4=DoubleConv(filters[2],filters[3])
        self.conv5=DoubleConv(filters[3],filters[4])
        self.maxpool = nn.MaxPool2d(2, 2)

        self.h1_PL_d4=nn.MaxPool2d(8,8)
        self.h2_PL_d4=nn.MaxPool2d(4,4)
        self.h3_PL_d4=nn.MaxPool2d(2,2)
        self.h5_UP_d4=nn.Upsample(scale_factor=2,mode='bilinear')
        cat_channel=filters[0]
        up_channel=cat_channel*5
        self.att4=Atten_block((filters[3],filters[0],filters[1],filters[2],filters[4]),out_channel=cat)
        self.conv_d4=DoubleConv(filters[3]*2,filters[2])

        self.h1_PL_d3=nn.MaxPool2d(4,4)
        self.h2_PL_d3=nn.MaxPool2d(2,2)
        self.h4_UP_d3=nn.Upsample(scale_factor=2,mode='bilinear')
        self.h5_UP_d3=nn.Upsample(scale_factor=4,mode='bilinear')
        self.att3=Atten_block((filters[2],filters[0],filters[1],filters[2],filters[4]),out_channel=cat)
        self.conv_d3=DoubleConv(filters[2]*2,filters[1])

        self.h1_PL_d2=nn.MaxPool2d(2,2)
        self.h3_UPd2=nn.Upsample(scale_factor=2,mode='bilinear')
        self.h4_UP_d2=nn.Upsample(scale_factor=4,mode='bilinear')
        self.h5_UP_d2=nn.Upsample(scale_factor=8,mode='bilinear')
        self.att2=Atten_block((filters[1],filters[0],filters[1],filters[2],filters[4]),out_channel=cat)
        self.conv_d2=DoubleConv(filters[1]*2,filters[0])

        self.h2_UP_d1=nn.Upsample(scale_factor=2,mode='bilinear')
        self.h3_UP_d1=nn.Upsample(scale_factor=4, mode='bilinear')
        self.h4_UP_d1=nn.Upsample(scale_factor=8,mode='bilinear')
        self.h5_UP_d1=nn.Upsample(scale_factor=16,mode='bilinear')
        self.att1=Atten_block((filters[0],filters[0],filters[1],filters[2],filters[4]),out_channel=cat)
        self.conv_d1=DoubleConv(filters[0]*2,filters[0]//2)
        self.conv_out=nn.Conv2d(filters[0]//2,out_channel,1,1,0)




    def forward(self,x):
        h1=self.conv1(x)
        h2=self.maxpool(h1)
        h2=self.conv2(h2)
        h3=self.maxpool(h2)
        h3=self.conv3(h3)
        h4=self.maxpool(h3)
        h4=self.conv4(h4)
        h5=self.maxpool(h4)
        hd5=self.conv5(h5)

        h1_d4=self.h1_PL_d4(h1)
        h2_d4=self.h2_PL_d4(h2)
        h3_d4=self.h3_PL_d4(h3)
        h5_d4=self.h5_UP_d4(hd5)
        # print(h4.shape,h1_d4.shape,h2_d4.shape,h3_d4.shape,h5_d4.shape)
        att_4=self.att4(h4,h1_d4,h2_d4,h3_d4,h5_d4)
        # print(att_4.shape)
        hd4=torch.cat((h4,att_4),dim=1)

        hd4=self.conv_d4(hd4)

        h1_d3=self.h1_PL_d3(h1)
        h2_d3=self.h2_PL_d3(h2)
        h4_d3=self.h4_UP_d3(hd4)
        h5_d3=self.h5_UP_d3(hd5)
        att_3=self.att3(h3,h1_d3,h2_d3,h4_d3,h5_d3)
        hd3=torch.cat((h3,att_3),dim=1)
        hd3=self.conv_d3(hd3)

        h1_d2=self.h1_PL_d2(h1)
        h3_d2=self.h3_UPd2(hd3)
        h4_d2=self.h4_UP_d2(hd4)
        h5_d2=self.h5_UP_d2(hd5)
        att_2=self.att2(h2,h1_d2,h3_d2,h4_d2,h5_d2)
        hd2=torch.cat((h2,att_2),dim=1)
        hd2=self.conv_d2(hd2)

        h2_d1=self.h2_UP_d1(hd2)
        h3_d1=self.h3_UP_d1(hd3)
        h4_d1=self.h4_UP_d1(hd4)
        h5_d1=self.h5_UP_d1(hd5)
        att_1=self.att1(h1,h2_d1,h3_d1,h4_d1,h5_d1)
        hd1=torch.cat((h1,att_1),dim=1)
        out=self.conv_d1(hd1)
        out=self.conv_out(out)

        return out



class A(Module):
    def __init__(self):
        super(A, self).__init__()
        self.att1=Atten_block()
if __name__=='__main__':
    aaa=2
    if aaa==1:
        net=Net().cuda()
    if aaa==2:
        net=Att_Unet_3plus(3,1).cuda()
    summary(net,(3,32,32))