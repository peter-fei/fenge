import torch
import torch.nn as nn
from torch.nn import Module
from torchsummary import summary


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



class DoubleConv(nn.Module):
    def __init__(self,in_channels,out_channels,use_act=True,gct=False):
        super(DoubleConv,self).__init__()
        self.conv1=nn.Sequential(nn.Conv2d(in_channels,out_channels,kernel_size=3,stride=1,padding=1),
                                 nn.BatchNorm2d(out_channels),
                                 nn.ReLU())
        self.conv2=nn.Sequential(nn.Conv2d(out_channels,out_channels,kernel_size=3,stride=1,padding=1),
                                 nn.BatchNorm2d(out_channels),
                                 nn.ReLU(),
                                 )
        self.gct = GCT(in_channels)
        self.usegct=gct

        self.use_act=use_act
        self.relu=nn.ReLU()

    def forward(self,x):
        if self.usegct:
            x=self.gct(x)
        x=self.conv1(x)
        y=self.conv2(x)
        if self.use_act:
            y=self.relu(y)
        return y


class Unet(Module):
    def __init__(self,in_channels,out_channels,filters=(64,128,256,512,1024),usegct=False):
        super(Unet, self).__init__()
        self.conv1=DoubleConv(in_channels,filters[0],gct=usegct)
        self.conv2=DoubleConv(filters[0],filters[1],gct=usegct)
        self.conv3=DoubleConv(filters[1],filters[2],gct=usegct)
        self.conv4=DoubleConv(filters[2],filters[3],gct=usegct)
        self.conv5=DoubleConv(filters[3],filters[4],gct=usegct)
        self.maxpool=nn.MaxPool2d(2,2)
        self.up5=nn.Sequential(
            nn.ConvTranspose2d(filters[4],filters[3],2,2),
            nn.BatchNorm2d(filters[3]),
            nn.ReLU(True)
        )
        self.up_conv4=DoubleConv(2*filters[3],filters[3],True,gct=usegct)
        self.up4=nn.Sequential(
            nn.ConvTranspose2d(filters[3],filters[2],2,2),
            nn.BatchNorm2d(filters[2]),
            nn.ReLU(True)
        )
        self.up_conv3=DoubleConv(2*filters[2],filters[2],True,gct=usegct)
        self.up3=nn.Sequential(
            nn.ConvTranspose2d(filters[2],filters[1],2,2),
            nn.BatchNorm2d(filters[1]),
            nn.ReLU(True)
        )
        self.up_conv2=DoubleConv(2*filters[1],filters[1],True,gct=usegct)
        self.up2=nn.Sequential(
            nn.ConvTranspose2d(filters[1],filters[0],2,2),
            nn.BatchNorm2d(filters[0]),
            nn.ReLU(True)
        )
        self.up_conv1=DoubleConv(2*filters[0],filters[0],True,gct=usegct)
        self.out=nn.Conv2d(filters[0],out_channels,1,1,0)


    def forward(self,x,train=False):
        x1=self.conv1(x)
        p1=self.maxpool(x1)
        x2=self.conv2(p1)
        p2=self.maxpool(x2)
        x3=self.conv3(p2)
        p3=self.maxpool(x3)
        x4=self.conv4(p3)
        p4=self.maxpool(x4)
        x5=self.conv5(p4)
        d5_4=self.up5(x5)
        hd4=torch.cat((x4,d5_4),dim=1)
        hd4=self.up_conv4(hd4)
        d4_3=self.up4(hd4)
        hd3=torch.cat((x3,d4_3),dim=1)
        hd3=self.up_conv3(hd3)
        d3_2=self.up3(hd3)
        hd2=torch.cat((x2,d3_2),dim=1)
        hd2=self.up_conv2(hd2)
        d2_1=self.up2(hd2)
        hd1=torch.cat((x1,d2_1),dim=1)
        hd1=self.up_conv1(hd1)
        out=self.out(hd1)
        if train:
            return (x1,x2,x3,x4,x5,hd4,hd3,hd2,hd1)
        return out





if __name__=='__main__':
    model=Unet(3,1,(4,8,16,32,64)).cuda()
    summary(model,(3,32,32))



