import torch
import torch.nn as nn
from torch.nn import Module
from torchsummary import summary
from functools import reduce

class SKConv(nn.Module):
    def __init__(self,in_channel,out_channel,M=2,r=16,L=32):
        super(SKConv, self).__init__()
        self.M=M
        self.out_channel=out_channel
        d=max(in_channel//r,L)
        self.conv=nn.ModuleList()
        for i in range(M):
            self.conv.append(
                nn.Sequential(
                nn.Conv2d(in_channel,out_channel,3,1,padding=1+i,dilation=1+i,groups=4),#group=32
                nn.BatchNorm2d(out_channel),
                nn.ReLU(True)
                )
            )
        self.global_pool=nn.AdaptiveAvgPool2d(1)
        self.fc1=nn.Sequential(
            nn.Conv2d(out_channel,d,1,1,0),
            # nn.BatchNorm2d(d),
            # nn.ReLU(True)
        )
        self.fc2=nn.Sequential(
            nn.Conv2d(d,M*out_channel,1,1,0)
        )
        self.softmax=nn.Softmax(dim=1)
        self.conv2=nn.Sequential(
            nn.Conv2d(out_channel,in_channel,3,1,1),
            nn.BatchNorm2d(in_channel)
        )
        self.relu=nn.ReLU(True)
        self.conv3=nn.Sequential(
            nn.Conv2d(in_channel,out_channel,3,1,1),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(True)
        )
    def forward(self,x):
        batch_size=x.size(0)
        out=[]
        for i,conv in enumerate(self.conv):
            out.append(conv(x))
        U=reduce(lambda x,y:x+y,out)
        S=self.global_pool(U)
        Z=self.fc1(S)
        a_b=self.fc2(Z)
        a_b=a_b.reshape(batch_size,self.M,self.out_channel,-1)

        a_b=self.softmax(a_b)
        a_b=list(torch.chunk(a_b,self.M,dim=1))
        a_b=list(map(lambda x:x.reshape(batch_size,self.out_channel,1,1),a_b))
        V=list(map(lambda x,y:x*y,a_b,out))
        V=reduce(lambda x,y:x+y,V)
        V=self.conv2(V)
        # print(V.shape, x.shape)
        out=self.relu(V+x)
        out=self.conv3(out)
        return out


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

class SKUnet_3plus(Module):
    def __init__(self,in_channel,out_channel,filters=(64,128,248,512,1024),M=2):
        super(SKUnet_3plus, self).__init__()
        self.M=M
        cat_channel=8
        cat_blocks=5
        up_channel=cat_blocks*cat_channel

        self.conv1=DoubleConv(in_channel,filters[0])
        self.conv2=DoubleConv(filters[0],filters[1])
        self.conv3=DoubleConv(filters[1],filters[2])
        self.conv4=DoubleConv(filters[2],filters[3])
        self.conv5=DoubleConv(filters[3],filters[4])
        self.maxpool=nn.MaxPool2d(2,2)

        self.h1_d4=nn.MaxPool2d(8,8)
        self.h1_d4_conv=SKConv(filters[0],cat_channel)
        # self.h1_d4_conv=nn.Sequential(
        #     nn.Conv2d(filters[0],cat_channel,1,1,0),
        #     nn.BatchNorm2d(cat_channel),
        #     nn.ReLU(True)
        # )
        self.h2_d4=nn.MaxPool2d(4,4)
        self.h2_d4_conv=SKConv(filters[1],cat_channel)
        # self.h2_d4_conv=nn.Sequential(
        #     nn.Conv2d(filters[1],cat_channel,1,1,0),
        #     nn.BatchNorm2d(cat_channel),
        #     nn.ReLU(True)
        # )
        self.h3_d4=nn.MaxPool2d(2,2)
        self.h3_d4_conv =SKConv(filters[2],cat_channel)
        # self.h3_d4_conv=nn.Sequential(
        #     nn.Conv2d(filters[2],cat_channel,1,1,0),
        #     nn.BatchNorm2d(cat_channel),
        #     nn.ReLU(True)
        # )
        self.h4_d4_conv =SKConv(filters[3],cat_channel)
        # self.h4_d4_conv=nn.Sequential(
        #     nn.Conv2d(filters[3],cat_channel,1,1,0),
        #     nn.BatchNorm2d(cat_channel),
        #     nn.ReLU()
        # )
        self.h5_d4=nn.Upsample(scale_factor=2,mode='bilinear',align_corners=True)
        self.h5_d4_conv =SKConv(filters[4],cat_channel)
        # self.h5_d4_conv=nn.Sequential(
        #     nn.Conv2d(filters[4],cat_channel,1,1,0),
        #     nn.BatchNorm2d(cat_channel),
        #     nn.ReLU()
        # )
        self.conv4_d1=nn.Sequential(
            nn.Conv2d(up_channel,up_channel,3,1,1),
            nn.BatchNorm2d(up_channel),
            nn.ReLU(True)
        )

        self.h1_d3=nn.MaxPool2d(4,4)
        self.h1_d3_conv =SKConv(filters[0],cat_channel)
        # self.h1_d3_conv=nn.Sequential(
        #     nn.Conv2d(filters[0],cat_channel,1,1,0),
        #     nn.BatchNorm2d(cat_channel),
        #     nn.ReLU()
        # )
        self.h2_d3=nn.MaxPool2d(2,2)
        self.h2_d3_conv =SKConv(filters[1],cat_channel)
        # self.h2_d3_conv=nn.Sequential(
        #     nn.Conv2d(filters[1],cat_channel,1,1,0),
        #     nn.BatchNorm2d(cat_channel),
        #     nn.ReLU()
        # )
        self.h3_d3_conv =SKConv(filters[2],cat_channel)
        # self.h3_d3_conv=nn.Sequential(
        #     nn.Conv2d(filters[2],cat_channel,1,1,0),
        #     nn.BatchNorm2d(cat_channel),
        #     nn.ReLU(True)
        # )
        self.h4_d3=nn.Upsample(scale_factor=2,mode='bilinear')
        self.h4_d3_conv =SKConv(up_channel,cat_channel)
        # self.h4_d3_conv=nn.Sequential(
        #     nn.Conv2d(up_channel,cat_channel,1,1,0),
        #     nn.BatchNorm2d(cat_channel),
        #     nn.ReLU(True)
        # )
        self.h5_d3=nn.Upsample(scale_factor=4,mode='bilinear')
        self.h5_d3_conv =SKConv(filters[4],cat_channel)
        # self.h5_d3_conv=nn.Sequential(
        #     nn.Conv2d(filters[4],cat_channel,1,1,0),
        #     nn.BatchNorm2d(cat_channel),
        #     nn.ReLU(True)
        # )
        self.conv3_d1 = nn.Sequential(
            nn.Conv2d(up_channel, up_channel, 3, 1, 1),
            nn.BatchNorm2d(up_channel),
            nn.ReLU(True)
        )

        self.h1_d2=nn.MaxPool2d(2,2)
        self.h1_d2_conv =SKConv(filters[0],cat_channel)
        # self.h1_d2_conv=nn.Sequential(
        #     nn.Conv2d(filters[0],cat_channel,1,1,0),
        #     nn.BatchNorm2d(cat_channel),
        #     nn.ReLU(True)
        # )
        self.h2_d2_conv =SKConv(filters[1],cat_channel)
        # self.h2_d2_conv=nn.Sequential(
        #     nn.Conv2d(filters[1],cat_channel,1,1,0),
        #     nn.BatchNorm2d(cat_channel),
        #     nn.ReLU(True)
        # )
        self.h3_d2=nn.Upsample(scale_factor=2,mode='bilinear')
        self.h3_d2_conv =SKConv(up_channel,cat_channel)
        # self.h3_d2_conv=nn.Sequential(
        #     nn.Conv2d(up_channel,cat_channel,1,1,0),
        #     nn.BatchNorm2d(cat_channel),
        #     nn.ReLU(True)
        # )
        self.h4_d2=nn.Upsample(scale_factor=4,mode='bilinear')
        self.h4_d2_conv =SKConv(up_channel,cat_channel)
        # self.h4_d2_conv=nn.Sequential(
        #     nn.Conv2d(up_channel,cat_channel,1,1,0),
        #     nn.BatchNorm2d(cat_channel),
        #     nn.ReLU(True)
        # )
        self.h5_d2=nn.Upsample(scale_factor=8,mode='bilinear')
        self.h5_d2_conv =SKConv(filters[4],cat_channel)
        # self.h5_d2_conv=nn.Sequential(
        #     nn.Conv2d(filters[4],cat_channel,1,1,0),
        #     nn.BatchNorm2d(cat_channel),
        #     nn.ReLU(True)
        # )
        self.conv2_d1 = nn.Sequential(
            nn.Conv2d(up_channel, up_channel, 3, 1, 1),
            nn.BatchNorm2d(up_channel),
            nn.ReLU(True)
        )
        self.h1_d1_conv =SKConv(filters[0],cat_channel)
        # self.h1_d1_conv=nn.Sequential(
        #     nn.Conv2d(filters[0],cat_channel,1,1,0),
        #     nn.BatchNorm2d(cat_channel),
        #     nn.ReLU(True)
        # )
        self.h2_d1=nn.Upsample(scale_factor=2,mode='bilinear')
        self.h2_d1_conv =SKConv(up_channel,cat_channel)
        # self.h2_d1_conv=nn.Sequential(
        #     nn.Conv2d(up_channel,cat_channel,1,1,0),
        #     nn.BatchNorm2d(cat_channel),
        #     nn.ReLU(True)
        # )
        self.h3_d1=nn.Upsample(scale_factor=4,mode='bilinear')
        self.h3_d1_conv =SKConv(up_channel,cat_channel)
        # self.h3_d1_conv=nn.Sequential(
        #     nn.Conv2d(up_channel,cat_channel,1,1,0),
        #     nn.BatchNorm2d(cat_channel),
        #     nn.ReLU(True)
        # )
        self.h4_d1=nn.Upsample(scale_factor=8,mode='bilinear')
        self.h4_d1_conv =SKConv(up_channel,cat_channel)
        # self.h4_d1_conv=nn.Sequential(
        #     nn.Conv2d(up_channel, cat_channel, 1, 1, 0),
        #     nn.BatchNorm2d(cat_channel),
        #     nn.ReLU(True)
        # )
        self.h5_d1=nn.Upsample(scale_factor=16,mode='bilinear')
        self.h5_d1_conv =SKConv(filters[4],cat_channel)
        # self.h5_d1_conv=nn.Sequential(
        #     nn.Conv2d(filters[4], cat_channel, 1, 1, 0),
        #     nn.BatchNorm2d(cat_channel),
        #     nn.ReLU(True)
        # )
        self.out_conv=nn.Conv2d(up_channel,out_channel,1,1,0)


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

        h1_d4=self.h1_d4(h1)
        h1_d4=self.h1_d4_conv(h1_d4)
        h2_d4=self.h2_d4(h2)
        h2_d4=self.h2_d4_conv(h2_d4)
        h3_d4=self.h3_d4(h3)
        h3_d4=self.h3_d4_conv(h3_d4)
        h4_d4=self.h4_d4_conv(h4)
        h5_d4=self.h5_d4(hd5)
        h5_d4=self.h5_d4_conv(h5_d4)
        # print(h1_d4.shape,h2_d4.shape,h3_d4.shape,h4_d4.shape,h5_d4.shape)
        hd4=torch.cat((h1_d4,h2_d4,h3_d4,h4_d4,h5_d4),dim=1)
        # print(hd4.shape)
        hd4=self.conv4_d1(hd4)

        h1_d3=self.h1_d3(h1)
        h1_d3=self.h1_d3_conv(h1_d3)
        h2_d3=self.h2_d3(h2)
        h2_d3=self.h2_d3_conv(h2_d3)
        h3_d3=self.h3_d3_conv(h3)
        h4_d3=self.h4_d3(hd4)
        h4_d3=self.h4_d3_conv(h4_d3)
        h5_d3=self.h5_d3(hd5)
        # print(h5_d3.shape)
        h5_d3=self.h5_d3_conv(h5_d3)
        hd3=torch.cat((h1_d3,h2_d3,h3_d3,h4_d3,h5_d3),dim=1)
        hd3=self.conv3_d1(hd3)

        h1_d2=self.h1_d2(h1)
        h1_d2=self.h1_d2_conv(h1_d2)
        h2_d2=self.h2_d2_conv(h2)
        h3_d2=self.h3_d2(hd3)
        h3_d2=self.h3_d2_conv(h3_d2)
        h4_d2=self.h4_d2(hd4)
        h4_d2=self.h4_d2_conv(h4_d2)
        h5_d2=self.h5_d2(hd5)
        h5_d2=self.h5_d2_conv(h5_d2)
        hd2=torch.cat((h1_d2,h2_d2,h3_d2,h4_d2,h5_d2),dim=1)
        hd2=self.conv2_d1(hd2)

        h1_d1=self.h1_d1_conv(h1)
        h2_d1=self.h2_d1(hd2)
        h2_d1=self.h2_d1_conv(h2_d1)
        h3_d1=self.h3_d1(hd3)
        h3_d1=self.h3_d1_conv(h3_d1)
        h4_d1=self.h4_d1(hd4)
        h4_d1=self.h4_d1_conv(h4_d1)
        h5_d1=self.h5_d1(hd5)
        h5_d1=self.h5_d1_conv(h5_d1)
        hd1=torch.cat((h1_d1,h2_d1,h3_d1,h4_d1,h5_d1),dim=1)
        out=self.out_conv(hd1)

        return out
if __name__=='__main__':
    net=SKUnet_3plus(3,1,(4,16,32,64,128)).cuda()
    summary(net,(3,128,128))

