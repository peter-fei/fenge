import torch
import torch.nn as nn
from torch.nn import Module
# from torchsummary import summary
from torch.nn import functional as F
import cv2

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

class Unet_3plus(Module):
    def __init__(self,in_channel,out_channel,filters=(64,128,248,512,1024)):
        super(Unet_3plus, self).__init__()
        cat_channel=filters[0]
        cat_blocks=5
        up_channel=cat_blocks*cat_channel

        self.conv1=DoubleConv(in_channel,filters[0])
        self.conv2=DoubleConv(filters[0],filters[1])
        self.conv3=DoubleConv(filters[1],filters[2])
        self.conv4=DoubleConv(filters[2],filters[3])
        self.conv5=DoubleConv(filters[3],filters[4])
        self.maxpool=nn.MaxPool2d(2,2)

        self.h1_d4=nn.MaxPool2d(8,8)
        self.h1_d4_conv=nn.Sequential(
            nn.Conv2d(filters[0],cat_channel,1,1,0),
            nn.BatchNorm2d(cat_channel),
            nn.ReLU(True)
        )
        self.h2_d4=nn.MaxPool2d(4,4)
        self.h2_d4_conv=nn.Sequential(
            nn.Conv2d(filters[1],cat_channel,1,1,0),
            nn.BatchNorm2d(cat_channel),
            nn.ReLU(True)
        )
        self.h3_d4=nn.MaxPool2d(2,2)
        self.h3_d4_conv=nn.Sequential(
            nn.Conv2d(filters[2],cat_channel,1,1,0),
            nn.BatchNorm2d(cat_channel),
            nn.ReLU(True)
        )
        self.h4_d4_conv=nn.Sequential(
            nn.Conv2d(filters[3],cat_channel,1,1,0),
            nn.BatchNorm2d(cat_channel),
            nn.ReLU()
        )
        self.h5_d4=nn.Upsample(scale_factor=2,mode='bilinear',align_corners=True)
        self.h5_d4_conv=nn.Sequential(
            nn.Conv2d(filters[4],cat_channel,1,1,0),
            nn.BatchNorm2d(cat_channel),
            nn.ReLU()
        )
        self.conv4_d1=nn.Sequential(
            nn.Conv2d(up_channel,up_channel,3,1,1),
            nn.BatchNorm2d(up_channel),
            nn.ReLU(True)
        )

        self.h1_d3=nn.MaxPool2d(4,4)
        self.h1_d3_conv=nn.Sequential(
            nn.Conv2d(filters[0],cat_channel,1,1,0),
            nn.BatchNorm2d(cat_channel),
            nn.ReLU()
        )
        self.h2_d3=nn.MaxPool2d(2,2)
        self.h2_d3_conv=nn.Sequential(
            nn.Conv2d(filters[1],cat_channel,1,1,0),
            nn.BatchNorm2d(cat_channel),
            nn.ReLU()
        )
        self.h3_d3_conv=nn.Sequential(
            nn.Conv2d(filters[2],cat_channel,1,1,0),
            nn.BatchNorm2d(cat_channel),
            nn.ReLU(True)
        )
        self.h4_d3=nn.Upsample(scale_factor=2,mode='bilinear')
        self.h4_d3_conv=nn.Sequential(
            nn.Conv2d(up_channel,cat_channel,1,1,0),
            nn.BatchNorm2d(cat_channel),
            nn.ReLU(True)
        )
        self.h5_d3=nn.Upsample(scale_factor=4,mode='bilinear')
        self.h5_d3_conv=nn.Sequential(
            nn.Conv2d(filters[4],cat_channel,1,1,0),
            nn.BatchNorm2d(cat_channel),
            nn.ReLU(True)
        )
        self.conv3_d1 = nn.Sequential(
            nn.Conv2d(up_channel, up_channel, 3, 1, 1),
            nn.BatchNorm2d(up_channel),
            nn.ReLU(True)
        )

        self.h1_d2=nn.MaxPool2d(2,2)
        self.h1_d2_conv=nn.Sequential(
            nn.Conv2d(filters[0],cat_channel,1,1,0),
            nn.BatchNorm2d(cat_channel),
            nn.ReLU(True)
        )
        self.h2_d2_conv=nn.Sequential(
            nn.Conv2d(filters[1],cat_channel,1,1,0),
            nn.BatchNorm2d(cat_channel),
            nn.ReLU(True)
        )
        self.h3_d2=nn.Upsample(scale_factor=2,mode='bilinear')
        self.h3_d2_conv=nn.Sequential(
            nn.Conv2d(up_channel,cat_channel,1,1,0),
            nn.BatchNorm2d(cat_channel),
            nn.ReLU(True)
        )
        self.h4_d2=nn.Upsample(scale_factor=4,mode='bilinear')
        self.h4_d2_conv=nn.Sequential(
            nn.Conv2d(up_channel,cat_channel,1,1,0),
            nn.BatchNorm2d(cat_channel),
            nn.ReLU(True)
        )
        self.h5_d2=nn.Upsample(scale_factor=8,mode='bilinear')
        self.h5_d2_conv=nn.Sequential(
            nn.Conv2d(filters[4],cat_channel,1,1,0),
            nn.BatchNorm2d(cat_channel),
            nn.ReLU(True)
        )
        self.conv2_d1 = nn.Sequential(
            nn.Conv2d(up_channel, up_channel, 3, 1, 1),
            nn.BatchNorm2d(up_channel),
            nn.ReLU(True)
        )

        self.h1_d1_conv=nn.Sequential(
            nn.Conv2d(filters[0],cat_channel,1,1,0),
            nn.BatchNorm2d(cat_channel),
            nn.ReLU(True)
        )
        self.h2_d1=nn.Upsample(scale_factor=2,mode='bilinear')
        self.h2_d1_conv=nn.Sequential(
            nn.Conv2d(up_channel,cat_channel,1,1,0),
            nn.BatchNorm2d(cat_channel),
            nn.ReLU(True)
        )
        self.h3_d1=nn.Upsample(scale_factor=4,mode='bilinear')
        self.h3_d1_conv=nn.Sequential(
            nn.Conv2d(up_channel,cat_channel,1,1,0),
            nn.BatchNorm2d(cat_channel),
            nn.ReLU(True)
        )
        self.h4_d1=nn.Upsample(scale_factor=8,mode='bilinear')
        self.h4_d1_conv=nn.Sequential(
            nn.Conv2d(up_channel, cat_channel, 1, 1, 0),
            nn.BatchNorm2d(cat_channel),
            nn.ReLU(True)
        )
        self.h5_d1=nn.Upsample(scale_factor=16,mode='bilinear')
        self.h5_d1_conv=nn.Sequential(
            nn.Conv2d(filters[4], cat_channel, 1, 1, 0),
            nn.BatchNorm2d(cat_channel),
            nn.ReLU(True)
        )
        self.conv1_d1=nn.Sequential(
            nn.Conv2d(up_channel,filters[0],3,1,1),
            nn.BatchNorm2d(filters[0]),
            nn.ReLU(True)
        )
        self.out_conv=nn.Conv2d(filters[0],out_channel,1,1,0)
        self.conv2_out=nn.Conv2d(up_channel,out_channel,1,1,0)

    def forward(self,x,res=True,train=False,res3=False):
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
        hd4=torch.cat((h1_d4,h2_d4,h3_d4,h4_d4,h5_d4),dim=1)
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
        # print(h1_d2.size(),h2_d2.size(),h3_d2.size(),h4_d2.size(),h5_d2.size())
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
        hd1=self.conv1_d1(hd1)
        if res:
            hd1=self.out_conv(hd1)
        if train:
            return hd5,hd4,hd3,hd2,hd1
        if res3:
            hd2=self.conv2_out(hd2)
            return hd2,hd1
        return hd1


class unetConv2(nn.Module):
    def __init__(self, in_size, out_size, is_batchnorm, n=2, ks=3, stride=1, padding=1):
        super(unetConv2, self).__init__()
        self.n = n
        self.ks = ks
        self.stride = stride
        self.padding = padding
        s = stride
        p = padding
        if is_batchnorm:
            for i in range(1, n + 1):
                conv = nn.Sequential(nn.Conv2d(in_size, out_size, ks, s, p),
                                     nn.BatchNorm2d(out_size),
                                     nn.ReLU(inplace=True), )
                setattr(self, 'conv%d' % i, conv)
                in_size = out_size

        else:
            for i in range(1, n + 1):
                conv = nn.Sequential(nn.Conv2d(in_size, out_size, ks, s, p),
                                     nn.ReLU(inplace=True), )
                setattr(self, 'conv%d' % i, conv)
                in_size = out_size

        # initialise the blocks
        for m in self.children():
            init_weights(m, init_type='kaiming')

    def forward(self, inputs):
        x = inputs
        for i in range(1, self.n + 1):
            conv = getattr(self, 'conv%d' % i)
            x = conv(x)

        return x
class UNet_3Plus_DeepSup(nn.Module):
    def __init__(self, in_channels=3, n_classes=1, feature_scale=4, filters = [64, 128, 256, 512, 1024],is_deconv=True, is_batchnorm=True):
        super(UNet_3Plus_DeepSup, self).__init__()
        self.is_deconv = is_deconv
        self.in_channels = in_channels
        self.is_batchnorm = is_batchnorm
        self.feature_scale = feature_scale
        ## -------------Encoder--------------
        self.conv1 = unetConv2(self.in_channels, filters[0], self.is_batchnorm)
        self.maxpool1 = nn.MaxPool2d(kernel_size=2)

        self.conv2 = unetConv2(filters[0], filters[1], self.is_batchnorm)
        self.maxpool2 = nn.MaxPool2d(kernel_size=2)

        self.conv3 = unetConv2(filters[1], filters[2], self.is_batchnorm)
        self.maxpool3 = nn.MaxPool2d(kernel_size=2)

        self.conv4 = unetConv2(filters[2], filters[3], self.is_batchnorm)
        self.maxpool4 = nn.MaxPool2d(kernel_size=2)

        self.conv5 = unetConv2(filters[3], filters[4], self.is_batchnorm)

        ## -------------Decoder--------------
        self.CatChannels = filters[0]
        self.CatBlocks = 5
        self.UpChannels = self.CatChannels * self.CatBlocks

        '''stage 4d'''
        # h1->320*320, hd4->40*40, Pooling 8 times
        self.h1_PT_hd4 = nn.MaxPool2d(8, 8,ceil_mode=True)
        self.h1_PT_hd4_conv = nn.Conv2d(filters[0], self.CatChannels, 3, padding=1)
        self.h1_PT_hd4_bn = nn.BatchNorm2d(self.CatChannels)
        self.h1_PT_hd4_relu = nn.ReLU(inplace=True)

        # h2->160*160, hd4->40*40, Pooling 4 times
        self.h2_PT_hd4 = nn.MaxPool2d(4, 4, ceil_mode=True)
        self.h2_PT_hd4_conv = nn.Conv2d(filters[1], self.CatChannels, 3, padding=1)
        self.h2_PT_hd4_bn = nn.BatchNorm2d(self.CatChannels)
        self.h2_PT_hd4_relu = nn.ReLU(inplace=True)

        # h3->80*80, hd4->40*40, Pooling 2 times
        self.h3_PT_hd4 = nn.MaxPool2d(2, 2, ceil_mode=True)
        self.h3_PT_hd4_conv = nn.Conv2d(filters[2], self.CatChannels, 3, padding=1)
        self.h3_PT_hd4_bn = nn.BatchNorm2d(self.CatChannels)
        self.h3_PT_hd4_relu = nn.ReLU(inplace=True)

        # h4->40*40, hd4->40*40, Concatenation
        self.h4_Cat_hd4_conv = nn.Conv2d(filters[3], self.CatChannels, 3, padding=1)
        self.h4_Cat_hd4_bn = nn.BatchNorm2d(self.CatChannels)
        self.h4_Cat_hd4_relu = nn.ReLU(inplace=True)

        # hd5->20*20, hd4->40*40, Upsample 2 times
        self.hd5_UT_hd4 = nn.Upsample(scale_factor=2, mode='bilinear')  # 14*14
        self.hd5_UT_hd4_conv = nn.Conv2d(filters[4], self.CatChannels, 3, padding=1)
        self.hd5_UT_hd4_bn = nn.BatchNorm2d(self.CatChannels)
        self.hd5_UT_hd4_relu = nn.ReLU(inplace=True)

        # fusion(h1_PT_hd4, h2_PT_hd4, h3_PT_hd4, h4_Cat_hd4, hd5_UT_hd4)
        self.conv4d_1 = nn.Conv2d(self.UpChannels, self.UpChannels, 3, padding=1)  # 16
        self.bn4d_1 = nn.BatchNorm2d(self.UpChannels)
        self.relu4d_1 = nn.ReLU(inplace=True)

        '''stage 3d'''
        # h1->320*320, hd3->80*80, Pooling 4 times
        self.h1_PT_hd3 = nn.MaxPool2d(4, 4)
        self.h1_PT_hd3_conv = nn.Conv2d(filters[0], self.CatChannels, 3, padding=1)
        self.h1_PT_hd3_bn = nn.BatchNorm2d(self.CatChannels)
        self.h1_PT_hd3_relu = nn.ReLU(inplace=True)

        # h2->160*160, hd3->80*80, Pooling 2 times
        self.h2_PT_hd3 = nn.MaxPool2d(2, 2, ceil_mode=True)
        self.h2_PT_hd3_conv = nn.Conv2d(filters[1], self.CatChannels, 3, padding=1)
        self.h2_PT_hd3_bn = nn.BatchNorm2d(self.CatChannels)
        self.h2_PT_hd3_relu = nn.ReLU(inplace=True)

        # h3->80*80, hd3->80*80, Concatenation
        self.h3_Cat_hd3_conv = nn.Conv2d(filters[2], self.CatChannels, 3, padding=1)
        self.h3_Cat_hd3_bn = nn.BatchNorm2d(self.CatChannels)
        self.h3_Cat_hd3_relu = nn.ReLU(inplace=True)

        # hd4->40*40, hd4->80*80, Upsample 2 times
        self.hd4_UT_hd3 = nn.Upsample(scale_factor=2, mode='bilinear')  # 14*14
        self.hd4_UT_hd3_conv = nn.Conv2d(self.UpChannels, self.CatChannels, 3, padding=1)
        self.hd4_UT_hd3_bn = nn.BatchNorm2d(self.CatChannels)
        self.hd4_UT_hd3_relu = nn.ReLU(inplace=True)

        # hd5->20*20, hd4->80*80, Upsample 4 times
        self.hd5_UT_hd3 = nn.Upsample(scale_factor=4, mode='bilinear')  # 14*14
        self.hd5_UT_hd3_conv = nn.Conv2d(filters[4], self.CatChannels, 3, padding=1)
        self.hd5_UT_hd3_bn = nn.BatchNorm2d(self.CatChannels)
        self.hd5_UT_hd3_relu = nn.ReLU(inplace=True)

        # fusion(h1_PT_hd3, h2_PT_hd3, h3_Cat_hd3, hd4_UT_hd3, hd5_UT_hd3)
        self.conv3d_1 = nn.Conv2d(self.UpChannels, self.UpChannels, 3, padding=1)  # 16
        self.bn3d_1 = nn.BatchNorm2d(self.UpChannels)
        self.relu3d_1 = nn.ReLU(inplace=True)

        '''stage 2d '''
        # h1->320*320, hd2->160*160, Pooling 2 times
        self.h1_PT_hd2 = nn.MaxPool2d(2, 2)
        self.h1_PT_hd2_conv = nn.Conv2d(filters[0], self.CatChannels, 3, padding=1)
        self.h1_PT_hd2_bn = nn.BatchNorm2d(self.CatChannels)
        self.h1_PT_hd2_relu = nn.ReLU(inplace=True)

        # h2->160*160, hd2->160*160, Concatenation
        self.h2_Cat_hd2_conv = nn.Conv2d(filters[1], self.CatChannels, 3, padding=1)
        self.h2_Cat_hd2_bn = nn.BatchNorm2d(self.CatChannels)
        self.h2_Cat_hd2_relu = nn.ReLU(inplace=True)

        # hd3->80*80, hd2->160*160, Upsample 2 times
        self.hd3_UT_hd2 = nn.Upsample(scale_factor=2, mode='bilinear')  # 14*14
        self.hd3_UT_hd2_conv = nn.Conv2d(self.UpChannels, self.CatChannels, 3, padding=1)
        self.hd3_UT_hd2_bn = nn.BatchNorm2d(self.CatChannels)
        self.hd3_UT_hd2_relu = nn.ReLU(inplace=True)

        # hd4->40*40, hd2->160*160, Upsample 4 times
        self.hd4_UT_hd2 = nn.Upsample(scale_factor=4, mode='bilinear')  # 14*14
        self.hd4_UT_hd2_conv = nn.Conv2d(self.UpChannels, self.CatChannels, 3, padding=1)
        self.hd4_UT_hd2_bn = nn.BatchNorm2d(self.CatChannels)
        self.hd4_UT_hd2_relu = nn.ReLU(inplace=True)

        # hd5->20*20, hd2->160*160, Upsample 8 times
        self.hd5_UT_hd2 = nn.Upsample(scale_factor=8, mode='bilinear')  # 14*14
        self.hd5_UT_hd2_conv = nn.Conv2d(filters[4], self.CatChannels, 3, padding=1)
        self.hd5_UT_hd2_bn = nn.BatchNorm2d(self.CatChannels)
        self.hd5_UT_hd2_relu = nn.ReLU(inplace=True)

        # fusion(h1_PT_hd2, h2_Cat_hd2, hd3_UT_hd2, hd4_UT_hd2, hd5_UT_hd2)
        self.conv2d_1 = nn.Conv2d(self.UpChannels, self.UpChannels, 3, padding=1)  # 16
        self.bn2d_1 = nn.BatchNorm2d(self.UpChannels)
        self.relu2d_1 = nn.ReLU(inplace=True)

        '''stage 1d'''
        # h1->320*320, hd1->320*320, Concatenation
        self.h1_Cat_hd1_conv = nn.Conv2d(filters[0], self.CatChannels, 3, padding=1)
        self.h1_Cat_hd1_bn = nn.BatchNorm2d(self.CatChannels)
        self.h1_Cat_hd1_relu = nn.ReLU(inplace=True)

        # hd2->160*160, hd1->320*320, Upsample 2 times
        self.hd2_UT_hd1 = nn.Upsample(scale_factor=2, mode='bilinear')  # 14*14
        self.hd2_UT_hd1_conv = nn.Conv2d(self.UpChannels, self.CatChannels, 3, padding=1)
        self.hd2_UT_hd1_bn = nn.BatchNorm2d(self.CatChannels)
        self.hd2_UT_hd1_relu = nn.ReLU(inplace=True)

        # hd3->80*80, hd1->320*320, Upsample 4 times
        self.hd3_UT_hd1 = nn.Upsample(scale_factor=4, mode='bilinear')  # 14*14
        self.hd3_UT_hd1_conv = nn.Conv2d(self.UpChannels, self.CatChannels, 3, padding=1)
        self.hd3_UT_hd1_bn = nn.BatchNorm2d(self.CatChannels)
        self.hd3_UT_hd1_relu = nn.ReLU(inplace=True)

        # hd4->40*40, hd1->320*320, Upsample 8 times
        self.hd4_UT_hd1 = nn.Upsample(scale_factor=8, mode='bilinear')  # 14*14
        self.hd4_UT_hd1_conv = nn.Conv2d(self.UpChannels, self.CatChannels, 3, padding=1)
        self.hd4_UT_hd1_bn = nn.BatchNorm2d(self.CatChannels)
        self.hd4_UT_hd1_relu = nn.ReLU(inplace=True)

        # hd5->20*20, hd1->320*320, Upsample 16 times
        self.hd5_UT_hd1 = nn.Upsample(scale_factor=16, mode='bilinear')  # 14*14
        self.hd5_UT_hd1_conv = nn.Conv2d(filters[4], self.CatChannels, 3, padding=1)
        self.hd5_UT_hd1_bn = nn.BatchNorm2d(self.CatChannels)
        self.hd5_UT_hd1_relu = nn.ReLU(inplace=True)

        # fusion(h1_Cat_hd1, hd2_UT_hd1, hd3_UT_hd1, hd4_UT_hd1, hd5_UT_hd1)
        self.conv1d_1 = nn.Conv2d(self.UpChannels, self.UpChannels, 3, padding=1)  # 16
        self.bn1d_1 = nn.BatchNorm2d(self.UpChannels)
        self.relu1d_1 = nn.ReLU(inplace=True)

        # -------------Bilinear Upsampling--------------
        self.upscore6 = nn.Upsample(scale_factor=32, mode='bilinear')  ###
        self.upscore5 = nn.Upsample(scale_factor=16, mode='bilinear')
        self.upscore4 = nn.Upsample(scale_factor=8, mode='bilinear')
        self.upscore3 = nn.Upsample(scale_factor=4, mode='bilinear')
        self.upscore2 = nn.Upsample(scale_factor=2, mode='bilinear')

        # DeepSup
        self.outconv1 = nn.Conv2d(self.UpChannels, n_classes, 3, padding=1)
        self.outconv2 = nn.Conv2d(self.UpChannels, n_classes, 3, padding=1)
        self.outconv3 = nn.Conv2d(self.UpChannels, n_classes, 3, padding=1)
        self.outconv4 = nn.Conv2d(self.UpChannels, n_classes, 3, padding=1)
        self.outconv5 = nn.Conv2d(filters[4], n_classes, 3, padding=1)

        # initialise weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init_weights(m, init_type='kaiming')
            elif isinstance(m, nn.BatchNorm2d):
                init_weights(m, init_type='kaiming')

    def forward(self, inputs,train=False):
        ## -------------Encoder-------------
        h1 = self.conv1(inputs)  # h1->320*320*64

        h2 = self.maxpool1(h1)
        h2 = self.conv2(h2)  # h2->160*160*128

        h3 = self.maxpool2(h2)
        h3 = self.conv3(h3)  # h3->80*80*256

        h4 = self.maxpool3(h3)
        h4 = self.conv4(h4)  # h4->40*40*512

        h5 = self.maxpool4(h4)
        hd5 = self.conv5(h5)  # h5->20*20*1024

        ## -------------Decoder-------------
        h1_PT_hd4 = self.h1_PT_hd4_relu(self.h1_PT_hd4_bn(self.h1_PT_hd4_conv(self.h1_PT_hd4(h1))))
        h2_PT_hd4 = self.h2_PT_hd4_relu(self.h2_PT_hd4_bn(self.h2_PT_hd4_conv(self.h2_PT_hd4(h2))))
        h3_PT_hd4 = self.h3_PT_hd4_relu(self.h3_PT_hd4_bn(self.h3_PT_hd4_conv(self.h3_PT_hd4(h3))))
        h4_Cat_hd4 = self.h4_Cat_hd4_relu(self.h4_Cat_hd4_bn(self.h4_Cat_hd4_conv(h4)))
        hd5_UT_hd4 = self.hd5_UT_hd4_relu(self.hd5_UT_hd4_bn(self.hd5_UT_hd4_conv(self.hd5_UT_hd4(hd5))))
        hd4 = self.relu4d_1(self.bn4d_1(self.conv4d_1(
            torch.cat((h1_PT_hd4, h2_PT_hd4, h3_PT_hd4, h4_Cat_hd4, hd5_UT_hd4), 1))))  # hd4->40*40*UpChannels
        # print(h1_PT_hd4.size(), h2_PT_hd4.size(), h3_PT_hd4.size(), h4_Cat_hd4.size(), hd5_UT_hd4.size())
        h1_PT_hd3 = self.h1_PT_hd3_relu(self.h1_PT_hd3_bn(self.h1_PT_hd3_conv(self.h1_PT_hd3(h1))))
        h2_PT_hd3 = self.h2_PT_hd3_relu(self.h2_PT_hd3_bn(self.h2_PT_hd3_conv(self.h2_PT_hd3(h2))))
        h3_Cat_hd3 = self.h3_Cat_hd3_relu(self.h3_Cat_hd3_bn(self.h3_Cat_hd3_conv(h3)))
        hd4_UT_hd3 = self.hd4_UT_hd3_relu(self.hd4_UT_hd3_bn(self.hd4_UT_hd3_conv(self.hd4_UT_hd3(hd4))))
        hd5_UT_hd3 = self.hd5_UT_hd3_relu(self.hd5_UT_hd3_bn(self.hd5_UT_hd3_conv(self.hd5_UT_hd3(hd5))))
        hd3 = self.relu3d_1(self.bn3d_1(self.conv3d_1(
            torch.cat((h1_PT_hd3, h2_PT_hd3, h3_Cat_hd3, hd4_UT_hd3, hd5_UT_hd3), 1))))  # hd3->80*80*UpChannels
        # print(h1_PT_hd3.size(), h2_PT_hd3.size(), h3_Cat_hd3.size(), hd4_UT_hd3.size(), hd5_UT_hd3.size())
        h1_PT_hd2 = self.h1_PT_hd2_relu(self.h1_PT_hd2_bn(self.h1_PT_hd2_conv(self.h1_PT_hd2(h1))))
        h2_Cat_hd2 = self.h2_Cat_hd2_relu(self.h2_Cat_hd2_bn(self.h2_Cat_hd2_conv(h2)))
        hd3_UT_hd2 = self.hd3_UT_hd2_relu(self.hd3_UT_hd2_bn(self.hd3_UT_hd2_conv(self.hd3_UT_hd2(hd3))))
        hd4_UT_hd2 = self.hd4_UT_hd2_relu(self.hd4_UT_hd2_bn(self.hd4_UT_hd2_conv(self.hd4_UT_hd2(hd4))))
        hd5_UT_hd2 = self.hd5_UT_hd2_relu(self.hd5_UT_hd2_bn(self.hd5_UT_hd2_conv(self.hd5_UT_hd2(hd5))))
        hd2 = self.relu2d_1(self.bn2d_1(self.conv2d_1(
            torch.cat((h1_PT_hd2, h2_Cat_hd2, hd3_UT_hd2, hd4_UT_hd2, hd5_UT_hd2), 1))))  # hd2->160*160*UpChannels

        h1_Cat_hd1 = self.h1_Cat_hd1_relu(self.h1_Cat_hd1_bn(self.h1_Cat_hd1_conv(h1)))
        hd2_UT_hd1 = self.hd2_UT_hd1_relu(self.hd2_UT_hd1_bn(self.hd2_UT_hd1_conv(self.hd2_UT_hd1(hd2))))
        hd3_UT_hd1 = self.hd3_UT_hd1_relu(self.hd3_UT_hd1_bn(self.hd3_UT_hd1_conv(self.hd3_UT_hd1(hd3))))
        hd4_UT_hd1 = self.hd4_UT_hd1_relu(self.hd4_UT_hd1_bn(self.hd4_UT_hd1_conv(self.hd4_UT_hd1(hd4))))
        hd5_UT_hd1 = self.hd5_UT_hd1_relu(self.hd5_UT_hd1_bn(self.hd5_UT_hd1_conv(self.hd5_UT_hd1(hd5))))
        # print(h1_Cat_hd1.size(),hd2_UT_hd1.size(),hd3_UT_hd1.size(),hd4_UT_hd1.size(),hd5_UT_hd1.size())
        hd1 = self.relu1d_1(self.bn1d_1(self.conv1d_1(
            torch.cat((h1_Cat_hd1, hd2_UT_hd1, hd3_UT_hd1, hd4_UT_hd1, hd5_UT_hd1), 1))))  # hd1->320*320*UpChannels

        # d5 = self.outconv5(hd5)
        # d5 = self.upscore5(d5)  # 16->256

        # d4 = self.outconv4(hd4)
        # d4 = self.upscore4(d4)  # 32->256

        # d3 = self.outconv3(hd3)
        # d3 = self.upscore3(d3)  # 64->256

        # d2 = self.outconv2(hd2)
        # d2 = self.upscore2(d2)  # 128->256

        # d1 = self.outconv1(hd1)  # 256
        # print(d5.size(),d4.size(),d3.size(),d2.size(),d1.size())
        if train:
            return hd5,hd4,hd3,hd2,hd1
        return hd1
        # if train:
        #     return hd5,hd4,hd3,hd2,hd1
        # return hd1


if __name__=='__main__':
    net=UNet_3Plus_DeepSup(3,1,filters=[8,16,24,36,42]).cuda()
    summary(net,(3,32,32))

