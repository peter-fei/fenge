import torch
import torch.nn as nn
from torch.nn import Module
from torch.nn.modules.conv import _ConvNd
from torch.nn.modules.utils import _pair
from net.init_weights import init_weights
from torchsummary import summary
from torch.nn import functional as F

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

class Gate1(Module):
    def __init__(self,inchannels,outchannels):
        super(Gate1, self).__init__()
        self.conv1=nn.Sequential(
            nn.Conv2d(1,inchannels,3,1,1),
            nn.BatchNorm2d(inchannels),
            nn.ReLU(True)
        )
        self.sigmoid=nn.Sigmoid()
        self.conv2=nn.Sequential(
            nn.Conv2d(inchannels,outchannels,3,1,1),
            nn.BatchNorm2d(outchannels),
            nn.ReLU(True)
        )
    def forward(self,x,y):
        y=self.conv1(y)
        z=self.sigmoid(y)
        out=y*(1-z)+x
        out=self.conv2(out)
        return out

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



class Down_Conv(Module):
    def __init__(self,inchannels,out_channels):
        super(Down_Conv, self).__init__()
        self.conv1=nn.Sequential(
            nn.Conv2d(inchannels,inchannels,3,1,1),
            nn.BatchNorm2d(inchannels),
            nn.ReLU(True)
        )
        self.conv2=nn.Sequential(
            nn.Conv2d(inchannels,out_channels,3,1,1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(True)
        )
    def forward(self,x):
        x=self.conv1(x)
        y=self.conv2(x)
        return y

class UP_Conv(Module):
    def __init__(self,inchannels,outchannels,scal=2):
        super(UP_Conv, self).__init__()
        self.conv1=nn.Sequential(
            nn.Conv2d(inchannels,inchannels,3,1,1),
            nn.BatchNorm2d(inchannels),
            nn.ReLU(True)
        )
        self.up=nn.Upsample(scale_factor=scal,mode='bilinear')
        self.conv2=nn.Sequential(
            nn.Conv2d(inchannels,outchannels,3,1,1),
            nn.BatchNorm2d(outchannels),
            nn.ReLU(True)
        )
    def forward(self,x):
        x=self.conv1(x)
        x=self.up(x)
        y=self.conv2(x)
        return y



class Pool_Conv(Module):
    def __init__(self,inchannels,outchannels,stride):
        super(Pool_Conv, self).__init__()
        self.conv1=nn.Sequential(
            nn.Conv2d(inchannels,inchannels,3,1,1),
            nn.BatchNorm2d(inchannels),
            nn.ReLU(True)
        )
        self.pool=nn.MaxPool2d(stride,stride)
        self.conv2=nn.Sequential(
            nn.Conv2d(inchannels,outchannels,3,1,1),
            nn.BatchNorm2d(outchannels),
            nn.ReLU(True)
        )
    def forward(self,x):
        x=self.conv1(x)
        x=self.pool(x)
        y=self.conv2(x)
        return y

class Up_Conv_scal(Module):
    def __init__(self,inchannels,outchannels,stride):
        super(Up_Conv_scal, self).__init__()
        self.conv1=nn.Sequential(
            nn.Conv2d(inchannels,inchannels,3,1,1),
            nn.BatchNorm2d(inchannels),
            nn.ReLU(True)
        )
        self.up=nn.Upsample(scale_factor=stride,mode='bilinear')
        self.conv2=nn.Sequential(
            nn.Conv2d(inchannels,outchannels,3,1,1),
            nn.BatchNorm2d(outchannels),
            nn.ReLU(True)
        )
    def forward(self,x):
        x=self.conv1(x)
        x=self.up(x)
        y=self.conv2(x)
        return y

class Edge_Conv(Module):
    def __init__(self,in_channels,out_channels,nums):
        super(Edge_Conv, self).__init__()
        convs = []
        for i in range(1, nums + 1):
            conv = nn.Sequential(
                nn.MaxPool2d(2 ** i, 2 ** i),
                nn.Conv2d(in_channels, in_channels, 3, 1, 1),
                nn.BatchNorm2d(in_channels),
                nn.Upsample(scale_factor=2 ** i, mode='bilinear')
            )
            convs.append(conv)
            self.convs = nn.ModuleList(convs)
            self.out_conv = nn.Sequential(
                nn.Conv2d(in_channels * (nums), out_channels, 3, 1, 1)
            )

    def forward(self, x):
        out = []
        for conv in self.convs:
            out.append(x - conv(x))
        out = torch.cat((out), dim=1)
        # out=torch.cat((out,x),dim=1)
        out = self.out_conv(out)
        return out


class Edge_Conv2(Module):
    def __init__(self,in_channels,out_channels,nums):
        super(Edge_Conv2, self).__init__()
        down_convs = []
        up_convs = []
        for i in range(1, nums + 1):
            conv = nn.Sequential(
                nn.MaxPool2d(2 ** i, 2 ** i),
                nn.Conv2d(in_channels, in_channels, 3, 1, 1),
                # nn.BatchNorm2d(in_channels),
            )
            up_conv=nn.Sequential(
                nn.Upsample(scale_factor=2 ** i, mode='bilinear'),
                nn.Conv2d(in_channels, in_channels, 3, 1, 1),
                nn.BatchNorm2d(in_channels),
                nn.ReLU(True)
            )
            down_convs.append(conv)
            up_convs.append(up_conv)
            self.down_convs = nn.ModuleList(down_convs)
            self.up_convs=nn.ModuleList(up_convs)
            self.out_conv = nn.Sequential(
                nn.Conv2d(in_channels * (nums), out_channels, 3, 1, 1)
            )

    def forward(self, x,ys=None):
        out = []
        # print(len(self.down_convs))
        for i in range(len(self.down_convs)):
            res=self.down_convs[i](x)-ys[i]
            res=self.up_convs[i](res)
            out.append(res)
        out = torch.cat((out), dim=1)
        # out=torch.cat((out,x),dim=1)
        out = self.out_conv(out)
        return out


class Hd_Conv(Module):
    def __init__(self,inchannels,outchannels):
        super(Hd_Conv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(inchannels, outchannels, 3, 1, 1),
            nn.BatchNorm2d(outchannels),
            nn.ReLU(True)
        )
    def forward(self,x):
        x=self.conv(x)
        return x

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

# class Att(Module)
class M1(Module):
    def __init__(self,inchannels,out_channels,filters=(64,128,256,512,1024)):
        super(M1, self).__init__()
        cat_channel=filters[0]
        self.down_conv1=Down_Conv(inchannels,filters[0])
        self.down_conv2=Down_Conv(filters[0],filters[1])
        self.down_conv3=Down_Conv(filters[1],filters[2])
        self.down_conv4=Down_Conv(filters[2],filters[3])
        self.down_conv5=Down_Conv(filters[3],filters[4])
        self.maxpool2=nn.MaxPool2d(2,2)
        self.maxpool4=nn.MaxPool2d(4,4)
        self.maxpool8=nn.MaxPool2d(8,8)
        self.maxpool16=nn.MaxPool2d(16,16)

        self.pool_conv1_5=Pool_Conv(filters[0], cat_channel, 16)
        self.pool_conv2_5=Pool_Conv(filters[1], cat_channel, 8)
        self.pool_conv3_5=Pool_Conv(filters[2], cat_channel, 4)
        self.pool_conv4_5=Pool_Conv(filters[3], cat_channel, 2)
        self.pool_conv5_5=Pool_Conv(filters[4], cat_channel, 1)
        self.pool_conv1_4=Pool_Conv(filters[0],cat_channel,8)
        self.pool_conv2_4=Pool_Conv(filters[1],cat_channel,4)
        self.pool_conv3_4=Pool_Conv(filters[2],cat_channel,2)
        self.pool_conv4_4=Pool_Conv(filters[3],cat_channel,1)
        self.up_conv5_4_scal=Up_Conv_scal(filters[4],cat_channel,2)
        self.pool_conv1_3=Pool_Conv(filters[0],cat_channel,4)
        self.pool_conv2_3=Pool_Conv(filters[1],cat_channel,2)
        self.pool_conv3_3=Pool_Conv(filters[2],cat_channel,1)
        self.up_conv4_3_scal=Up_Conv_scal(filters[3],cat_channel,2)
        self.up_conv5_3_scal=Up_Conv_scal(filters[4],cat_channel,4)
        self.pool_conv1_2=Pool_Conv(filters[0],cat_channel,2)
        self.pool_conv2_2=Pool_Conv(filters[1],cat_channel,1)
        self.up_conv3_2_scal=Up_Conv_scal(filters[2],cat_channel,2)
        self.up_conv4_2_scal=Up_Conv_scal(filters[3],cat_channel,4)
        self.up_conv5_2_scal=Up_Conv_scal(filters[4],cat_channel,8)
        self.pool_conv1_1=Pool_Conv(filters[0],cat_channel,1)
        self.up_conv2_1_scal=Up_Conv_scal(filters[1],cat_channel,2)
        self.up_conv3_1_scal=Up_Conv_scal(filters[2],cat_channel,4)
        self.up_conv4_1_scal=Up_Conv_scal(filters[3],cat_channel,8)
        self.up_conv5_1_scal=Up_Conv_scal(filters[4],cat_channel,16)

        self.hdconv5=Hd_Conv(cat_channel*5,cat_channel)
        self.hdconv4=Hd_Conv(cat_channel*4,cat_channel)
        self.hdconv3=Hd_Conv(cat_channel*3,cat_channel)
        self.hdconv2=Hd_Conv(cat_channel*2,cat_channel)
        self.hdconv1=Hd_Conv(cat_channel,cat_channel)

        self.edge4=Edge_Conv(cat_channel,1,1)
        self.edge3=Edge_Conv(cat_channel,1,2)
        self.edge2=Edge_Conv(cat_channel,1,3)
        # self.edge1=Edge_Conv(cat_channel,1,4)

        self.up_conv5_4=UP_Conv(cat_channel*5,cat_channel)
        self.up_conv4_3=UP_Conv(cat_channel,cat_channel)
        self.up_conv3_2=UP_Conv(cat_channel,cat_channel)
        self.up_conv2_1=UP_Conv(cat_channel,cat_channel)

        self.gate4=Gate(cat_channel,cat_channel)
        self.gate3=Gate(cat_channel,cat_channel)
        self.gate2=Gate(cat_channel,cat_channel)
        self.gate1=Gate(cat_channel,cat_channel)
        self.edge_up2=nn.Sequential(
            nn.Upsample(scale_factor=2,mode='bilinear'),
            nn.Conv2d(1,1,3,1,1)
        )
        self.edge_up3=nn.Sequential(
            nn.Upsample(scale_factor=4,mode='bilinear'),
            nn.Conv2d(1,1,3,1,1)
        )
        self.edge_up4=nn.Sequential(
            nn.Upsample(scale_factor=8,mode='bilinear'),
            nn.Conv2d(1,1,3,1,1),

        )
        # self.att1=Att()
        self.out=nn.Sequential(
            nn.Conv2d(cat_channel*2,cat_channel,3,1,1),
            nn.BatchNorm2d(cat_channel),
            nn.ReLU(True),
            nn.Conv2d(cat_channel,out_channels,1,1,0),
        )

        self.gate_up2 = nn.Sequential(
            nn.Conv2d(cat_channel, out_channels, 3, 1, 1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(True),
            nn.Upsample(scale_factor=2, mode='bilinear'),
            nn.Conv2d(out_channels, out_channels, 1, 1, 0)
        )
        self.gate_up3 = nn.Sequential(
            nn.Conv2d(cat_channel, out_channels, 3, 1, 1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(True),
            nn.Upsample(scale_factor=4, mode='bilinear'),
            nn.Conv2d(out_channels, out_channels, 1, 1, 0)
        )
        self.gate_up4 = nn.Sequential(
            nn.Conv2d(cat_channel, out_channels, 3, 1, 1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(True),
            nn.Upsample(scale_factor=8, mode='bilinear'),
            nn.Conv2d(out_channels, out_channels, 1, 1, 0)
        )

        self.conv1_d1 = nn.Sequential(
            nn.Conv2d(cat_channel, cat_channel, 3, 1, 1),
            nn.BatchNorm2d(cat_channel),
            # nn.ReLU(True)
        )
        self.conv1_d2 = nn.Sequential(
            # nn.Upsample(scale_factor=2,mode='bilinear'),
            nn.Conv2d(cat_channel, cat_channel, 3, 1, 4, dilation=4),
            nn.BatchNorm2d(cat_channel),
            # nn.ReLU(True)
        )
        self.convd2 = nn.Sequential(
            nn.Conv2d(cat_channel * 2, cat_channel, 3, 1, 1),
            nn.BatchNorm2d(cat_channel),
            # nn.ReLU(True)
        )
        self.conv1_d3 = nn.Sequential(
            # nn.Upsample(scale_factor=4,mode='bilinear'),
            nn.Conv2d(cat_channel, cat_channel, 3, 1, 8, dilation=8),
            nn.BatchNorm2d(cat_channel),
            nn.ReLU(True)
        )
        self.convd3 = nn.Sequential(
            nn.Conv2d(cat_channel * 2, cat_channel, 3, 1, 1),
            nn.BatchNorm2d(cat_channel),
            # nn.ReLU(True)
        )
        self.conv1_d4 = nn.Sequential(
            # nn.Upsample(scale_factor=8,mode='bilinear'),
            nn.Conv2d(cat_channel, cat_channel, 3, 1, 12, dilation=12),
            nn.BatchNorm2d(cat_channel),
            nn.ReLU(True)
        )
        self.convd4 = nn.Sequential(
            nn.Conv2d(cat_channel * 2, cat_channel, 3, 1, 1),
            nn.BatchNorm2d(cat_channel),
            # nn.ReLU(True)
        )
        self.conv1_d5 = nn.Sequential(
            # nn.Upsample(scale_factor=16,mode='bilinear'),
            nn.Conv2d(cat_channel, cat_channel, 3, 1, 16, dilation=16),
            nn.BatchNorm2d(cat_channel),
            nn.ReLU(True)
        )
        self.convd5 = nn.Sequential(
            nn.Conv2d(cat_channel , cat_channel, 3, 1, 1),
            nn.BatchNorm2d(cat_channel),
            # nn.ReLU(True)
        )
        self.relu=nn.ReLU(True)
        self.up_2 = nn.Upsample(scale_factor=2, mode='bilinear')
        self.up_4 = nn.Upsample(scale_factor=4, mode='bilinear')
        self.up_8 = nn.Upsample(scale_factor=8, mode='bilinear')
        self.up_16 = nn.Upsample(scale_factor=16, mode='bilinear')
        self.aspp=ASPP(filters[4],cat_channel)

    def forward(self,x,train=False):
        x1=self.down_conv1(x)
        p1=self.maxpool2(x1)
        x2=self.down_conv2(p1)
        p2=self.maxpool2(x2)
        x3=self.down_conv3(p2)
        p3=self.maxpool2(x3)
        x4=self.down_conv4(p3)
        p4=self.maxpool2(x4)
        x5=self.down_conv5(p4)

        h1_4=self.pool_conv1_4(x1)
        h2_4=self.pool_conv2_4(x2)
        h3_4=self.pool_conv3_4(x3)
        h4_4=self.pool_conv4_4(x4)
        # h5_4=self.up_conv5_4_scal(x5)
        hd4=torch.cat((h1_4,h2_4,h3_4,h4_4),dim=1)
        # hd4 = torch.cat((h1_4,h2_4,h3_4, h4_4,h5_4), dim=1)
        hd4=self.hdconv4(hd4)
        edge4=self.edge4(hd4)

        h1_3=self.pool_conv1_3(x1)
        h2_3=self.pool_conv2_3(x2)
        h3_3=self.pool_conv3_3(x3)
        # h4_3=self.up_conv4_3_scal(x4)
        # h5_3=self.up_conv5_3_scal(x5)
        hd3=torch.cat((h1_3,h2_3,h3_3),dim=1)
        hd3=self.hdconv3(hd3)
        edge3=self.edge3(hd3)

        h1_2=self.pool_conv1_2(x1)
        h2_2=self.pool_conv2_2(x2)
        # h3_2=self.up_conv3_2_scal(x3)
        # h4_2=self.up_conv4_2_scal(x4)
        # h5_2=self.up_conv5_2_scal(x5)
        # hd2=torch.cat((h2_2,h3_2,h4_2,h5_2),dim=1)
        hd2 = torch.cat((h1_2,h2_2), dim=1)
        hd2=self.hdconv2(hd2)
        edge2=self.edge2(hd2)
        h1_1=self.pool_conv1_1(x1)
        hd1=self.hdconv1(h1_1)
        # edge1=self.edge1(hd1)

        h1_5=self.pool_conv1_5(x1)
        h2_5=self.pool_conv2_5(x2)
        h3_5=self.pool_conv3_5(x3)
        h4_5=self.pool_conv4_5(x4)
        h5_5=self.pool_conv5_5(x5)
        #
        # print(h1_5.size(),h2_5.size(),h3_5.size(),h4_5.size(),h5_5.size())
        # hd5=self.aspp(x5)
        hd5=torch.cat((h1_5,h2_5,h3_5,h4_5,h5_5),dim=1)
        d5_4=self.up_conv5_4(hd5)
        gate4=self.gate4(d5_4,edge4)
        d4_3=self.up_conv4_3(gate4)
        gate3=self.gate3(d4_3,edge3)
        d3_2=self.up_conv3_2(gate3)
        gate2=self.gate2(d3_2,edge2)
        d2_1=self.up_conv2_1(gate2)
        # gate1=self.gate1(d2_1,edge1)
        # out=torch.cat((gate1,hd1),dim=1)

        edge2=self.gate_up2(gate2)
        edge3=self.gate_up3(gate3)
        edge4=self.gate_up4(gate4)
        #
        a1_5=self.conv1_d5(hd1)
        a1_4=self.conv1_d4(hd1)
        a1_3=self.conv1_d3(hd1)
        a1_2=self.conv1_d2(hd1)
        # print(hd5.size())
        a1_5=self.convd5(a1_5)
        a1_4=self.convd4(torch.cat((a1_4,self.up_8(hd4)),dim=1))
        a1_3=self.convd3(torch.cat((a1_3,self.up_4(hd3)),dim=1))
        a1_2=self.convd2(torch.cat((a1_2,self.up_2(hd2)),dim=1))
        a1=self.relu(a1_4+a1_3+a1_2+hd1+a1_5)
        out=torch.cat((a1,d2_1),dim=1)
        out = self.out(out)


        if train:
            return out,(edge2,edge3,edge4)
        return out





class M2(Module):
    def __init__(self,inchannels,out_channels,filters=(64,128,256,512,1024)):
        super(M2, self).__init__()
        cat_channel=filters[0]
        self.down_conv1=Down_Conv(inchannels,filters[0])
        self.down_conv2=Down_Conv(filters[0],filters[1])
        self.down_conv3=Down_Conv(filters[1],filters[2])
        self.down_conv4=Down_Conv(filters[2],filters[3])
        self.down_conv5=Down_Conv(filters[3],filters[4])
        self.maxpool2=nn.MaxPool2d(2,2)
        self.maxpool4=nn.MaxPool2d(4,4)
        self.maxpool8=nn.MaxPool2d(8,8)
        self.maxpool16=nn.MaxPool2d(16,16)

        self.pool_conv1_5=Pool_Conv(filters[0], cat_channel, 16)
        self.pool_conv2_5=Pool_Conv(filters[1], cat_channel, 8)
        self.pool_conv3_5=Pool_Conv(filters[2], cat_channel, 4)
        self.pool_conv4_5=Pool_Conv(filters[3], cat_channel, 2)
        self.pool_conv5_5=Pool_Conv(filters[4], cat_channel, 1)
        self.pool_conv1_4=Pool_Conv(filters[0],cat_channel,8)
        self.pool_conv2_4=Pool_Conv(filters[1],cat_channel,4)
        self.pool_conv3_4=Pool_Conv(filters[2],cat_channel,2)
        self.pool_conv4_4=Pool_Conv(filters[3],cat_channel,1)
        self.up_conv5_4_scal=Up_Conv_scal(filters[4],cat_channel,2)
        self.pool_conv1_3=Pool_Conv(filters[0],cat_channel,4)
        self.pool_conv2_3=Pool_Conv(filters[1],cat_channel,2)
        self.pool_conv3_3=Pool_Conv(filters[2],cat_channel,1)
        self.up_conv4_3_scal=Up_Conv_scal(filters[3],cat_channel,2)
        self.up_conv5_3_scal=Up_Conv_scal(filters[4],cat_channel,4)
        self.pool_conv1_2=Pool_Conv(filters[0],cat_channel,2)
        self.pool_conv2_2=Pool_Conv(filters[1],cat_channel,1)
        self.up_conv3_2_scal=Up_Conv_scal(filters[2],cat_channel,2)
        self.up_conv4_2_scal=Up_Conv_scal(filters[3],cat_channel,4)
        self.up_conv5_2_scal=Up_Conv_scal(filters[4],cat_channel,8)
        self.pool_conv1_1=Pool_Conv(filters[0],cat_channel,1)
        self.up_conv2_1_scal=Up_Conv_scal(filters[1],cat_channel,2)
        self.up_conv3_1_scal=Up_Conv_scal(filters[2],cat_channel,4)
        self.up_conv4_1_scal=Up_Conv_scal(filters[3],cat_channel,8)
        self.up_conv5_1_scal=Up_Conv_scal(filters[4],cat_channel,16)

        self.hdconv5=Hd_Conv(cat_channel*5,cat_channel)
        self.hdconv4=Hd_Conv(cat_channel*4,cat_channel)
        self.hdconv3=Hd_Conv(cat_channel*3,cat_channel)
        self.hdconv2=Hd_Conv(cat_channel*2,cat_channel)
        self.hdconv1=Hd_Conv(cat_channel,cat_channel)

        self.edge4=Edge_Conv(cat_channel,1,1)
        self.edge3=Edge_Conv(cat_channel,1,2)
        self.edge2=Edge_Conv(cat_channel,1,3)
        # self.edge1=Edge_Conv(cat_channel,1,4)

        self.up_conv5_4=UP_Conv(cat_channel*5,cat_channel)
        self.up_conv4_3=UP_Conv(cat_channel,cat_channel)
        self.up_conv3_2=UP_Conv(cat_channel,cat_channel)
        self.up_conv2_1=UP_Conv(cat_channel,cat_channel)

        self.gate4=Gate(cat_channel,cat_channel)
        self.gate3=Gate(cat_channel,cat_channel)
        self.gate2=Gate(cat_channel,cat_channel)
        self.gate1=Gate(cat_channel,cat_channel)
        self.edge_up2=nn.Sequential(
            nn.Upsample(scale_factor=2,mode='bilinear'),
            nn.Conv2d(1,1,3,1,1)
        )
        self.edge_up3=nn.Sequential(
            nn.Upsample(scale_factor=4,mode='bilinear'),
            nn.Conv2d(1,1,3,1,1)
        )
        self.edge_up4=nn.Sequential(
            nn.Upsample(scale_factor=8,mode='bilinear'),
            nn.Conv2d(1,1,3,1,1),

        )
        # self.att1=Att()
        self.out=nn.Sequential(
            nn.Conv2d(cat_channel,cat_channel,3,1,1),
            nn.BatchNorm2d(cat_channel),
            nn.ReLU(True),
            nn.Conv2d(cat_channel,out_channels,1,1,0),
        )

        self.gate_up2 = nn.Sequential(
            nn.Conv2d(cat_channel, out_channels, 3, 1, 1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(True),
            nn.Upsample(scale_factor=2, mode='bilinear'),
            nn.Conv2d(out_channels, out_channels, 1, 1, 0)
        )
        self.gate_up3 = nn.Sequential(
            nn.Conv2d(cat_channel, out_channels, 3, 1, 1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(True),
            nn.Upsample(scale_factor=4, mode='bilinear'),
            nn.Conv2d(out_channels, out_channels, 1, 1, 0)
        )
        self.gate_up4 = nn.Sequential(
            nn.Conv2d(cat_channel, out_channels, 3, 1, 1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(True),
            nn.Upsample(scale_factor=8, mode='bilinear'),
            nn.Conv2d(out_channels, out_channels, 1, 1, 0)
        )

        self.conv1_d1 = nn.Sequential(
            nn.Conv2d(cat_channel, cat_channel, 3, 1, 1),
            nn.BatchNorm2d(cat_channel),
            # nn.ReLU(True)
        )
        self.conv1_d2 = nn.Sequential(
            # nn.Upsample(scale_factor=2,mode='bilinear'),
            nn.Conv2d(cat_channel, cat_channel, 3, 1, 4, dilation=4),
            nn.BatchNorm2d(cat_channel),
            # nn.ReLU(True)
        )
        self.convd2 = nn.Sequential(
            nn.Conv2d(cat_channel * 2, cat_channel, 3, 1, 1),
            nn.BatchNorm2d(cat_channel),
            # nn.ReLU(True)
        )
        self.conv1_d3 = nn.Sequential(
            # nn.Upsample(scale_factor=4,mode='bilinear'),
            nn.Conv2d(cat_channel, cat_channel, 3, 1, 8, dilation=8),
            nn.BatchNorm2d(cat_channel),
            nn.ReLU(True)
        )
        self.convd3 = nn.Sequential(
            nn.Conv2d(cat_channel * 2, cat_channel, 3, 1, 1),
            nn.BatchNorm2d(cat_channel),
            # nn.ReLU(True)
        )
        self.conv1_d4 = nn.Sequential(
            # nn.Upsample(scale_factor=8,mode='bilinear'),
            nn.Conv2d(cat_channel, cat_channel, 3, 1, 12, dilation=12),
            nn.BatchNorm2d(cat_channel),
            nn.ReLU(True)
        )
        self.convd4 = nn.Sequential(
            nn.Conv2d(cat_channel * 2, cat_channel, 3, 1, 1),
            nn.BatchNorm2d(cat_channel),
            # nn.ReLU(True)
        )
        self.conv1_d5 = nn.Sequential(
            # nn.Upsample(scale_factor=16,mode='bilinear'),
            nn.Conv2d(cat_channel, cat_channel, 3, 1, 16, dilation=16),
            nn.BatchNorm2d(cat_channel),
            nn.ReLU(True)
        )
        self.convd5 = nn.Sequential(
            nn.Conv2d(cat_channel , cat_channel, 3, 1, 1),
            nn.BatchNorm2d(cat_channel),
            # nn.ReLU(True)
        )
        self.relu=nn.ReLU(True)
        self.up_2 = nn.Upsample(scale_factor=2, mode='bilinear')
        self.up_4 = nn.Upsample(scale_factor=4, mode='bilinear')
        self.up_8 = nn.Upsample(scale_factor=8, mode='bilinear')
        self.up_16 = nn.Upsample(scale_factor=16, mode='bilinear')
        self.aspp = ASPP(filters[4], cat_channel)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init_weights(m, init_type='kaiming')
            elif isinstance(m, nn.BatchNorm2d):
                init_weights(m, init_type='kaiming')

    def forward(self,x,train=False):
        x1=self.down_conv1(x)
        p1=self.maxpool2(x1)
        x2=self.down_conv2(p1)
        p2=self.maxpool2(x2)
        x3=self.down_conv3(p2)
        p3=self.maxpool2(x3)
        x4=self.down_conv4(p3)
        p4=self.maxpool2(x4)
        x5=self.down_conv5(p4)

        h1_4=self.pool_conv1_4(x1)
        h2_4=self.pool_conv2_4(x2)
        h3_4=self.pool_conv3_4(x3)
        h4_4=self.pool_conv4_4(x4)
        # h5_4=self.up_conv5_4_scal(x5)
        hd4=torch.cat((h1_4,h2_4,h3_4,h4_4),dim=1)
        # hd4 = torch.cat((h1_4,h2_4,h3_4, h4_4,h5_4), dim=1)
        hd4=self.hdconv4(hd4)
        edge4=self.edge4(hd4)

        h1_3=self.pool_conv1_3(x1)
        h2_3=self.pool_conv2_3(x2)
        h3_3=self.pool_conv3_3(x3)
        # h4_3=self.up_conv4_3_scal(x4)
        # h5_3=self.up_conv5_3_scal(x5)
        hd3=torch.cat((h1_3,h2_3,h3_3),dim=1)
        hd3=self.hdconv3(hd3)
        edge3=self.edge3(hd3)

        h1_2=self.pool_conv1_2(x1)
        h2_2=self.pool_conv2_2(x2)
        # h3_2=self.up_conv3_2_scal(x3)
        # h4_2=self.up_conv4_2_scal(x4)
        # h5_2=self.up_conv5_2_scal(x5)
        # hd2=torch.cat((h2_2,h3_2,h4_2,h5_2),dim=1)
        hd2 = torch.cat((h1_2,h2_2), dim=1)
        hd2=self.hdconv2(hd2)
        edge2=self.edge2(hd2)
        h1_1=self.pool_conv1_1(x1)
        hd1=self.hdconv1(h1_1)
        # edge1=self.edge1(hd1)

        h1_5=self.pool_conv1_5(x1)
        h2_5=self.pool_conv2_5(x2)
        h3_5=self.pool_conv3_5(x3)
        h4_5=self.pool_conv4_5(x4)
        h5_5=self.pool_conv5_5(x5)
        #
        # print(h1_5.size(),h2_5.size(),h3_5.size(),h4_5.size(),h5_5.size())
        # hd5=self.aspp(x5)
        hd5=torch.cat((h1_5,h2_5,h3_5,h4_5,h5_5),dim=1)
        d5_4=self.up_conv5_4(hd5)
        gate4=self.gate4(d5_4,edge4)
        d4_3=self.up_conv4_3(gate4)
        gate3=self.gate3(d4_3,edge3)
        d3_2=self.up_conv3_2(gate3)
        gate2=self.gate2(d3_2,edge2)
        # d2_1=self.up_conv2_1(gate2)
        # gate1=self.gate1(d2_1,edge1)
        # out=torch.cat((gate1,hd1),dim=1)

        edge2=self.gate_up2(gate2)
        edge3=self.gate_up3(gate3)
        edge4=self.gate_up4(gate4)
        #
        a1_5=self.conv1_d5(hd1)
        a1_4=self.conv1_d4(hd1)
        a1_3=self.conv1_d3(hd1)
        a1_2=self.conv1_d2(hd1)
        # print(hd5.size())
        a1_5=self.convd5(a1_5)
        a1_4=self.convd4(torch.cat((a1_4,self.up_8(gate4)),dim=1))
        a1_3=self.convd3(torch.cat((a1_3,self.up_4(gate3)),dim=1))
        a1_2=self.convd2(torch.cat((a1_2,self.up_2(gate2)),dim=1))
        a1=self.relu(a1_4+a1_3+a1_2+hd1+a1_5)
        # out=torch.cat((a1,d2_1),dim=1)
        out = self.out(a1)


        if train:
            return out,(edge2,edge3,edge4)
        return out



class M3(Module):
    def __init__(self,inchannels,out_channels,filters=(64,128,256,512,1024)):
        super(M3, self).__init__()
        cat_channel=filters[0]
        self.down_conv1=Down_Conv(inchannels,filters[0])
        self.down_conv2=Down_Conv(filters[0],filters[1])
        self.down_conv3=Down_Conv(filters[1],filters[2])
        self.down_conv4=Down_Conv(filters[2],filters[3])
        self.down_conv5=Down_Conv(filters[3],filters[4])
        self.maxpool2=nn.MaxPool2d(2,2)
        self.maxpool4=nn.MaxPool2d(4,4)
        self.maxpool8=nn.MaxPool2d(8,8)
        self.maxpool16=nn.MaxPool2d(16,16)

        self.pool_conv1_5=Pool_Conv(filters[0], cat_channel, 16)
        self.pool_conv2_5=Pool_Conv(filters[1], cat_channel, 8)
        self.pool_conv3_5=Pool_Conv(filters[2], cat_channel, 4)
        self.pool_conv4_5=Pool_Conv(filters[3], cat_channel, 2)
        self.pool_conv5_5=Pool_Conv(filters[4], cat_channel, 1)
        self.pool_conv1_4=Pool_Conv(filters[0],cat_channel,8)
        self.pool_conv2_4=Pool_Conv(filters[1],cat_channel,4)
        self.pool_conv3_4=Pool_Conv(filters[2],cat_channel,2)
        self.pool_conv4_4=Pool_Conv(filters[3],cat_channel,1)
        self.up_conv5_4_scal=Up_Conv_scal(filters[4],cat_channel,2)
        self.pool_conv1_3=Pool_Conv(filters[0],cat_channel,4)
        self.pool_conv2_3=Pool_Conv(filters[1],cat_channel,2)
        self.pool_conv3_3=Pool_Conv(filters[2],cat_channel,1)
        self.up_conv4_3_scal=Up_Conv_scal(filters[3],cat_channel,2)
        self.up_conv5_3_scal=Up_Conv_scal(filters[4],cat_channel,4)
        self.pool_conv1_2=Pool_Conv(filters[0],cat_channel,2)
        self.pool_conv2_2=Pool_Conv(filters[1],cat_channel,1)
        self.up_conv3_2_scal=Up_Conv_scal(filters[2],cat_channel,2)
        self.up_conv4_2_scal=Up_Conv_scal(filters[3],cat_channel,4)
        self.up_conv5_2_scal=Up_Conv_scal(filters[4],cat_channel,8)
        self.pool_conv1_1=Pool_Conv(filters[0],cat_channel,1)
        self.up_conv2_1_scal=Up_Conv_scal(filters[1],cat_channel,2)
        self.up_conv3_1_scal=Up_Conv_scal(filters[2],cat_channel,4)
        self.up_conv4_1_scal=Up_Conv_scal(filters[3],cat_channel,8)
        self.up_conv5_1_scal=Up_Conv_scal(filters[4],cat_channel,16)

        self.hdconv5=Hd_Conv(cat_channel*5,cat_channel)
        self.hdconv4=Hd_Conv(cat_channel*4,cat_channel)
        self.hdconv3=Hd_Conv(cat_channel*3,cat_channel)
        self.hdconv2=Hd_Conv(cat_channel*2,cat_channel)
        self.hdconv1=Hd_Conv(cat_channel,cat_channel)

        self.edge4=Edge_Conv2(cat_channel,1,1)
        self.edge3=Edge_Conv2(cat_channel,1,2)
        self.edge2=Edge_Conv2(cat_channel,1,3)
        # self.edge1=Edge_Conv(cat_channel,1,4)

        self.up_conv5_4=UP_Conv(cat_channel*5,cat_channel)
        self.up_conv4_3=UP_Conv(cat_channel,cat_channel)
        self.up_conv3_2=UP_Conv(cat_channel,cat_channel)
        self.up_conv2_1=UP_Conv(cat_channel,cat_channel)

        self.gate4=Gate(cat_channel,cat_channel)
        self.gate3=Gate(cat_channel,cat_channel)
        self.gate2=Gate(cat_channel,cat_channel)
        self.gate1=Gate(cat_channel,cat_channel)
        self.edge_up2=nn.Sequential(
            nn.Upsample(scale_factor=2,mode='bilinear'),
            nn.Conv2d(1,1,3,1,1)
        )
        self.edge_up3=nn.Sequential(
            nn.Upsample(scale_factor=4,mode='bilinear'),
            nn.Conv2d(1,1,3,1,1)
        )
        self.edge_up4=nn.Sequential(
            nn.Upsample(scale_factor=8,mode='bilinear'),
            nn.Conv2d(1,1,3,1,1),

        )

        self.out=nn.Sequential(
            nn.Conv2d(cat_channel,cat_channel,3,1,1),
            nn.BatchNorm2d(cat_channel),
            nn.ReLU(True),
            nn.Conv2d(cat_channel,out_channels,1,1,0),
        )

        self.gate_up2 = nn.Sequential(
            nn.Conv2d(cat_channel, out_channels, 3, 1, 1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(True),
            nn.Upsample(scale_factor=2, mode='bilinear'),
            nn.Conv2d(out_channels, out_channels, 1, 1, 0)
        )
        self.gate_up3 = nn.Sequential(
            nn.Conv2d(cat_channel, out_channels, 3, 1, 1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(True),
            nn.Upsample(scale_factor=4, mode='bilinear'),
            nn.Conv2d(out_channels, out_channels, 1, 1, 0)
        )
        self.gate_up4 = nn.Sequential(
            nn.Conv2d(cat_channel, out_channels, 3, 1, 1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(True),
            nn.Upsample(scale_factor=8, mode='bilinear'),
            nn.Conv2d(out_channels, out_channels, 1, 1, 0)
        )

        self.conv1_d1 = nn.Sequential(
            nn.Conv2d(cat_channel, cat_channel, 3, 1, 1),
            nn.BatchNorm2d(cat_channel),
            nn.ReLU(True)
        )
        self.convd1 = nn.Sequential(
            nn.Conv2d(cat_channel * 5, cat_channel*5, 3, 1, 1),
            nn.BatchNorm2d(cat_channel*5),
            nn.ReLU(True),
            nn.Conv2d(cat_channel *5, cat_channel, 1, 1, 0),
            nn.BatchNorm2d(cat_channel),
            nn.ReLU(True)
        )
        self.conv1_d2 = nn.Sequential(
            # nn.Upsample(scale_factor=2,mode='bilinear'),
            nn.Conv2d(cat_channel, cat_channel, 3, 1, 4, dilation=4),
            nn.BatchNorm2d(cat_channel),
            nn.ReLU(True)
        )
        self.convd2 = nn.Sequential(
            nn.Conv2d(cat_channel * 2, cat_channel, 3, 1, 1),
            nn.BatchNorm2d(cat_channel),
            # nn.ReLU(True)
        )
        self.conv1_d3 = nn.Sequential(
            # nn.Upsample(scale_factor=4,mode='bilinear'),
            nn.Conv2d(cat_channel, cat_channel, 3, 1, 8, dilation=8),
            nn.BatchNorm2d(cat_channel),
            nn.ReLU(True)
        )
        self.convd3 = nn.Sequential(
            nn.Conv2d(cat_channel * 2, cat_channel, 3, 1, 1),
            nn.BatchNorm2d(cat_channel),
            # nn.ReLU(True)
        )
        self.conv1_d4 = nn.Sequential(
            # nn.Upsample(scale_factor=8,mode='bilinear'),
            nn.Conv2d(cat_channel, cat_channel, 3, 1, 12, dilation=12),
            nn.BatchNorm2d(cat_channel),
            nn.ReLU(True)
        )
        self.convd4 = nn.Sequential(
            nn.Conv2d(cat_channel * 2, cat_channel, 3, 1, 1),
            nn.BatchNorm2d(cat_channel),
            # nn.ReLU(True)
        )
        self.conv1_d5 = nn.Sequential(
            # nn.Upsample(scale_factor=16,mode='bilinear'),
            nn.Conv2d(cat_channel, cat_channel, 3, 1, 16, dilation=16),
            nn.BatchNorm2d(cat_channel),
            nn.ReLU(True)
        )
        self.convd5 = nn.Sequential(
            nn.Conv2d(cat_channel , cat_channel, 3, 1, 1),
            nn.BatchNorm2d(cat_channel),
            # nn.ReLU(True)
        )
        self.relu=nn.ReLU(True)
        self.up_2 = nn.Upsample(scale_factor=2, mode='bilinear')
        self.up_4 = nn.Upsample(scale_factor=4, mode='bilinear')
        self.up_8 = nn.Upsample(scale_factor=8, mode='bilinear')
        self.up_16 = nn.Upsample(scale_factor=16, mode='bilinear')
        self.aspp = ASPP(filters[4], cat_channel)
        # for m in self.modules():
        #     if isinstance(m, nn.Conv2d):
        #         init_weights(m, init_type='kaiming')
        #     elif isinstance(m, nn.BatchNorm2d):
        #         init_weights(m, init_type='kaiming')

    def forward(self,x,train=False):
        x1=self.down_conv1(x)
        p1=self.maxpool2(x1)
        x2=self.down_conv2(p1)
        p2=self.maxpool2(x2)
        x3=self.down_conv3(p2)
        p3=self.maxpool2(x3)
        x4=self.down_conv4(p3)
        p4=self.maxpool2(x4)
        x5=self.down_conv5(p4)

        h1_5 = self.pool_conv1_5(x1)
        h2_5 = self.pool_conv2_5(x2)
        h3_5 = self.pool_conv3_5(x3)
        h4_5 = self.pool_conv4_5(x4)
        h5_5 = self.pool_conv5_5(x5)
        hd5 = torch.cat((h1_5, h2_5, h3_5, h4_5, h5_5), dim=1)
        hd5_=self.hdconv5(hd5)

        h1_4=self.pool_conv1_4(x1)
        h2_4=self.pool_conv2_4(x2)
        h3_4=self.pool_conv3_4(x3)
        h4_4=self.pool_conv4_4(x4)
        # h5_4=self.up_conv5_4_scal(x5)
        hd4=torch.cat((h1_4,h2_4,h3_4,h4_4),dim=1)
        # hd4 = torch.cat((h1_4,h2_4,h3_4, h4_4,h5_4), dim=1)
        hd4=self.hdconv4(hd4)
        edge4=self.edge4(hd4,(hd5_))

        h1_3=self.pool_conv1_3(x1)
        h2_3=self.pool_conv2_3(x2)
        h3_3=self.pool_conv3_3(x3)
        # h4_3=self.up_conv4_3_scal(x4)
        # h5_3=self.up_conv5_3_scal(x5)
        hd3=torch.cat((h1_3,h2_3,h3_3),dim=1)
        hd3=self.hdconv3(hd3)
        edge3=self.edge3(hd3,(hd4,hd5_))

        h1_2=self.pool_conv1_2(x1)
        h2_2=self.pool_conv2_2(x2)
        # h3_2=self.up_conv3_2_scal(x3)
        # h4_2=self.up_conv4_2_scal(x4)
        # h5_2=self.up_conv5_2_scal(x5)
        # hd2=torch.cat((h2_2,h3_2,h4_2,h5_2),dim=1)
        hd2 = torch.cat((h1_2,h2_2), dim=1)
        hd2=self.hdconv2(hd2)
        edge2=self.edge2(hd2,(hd3,hd4,hd5_))
        h1_1=self.pool_conv1_1(x1)
        hd1=self.hdconv1(h1_1)
        # edge1=self.edge1(hd1)


        d5_4=self.up_conv5_4(hd5)
        gate4=self.gate4(d5_4,edge4)
        d4_3=self.up_conv4_3(gate4)
        gate3=self.gate3(d4_3,edge3)
        d3_2=self.up_conv3_2(gate3)
        gate2=self.gate2(d3_2,edge2)
        # d2_1=self.up_conv2_1(gate2)
        # gate1=self.gate1(d2_1,edge1)
        # out=torch.cat((gate1,hd1),dim=1)

        edge2=self.gate_up2(gate2)
        # edge3=self.gate_up3(gate3)
        # edge4=self.gate_up4(gate4)

        a1_1=self.conv1_d1(hd1)
        a1_5=self.conv1_d5(hd1)
        a1_4=self.conv1_d4(hd1)
        a1_3=self.conv1_d3(hd1)
        a1_2=self.conv1_d2(hd1)
        # print(hd5.size())
        a1_5=self.convd5(a1_5)
        a1_4=self.convd4(torch.cat((a1_4,self.up_8(gate4)),dim=1))
        a1_3=self.convd3(torch.cat((a1_3,self.up_4(gate3)),dim=1))
        a1_2=self.convd2(torch.cat((a1_2,self.up_2(gate2)),dim=1))
        # a1=self.convd1(torch.cat((a1_2,a1_3,a1_4,a1_5,a1_1),dim=1))
        a1=self.relu(a1_4+a1_3+a1_2+a1_1+a1_5)
        # out=torch.cat((a1,d2_1),dim=1)
        out = self.out(a1)


        if train:
            return out,edge2
        return out




class M4(Module):
    def __init__(self,inchannels,out_channels,filters=(64,128,256,512,1024)):
        super(M4, self).__init__()
        cat_channel=filters[0]
        self.down_conv1=Down_Conv(inchannels,filters[0])
        self.down_conv2=Down_Conv(filters[0],filters[1])
        self.down_conv3=Down_Conv(filters[1],filters[2])
        self.down_conv4=Down_Conv(filters[2],filters[3])
        self.down_conv5=Down_Conv(filters[3],filters[4])
        self.maxpool2=nn.MaxPool2d(2,2)
        self.maxpool4=nn.MaxPool2d(4,4)
        self.maxpool8=nn.MaxPool2d(8,8)
        self.maxpool16=nn.MaxPool2d(16,16)

        self.pool_conv1_5=Pool_Conv(filters[0], cat_channel, 16)
        self.pool_conv2_5=Pool_Conv(filters[1], cat_channel, 8)
        self.pool_conv3_5=Pool_Conv(filters[2], cat_channel, 4)
        self.pool_conv4_5=Pool_Conv(filters[3], cat_channel, 2)
        self.pool_conv5_5=Pool_Conv(filters[4], cat_channel, 1)
        self.pool_conv1_4=Pool_Conv(filters[0],cat_channel,8)
        self.pool_conv2_4=Pool_Conv(filters[1],cat_channel,4)
        self.pool_conv3_4=Pool_Conv(filters[2],cat_channel,2)
        self.pool_conv4_4=Pool_Conv(filters[3],cat_channel,1)
        self.up_conv5_4_scal=Up_Conv_scal(filters[4],cat_channel,2)
        self.pool_conv1_3=Pool_Conv(filters[0],cat_channel,4)
        self.pool_conv2_3=Pool_Conv(filters[1],cat_channel,2)
        self.pool_conv3_3=Pool_Conv(filters[2],cat_channel,1)
        self.up_conv4_3_scal=Up_Conv_scal(filters[3],cat_channel,2)
        self.up_conv5_3_scal=Up_Conv_scal(filters[4],cat_channel,4)
        self.pool_conv1_2=Pool_Conv(filters[0],cat_channel,2)
        self.pool_conv2_2=Pool_Conv(filters[1],cat_channel,1)
        self.up_conv3_2_scal=Up_Conv_scal(filters[2],cat_channel,2)
        self.up_conv4_2_scal=Up_Conv_scal(filters[3],cat_channel,4)
        self.up_conv5_2_scal=Up_Conv_scal(filters[4],cat_channel,8)
        self.pool_conv1_1=Pool_Conv(filters[0],cat_channel,1)
        self.up_conv2_1_scal=Up_Conv_scal(filters[1],cat_channel,2)
        self.up_conv3_1_scal=Up_Conv_scal(filters[2],cat_channel,4)
        self.up_conv4_1_scal=Up_Conv_scal(filters[3],cat_channel,8)
        self.up_conv5_1_scal=Up_Conv_scal(filters[4],cat_channel,16)

        self.hdconv5=Hd_Conv(cat_channel*5,cat_channel)
        self.hdconv4=Hd_Conv(cat_channel*4,cat_channel)
        self.hdconv3=Hd_Conv(cat_channel*3,cat_channel)
        self.hdconv2=Hd_Conv(cat_channel*2,cat_channel)
        self.hdconv1=Hd_Conv(cat_channel,cat_channel)

        self.edge4=Edge_Conv2(cat_channel,1,1)
        self.edge3=Edge_Conv2(cat_channel,1,2)
        self.edge2=Edge_Conv2(cat_channel,1,3)
        # self.edge1=Edge_Conv(cat_channel,1,4)

        self.up_conv5_4=Up_Conv_scal(cat_channel,cat_channel,stride=2)
        self.up_conv5_3=Up_Conv_scal(cat_channel*5,cat_channel,stride=4)
        self.up_conv5_2=Up_Conv_scal(cat_channel*5,cat_channel,stride=8)
        self.up_conv5_1=Up_Conv_scal(cat_channel*5,cat_channel,stride=16)

        self.gate4=Gate(cat_channel,cat_channel)
        self.gate3=Gate(cat_channel,cat_channel)
        self.gate2=Gate(cat_channel,cat_channel)
        self.gate1=Gate(cat_channel,cat_channel)
        self.edge_out2=nn.Sequential(
            nn.Conv2d(cat_channel,cat_channel,3,1,1),
            nn.BatchNorm2d(cat_channel),
            nn.ReLU(True),
            nn.Conv2d(cat_channel,1,3,1,1)
        )
        self.edge_out3=nn.Sequential(
            nn.Conv2d(cat_channel, cat_channel, 3, 1, 1),
            nn.BatchNorm2d(cat_channel),
            nn.ReLU(True),
            nn.Conv2d(cat_channel,1,3,1,1)
        )
        self.edge_out4=nn.Sequential(
            nn.Conv2d(cat_channel, cat_channel, 3, 1, 1),
            nn.BatchNorm2d(cat_channel),
            nn.ReLU(True),
            nn.Conv2d(cat_channel, 1, 3, 1, 1)

        )

        self.out=nn.Sequential(
            nn.Conv2d(cat_channel,cat_channel,3,1,1),
            nn.BatchNorm2d(cat_channel),
            nn.ReLU(True),
            nn.Conv2d(cat_channel,out_channels,1,1,0),
        )

        self.gate_up2 = nn.Sequential(
            nn.Conv2d(cat_channel, cat_channel, 3, 1, 1),
            nn.BatchNorm2d(cat_channel),
            nn.ReLU(True),
            nn.Upsample(scale_factor=2, mode='bilinear'),
            nn.Conv2d(cat_channel, cat_channel, 1, 1, 0)
        )
        self.gate_up3 = nn.Sequential(
            nn.Conv2d(cat_channel, cat_channel, 3, 1, 1),
            nn.BatchNorm2d(cat_channel),
            nn.ReLU(True),
            nn.Upsample(scale_factor=4, mode='bilinear'),
            nn.Conv2d(cat_channel, cat_channel, 1, 1, 0)
        )
        self.gate_up4 = nn.Sequential(
            nn.Conv2d(cat_channel, cat_channel, 3, 1, 1),
            nn.BatchNorm2d(cat_channel),
            nn.ReLU(True),
            nn.Upsample(scale_factor=8, mode='bilinear'),
            nn.Conv2d(cat_channel, cat_channel, 1, 1, 0)
        )

        self.conv1_d1 = nn.Sequential(
            nn.Conv2d(cat_channel, cat_channel, 3, 1, 1),
            nn.BatchNorm2d(cat_channel),
            nn.ReLU(True)
        )
        self.conv1_d2 = nn.Sequential(
            # nn.Upsample(scale_factor=2,mode='bilinear'),
            nn.Conv2d(cat_channel, cat_channel, 3, 1, 4, dilation=4),
            nn.BatchNorm2d(cat_channel),
            nn.ReLU(True)
        )
        self.convd2 = nn.Sequential(
            nn.Conv2d(cat_channel * 2, cat_channel, 3, 1, 1),
            nn.BatchNorm2d(cat_channel),
            # nn.ReLU(True)
        )
        self.conv1_d3 = nn.Sequential(
            # nn.Upsample(scale_factor=4,mode='bilinear'),
            nn.Conv2d(cat_channel, cat_channel, 3, 1, 8, dilation=8),
            nn.BatchNorm2d(cat_channel),
            nn.ReLU(True)
        )
        self.convd3 = nn.Sequential(
            nn.Conv2d(cat_channel * 2, cat_channel, 3, 1, 1),
            nn.BatchNorm2d(cat_channel),
            # nn.ReLU(True)
        )
        self.conv1_d4 = nn.Sequential(
            # nn.Upsample(scale_factor=8,mode='bilinear'),
            nn.Conv2d(cat_channel, cat_channel, 3, 1, 12, dilation=12),
            nn.BatchNorm2d(cat_channel),
            nn.ReLU(True)
        )
        self.convd4 = nn.Sequential(
            nn.Conv2d(cat_channel * 2, cat_channel, 3, 1, 1),
            nn.BatchNorm2d(cat_channel),
            # nn.ReLU(True)
        )
        self.conv1_d5 = nn.Sequential(
            # nn.Upsample(scale_factor=16,mode='bilinear'),
            nn.Conv2d(cat_channel, cat_channel, 3, 1, 16, dilation=16),
            nn.BatchNorm2d(cat_channel),
            nn.ReLU(True)
        )
        self.convd5 = nn.Sequential(
            nn.Conv2d(cat_channel , cat_channel, 3, 1, 1),
            nn.BatchNorm2d(cat_channel),
            nn.ReLU(True)
        )
        self.convd1=nn.Sequential(
            nn.Conv2d(cat_channel*5,cat_channel,3,1,1),
            nn.BatchNorm2d(cat_channel),
            nn.ReLU(True)
        )

        self.relu=nn.ReLU(True)
        self.up_2 = nn.Upsample(scale_factor=2, mode='bilinear')
        self.up_4 = nn.Upsample(scale_factor=4, mode='bilinear')
        self.up_8 = nn.Upsample(scale_factor=8, mode='bilinear')
        self.up_16 = nn.Upsample(scale_factor=16, mode='bilinear')
        self.aspp = ASPP(cat_channel, cat_channel)
        self.sigmoid=nn.Sigmoid()
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init_weights(m, init_type='kaiming')
            elif isinstance(m, nn.BatchNorm2d):
                init_weights(m, init_type='kaiming')

    def forward(self, x, train=False):
        x1 = self.down_conv1(x)
        p1 = self.maxpool2(x1)
        x2 = self.down_conv2(p1)
        p2 = self.maxpool2(x2)
        x3 = self.down_conv3(p2)
        p3 = self.maxpool2(x3)
        x4 = self.down_conv4(p3)
        p4 = self.maxpool2(x4)
        x5 = self.down_conv5(p4)

        h1_5 = self.pool_conv1_5(x1)
        h2_5 = self.pool_conv2_5(x2)
        h3_5 = self.pool_conv3_5(x3)
        h4_5 = self.pool_conv4_5(x4)
        h5_5 = self.pool_conv5_5(x5)
        hd5 = torch.cat((h1_5, h2_5, h3_5, h4_5, h5_5), dim=1)
        hd5_=self.hdconv5(hd5)
        # hd5_=self.aspp(hd5_)
        h1_4 = self.pool_conv1_4(x1)
        h2_4 = self.pool_conv2_4(x2)
        h3_4 = self.pool_conv3_4(x3)
        h4_4 = self.pool_conv4_4(x4)
        # h5_4=self.up_conv5_4_scal(x5)
        hd4 = torch.cat((h1_4, h2_4, h3_4, h4_4), dim=1)
        # hd4 = torch.cat((h1_4,h2_4,h3_4, h4_4,h5_4), dim=1)
        hd4 = self.hdconv4(hd4)
        edge4 = self.edge4(hd4,(hd5_))
        d5_4 = self.up_conv5_4(hd5_)
        gate4 = self.gate4(d5_4, edge4)

        h1_3 = self.pool_conv1_3(x1)
        h2_3 = self.pool_conv2_3(x2)
        h3_3 = self.pool_conv3_3(x3)
        # h4_3=self.up_conv4_3_scal(x4)
        # h5_3=self.up_conv5_3_scal(x5)
        hd3 = torch.cat((h1_3, h2_3, h3_3), dim=1)
        hd3 = self.hdconv3(hd3)
        edge3 = self.edge3(hd3,(hd4,hd5_))
        d5_3 = self.up_conv5_3(hd5)
        gate3 = self.gate3(d5_3, edge3)

        h1_2 = self.pool_conv1_2(x1)
        h2_2 = self.pool_conv2_2(x2)
        # h3_2=self.up_conv3_2_scal(x3)
        # h4_2=self.up_conv4_2_scal(x4)
        # h5_2=self.up_conv5_2_scal(x5)
        # hd2=torch.cat((h2_2,h3_2,h4_2,h5_2),dim=1)
        hd2 = torch.cat((h1_2, h2_2), dim=1)
        hd2 = self.hdconv2(hd2)
        edge2 = self.edge2(hd2,(hd3,hd4,hd5_))
        d5_2 = self.up_conv5_2(hd5)
        gate2 = self.gate2(d5_2, edge2)

        h1_1 = self.pool_conv1_1(x1)
        hd1 = self.hdconv1(h1_1)
        # edge1=self.edge1(hd1)

        a1_5 = self.conv1_d5(hd1)
        a1_4 = self.conv1_d4(hd1)
        a1_3 = self.conv1_d3(hd1)
        a1_2 = self.conv1_d2(hd1)
        a1_1=self.conv1_d1(hd1)
        # a1_5 = self.convd5(a1_5)
        # a1_4=self.relu(a1_4+self.up_8(gate4))
        # a1_3=self.relu(a1_3+self.up_4(gate3))
        # a1_2=self.relu(a1_2+self.up_2(gate2))

        gate4_up=self.gate_up4(gate4)
        gate3_up=self.gate_up3(gate3)
        gate2_up=self.gate_up2(gate2)

        # print(a1_4.size(),gate4_up.size())
        a1_4 = self.convd4(torch.cat((a1_4, gate4_up), dim=1))
        a1_3 = self.convd3(torch.cat((a1_3, gate3_up), dim=1))
        a1_2 = self.convd2(torch.cat((a1_2, gate2_up), dim=1))
        # a1 = self.convd1(torch.cat((a1_4 , a1_3 , a1_2 , hd1 , a1_5),dim=1))
        # alpha=self.sigmoid(a1_4+a1_3+a1_2+a1_5+a1_1)
        # a1=hd1*(alpha+1)
        a1=self.relu(a1_4+a1_3+a1_2+a1_5+a1_1)
        out=self.out(a1)

        edge2_out=self.edge_out2(gate2_up)
        edge3_out=self.edge_out3(gate3_up)
        edge4_out=self.edge_out4(gate4_up)

        if train:
            return out, (edge2_out, edge3_out, edge4_out)
        return out


class M5(Module):
    def __init__(self,inchannels,out_channels,filters=(64,128,256,512,1024)):
        super(M5, self).__init__()
        cat_channel=filters[0]
        self.down_conv1=Down_Conv(inchannels,filters[0])
        self.down_conv2=Down_Conv(filters[0],filters[1])
        self.down_conv3=Down_Conv(filters[1],filters[2])
        self.down_conv4=Down_Conv(filters[2],filters[3])
        self.down_conv5=Down_Conv(filters[3],filters[4])
        self.maxpool2=nn.MaxPool2d(2,2)
        self.maxpool4=nn.MaxPool2d(4,4)
        self.maxpool8=nn.MaxPool2d(8,8)
        self.maxpool16=nn.MaxPool2d(16,16)

        self.pool_conv1_5=Pool_Conv(filters[0], cat_channel, 16)
        self.pool_conv2_5=Pool_Conv(filters[1], cat_channel, 8)
        self.pool_conv3_5=Pool_Conv(filters[2], cat_channel, 4)
        self.pool_conv4_5=Pool_Conv(filters[3], cat_channel, 2)
        self.pool_conv5_5=Pool_Conv(filters[4], cat_channel, 1)
        self.pool_conv1_4=Pool_Conv(filters[0],cat_channel,8)
        self.pool_conv2_4=Pool_Conv(filters[1],cat_channel,4)
        self.pool_conv3_4=Pool_Conv(filters[2],cat_channel,2)
        self.pool_conv4_4=Pool_Conv(filters[3],cat_channel,1)
        self.up_conv5_4_scal=Up_Conv_scal(filters[4],cat_channel,2)
        self.pool_conv1_3=Pool_Conv(filters[0],cat_channel,4)
        self.pool_conv2_3=Pool_Conv(filters[1],cat_channel,2)
        self.pool_conv3_3=Pool_Conv(filters[2],cat_channel,1)
        self.up_conv4_3_scal=Up_Conv_scal(filters[3],cat_channel,2)
        self.up_conv5_3_scal=Up_Conv_scal(filters[4],cat_channel,4)
        self.pool_conv1_2=Pool_Conv(filters[0],cat_channel,2)
        self.pool_conv2_2=Pool_Conv(filters[1],cat_channel,1)
        self.up_conv3_2_scal=Up_Conv_scal(filters[2],cat_channel,2)
        self.up_conv4_2_scal=Up_Conv_scal(filters[3],cat_channel,4)
        self.up_conv5_2_scal=Up_Conv_scal(filters[4],cat_channel,8)
        self.pool_conv1_1=Pool_Conv(filters[0],cat_channel,1)
        self.up_conv2_1_scal=Up_Conv_scal(filters[1],cat_channel,2)
        self.up_conv3_1_scal=Up_Conv_scal(filters[2],cat_channel,4)
        self.up_conv4_1_scal=Up_Conv_scal(filters[3],cat_channel,8)
        self.up_conv5_1_scal=Up_Conv_scal(filters[4],cat_channel,16)

        self.hdconv5=Hd_Conv(cat_channel*5,cat_channel)
        self.hdconv4=Hd_Conv(cat_channel*4,cat_channel)
        self.hdconv3=Hd_Conv(cat_channel*4,cat_channel)
        self.hdconv2=Hd_Conv(cat_channel*3,cat_channel)
        self.hdconv1=Hd_Conv(cat_channel*2,cat_channel)

        self.gate_hdconv4_3=Up_Conv_scal(cat_channel,cat_channel,2)
        self.gate_hdconv3_2=Up_Conv_scal(cat_channel, cat_channel, 2)
        self.gate_hdconv2_1=Up_Conv_scal(cat_channel, cat_channel, 2)

        self.edge4=Edge_Conv2(cat_channel,1,1)
        self.edge3=Edge_Conv2(cat_channel,1,2)
        self.edge2=Edge_Conv2(cat_channel,1,3)
        # self.edge1=Edge_Conv(cat_channel,1,4)

        self.up_conv5_4=Up_Conv_scal(cat_channel,cat_channel,stride=2)
        self.up_conv5_3=Up_Conv_scal(cat_channel*5,cat_channel,stride=4)
        self.up_conv5_2=Up_Conv_scal(cat_channel*5,cat_channel,stride=8)
        self.up_conv5_1=Up_Conv_scal(cat_channel*5,cat_channel,stride=16)

        self.gate4=Gate(cat_channel,cat_channel)
        self.gate3=Gate(cat_channel,cat_channel)
        self.gate2=Gate(cat_channel,cat_channel)
        self.gate1=Gate(cat_channel,cat_channel)
        self.edge_out2=nn.Sequential(
            nn.Conv2d(cat_channel,cat_channel,3,1,1),
            nn.BatchNorm2d(cat_channel),
            nn.ReLU(True),
            nn.Conv2d(cat_channel,1,3,1,1)
        )
        self.edge_out3=nn.Sequential(
            nn.Conv2d(cat_channel, cat_channel, 3, 1, 1),
            nn.BatchNorm2d(cat_channel),
            nn.ReLU(True),
            nn.Conv2d(cat_channel,1,3,1,1)
        )
        self.edge_out4=nn.Sequential(
            nn.Conv2d(cat_channel, cat_channel, 3, 1, 1),
            nn.BatchNorm2d(cat_channel),
            nn.ReLU(True),
            nn.Conv2d(cat_channel, 1, 3, 1, 1)

        )

        self.out=nn.Sequential(
            nn.Conv2d(cat_channel,cat_channel,3,1,1),
            nn.BatchNorm2d(cat_channel),
            nn.ReLU(True),
            nn.Conv2d(cat_channel,out_channels,1,1,0),
        )

        self.gate_up2 = nn.Sequential(
            nn.Conv2d(cat_channel, cat_channel, 3, 1, 1),
            nn.BatchNorm2d(cat_channel),
            nn.ReLU(True),
            nn.Upsample(scale_factor=2, mode='bilinear'),
            nn.Conv2d(cat_channel, cat_channel, 1, 1, 0)
        )
        self.gate_up3 = nn.Sequential(
            nn.Conv2d(cat_channel, cat_channel, 3, 1, 1),
            nn.BatchNorm2d(cat_channel),
            nn.ReLU(True),
            nn.Upsample(scale_factor=4, mode='bilinear'),
            nn.Conv2d(cat_channel, cat_channel, 1, 1, 0)
        )
        self.gate_up4 = nn.Sequential(
            nn.Conv2d(cat_channel, cat_channel, 3, 1, 1),
            nn.BatchNorm2d(cat_channel),
            nn.ReLU(True),
            nn.Upsample(scale_factor=8, mode='bilinear'),
            nn.Conv2d(cat_channel, cat_channel, 1, 1, 0)
        )

        self.conv1_d1 = nn.Sequential(
            nn.Conv2d(cat_channel, cat_channel, 3, 1, 1),
            nn.BatchNorm2d(cat_channel),
            nn.ReLU(True)
        )
        self.conv1_d2 = nn.Sequential(
            # nn.Upsample(scale_factor=2,mode='bilinear'),
            nn.Conv2d(cat_channel, cat_channel, 3, 1, 4, dilation=4),
            nn.BatchNorm2d(cat_channel),
            nn.ReLU(True)
        )
        self.convd2 = nn.Sequential(
            nn.Conv2d(cat_channel * 2, cat_channel, 3, 1, 1),
            nn.BatchNorm2d(cat_channel),
            # nn.ReLU(True)
        )
        self.conv1_d3 = nn.Sequential(
            # nn.Upsample(scale_factor=4,mode='bilinear'),
            nn.Conv2d(cat_channel, cat_channel, 3, 1, 8, dilation=8),
            nn.BatchNorm2d(cat_channel),
            nn.ReLU(True)
        )
        self.convd3 = nn.Sequential(
            nn.Conv2d(cat_channel * 2, cat_channel, 3, 1, 1),
            nn.BatchNorm2d(cat_channel),
            # nn.ReLU(True)
        )
        self.conv1_d4 = nn.Sequential(
            # nn.Upsample(scale_factor=8,mode='bilinear'),
            nn.Conv2d(cat_channel, cat_channel, 3, 1, 12, dilation=12),
            nn.BatchNorm2d(cat_channel),
            nn.ReLU(True)
        )
        self.convd4 = nn.Sequential(
            nn.Conv2d(cat_channel * 2, cat_channel, 3, 1, 1),
            nn.BatchNorm2d(cat_channel),
            # nn.ReLU(True)
        )
        self.conv1_d5 = nn.Sequential(
            # nn.Upsample(scale_factor=16,mode='bilinear'),
            nn.Conv2d(cat_channel, cat_channel, 3, 1, 16, dilation=16),
            nn.BatchNorm2d(cat_channel),
            nn.ReLU(True)
        )
        self.convd5 = nn.Sequential(
            nn.Conv2d(cat_channel , cat_channel, 3, 1, 1),
            nn.BatchNorm2d(cat_channel),
            nn.ReLU(True)
        )
        self.convd1=nn.Sequential(
            nn.Conv2d(cat_channel*5,cat_channel,3,1,1),
            nn.BatchNorm2d(cat_channel),
            nn.ReLU(True)
        )

        self.relu=nn.ReLU(True)
        self.up_2 = nn.Upsample(scale_factor=2, mode='bilinear')
        self.up_4 = nn.Upsample(scale_factor=4, mode='bilinear')
        self.up_8 = nn.Upsample(scale_factor=8, mode='bilinear')
        self.up_16 = nn.Upsample(scale_factor=16, mode='bilinear')
        self.aspp = ASPP(cat_channel, cat_channel)
        self.sigmoid=nn.Sigmoid()
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init_weights(m, init_type='kaiming')
            elif isinstance(m, nn.BatchNorm2d):
                init_weights(m, init_type='kaiming')

    def forward(self, x, train=False):
        x1 = self.down_conv1(x)
        p1 = self.maxpool2(x1)
        x2 = self.down_conv2(p1)
        p2 = self.maxpool2(x2)
        x3 = self.down_conv3(p2)
        p3 = self.maxpool2(x3)
        x4 = self.down_conv4(p3)
        p4 = self.maxpool2(x4)
        x5 = self.down_conv5(p4)

        h1_5 = self.pool_conv1_5(x1)
        h2_5 = self.pool_conv2_5(x2)
        h3_5 = self.pool_conv3_5(x3)
        h4_5 = self.pool_conv4_5(x4)
        h5_5 = self.pool_conv5_5(x5)
        hd5 = torch.cat((h1_5, h2_5, h3_5, h4_5, h5_5), dim=1)
        hd5_=self.hdconv5(hd5)
        # hd5_=self.aspp(hd5_)
        h1_4 = self.pool_conv1_4(x1)
        h2_4 = self.pool_conv2_4(x2)
        h3_4 = self.pool_conv3_4(x3)
        h4_4 = self.pool_conv4_4(x4)
        # h5_4=self.up_conv5_4_scal(x5)
        hd4 = torch.cat((h1_4, h2_4, h3_4, h4_4), dim=1)
        # hd4 = torch.cat((h1_4,h2_4,h3_4, h4_4,h5_4), dim=1)
        hd4 = self.hdconv4(hd4)
        edge4 = self.edge4(hd4,(hd5_))
        d5_4 = self.up_conv5_4(hd5_)
        gate4 = self.gate4(d5_4, edge4)

        h1_3 = self.pool_conv1_3(x1)
        h2_3 = self.pool_conv2_3(x2)
        h3_3 = self.pool_conv3_3(x3)
        # h4_3=self.up_conv4_3_scal(x4)
        # h5_3=self.up_conv5_3_scal(x5)
        hd3 = torch.cat((h1_3, h2_3, h3_3,self.gate_hdconv4_3(gate4)), dim=1)
        hd3 = self.hdconv3(hd3)
        edge3 = self.edge3(hd3,(hd4,hd5_))
        d5_3 = self.up_conv5_3(hd5)
        gate3 = self.gate3(d5_3, edge3)

        h1_2 = self.pool_conv1_2(x1)
        h2_2 = self.pool_conv2_2(x2)
        # h3_2=self.up_conv3_2_scal(x3)
        # h4_2=self.up_conv4_2_scal(x4)
        # h5_2=self.up_conv5_2_scal(x5)
        # hd2=torch.cat((h2_2,h3_2,h4_2,h5_2),dim=1)
        hd2 = torch.cat((h1_2, h2_2,self.gate_hdconv3_2(gate3)), dim=1)
        hd2 = self.hdconv2(hd2)
        edge2 = self.edge2(hd2,(hd3,hd4,hd5_))
        d5_2 = self.up_conv5_2(hd5)
        gate2 = self.gate2(d5_2, edge2)

        h1_1 = self.pool_conv1_1(x1)
        hd1 = self.hdconv1(torch.cat((h1_1,self.gate_hdconv2_1(gate2)),dim=1))
        # edge1=self.edge1(hd1)

        a1_5 = self.conv1_d5(hd1)
        a1_4 = self.conv1_d4(hd1)
        a1_3 = self.conv1_d3(hd1)
        a1_2 = self.conv1_d2(hd1)
        a1_1=self.conv1_d1(hd1)
        # a1_5 = self.convd5(a1_5)
        # a1_4=self.relu(a1_4+self.up_8(gate4))
        # a1_3=self.relu(a1_3+self.up_4(gate3))
        # a1_2=self.relu(a1_2+self.up_2(gate2))

        gate4_up=self.gate_up4(gate4)
        gate3_up=self.gate_up3(gate3)
        gate2_up=self.gate_up2(gate2)

        # print(a1_4.size(),gate4_up.size())
        a1_4 = self.convd4(torch.cat((a1_4, gate4_up), dim=1))
        a1_3 = self.convd3(torch.cat((a1_3, gate3_up), dim=1))
        a1_2 = self.convd2(torch.cat((a1_2, gate2_up), dim=1))
        # a1 = self.convd1(torch.cat((a1_4 , a1_3 , a1_2 , hd1 , a1_5),dim=1))
        # alpha=self.sigmoid(a1_4+a1_3+a1_2+a1_5+a1_1)
        # a1=hd1*(alpha+1)
        a1=self.relu(a1_4+a1_3+a1_2+a1_5+a1_1)
        out=self.out(a1)

        edge2_out=self.edge_out2(gate2_up)
        edge3_out=self.edge_out3(gate3_up)
        edge4_out=self.edge_out4(gate4_up)

        if train:
            return out, (edge2_out, edge3_out, edge4_out)
        return out


class M6(Module):
    def __init__(self,inchannels,out_channels,filters=(64,128,256,512,1024)):
        super(M6, self).__init__()
        cat_channel=filters[0]
        self.down_conv1=Down_Conv(inchannels,filters[0])
        self.down_conv2=Down_Conv(filters[0],filters[1])
        self.down_conv3=Down_Conv(filters[1],filters[2])
        self.down_conv4=Down_Conv(filters[2],filters[3])
        self.down_conv5=Down_Conv(filters[3],filters[4])
        self.maxpool2=nn.MaxPool2d(2,2)
        self.maxpool4=nn.MaxPool2d(4,4)
        self.maxpool8=nn.MaxPool2d(8,8)
        self.maxpool16=nn.MaxPool2d(16,16)

        self.pool_conv1_5=Pool_Conv(filters[0], cat_channel, 16)
        self.pool_conv2_5=Pool_Conv(filters[1], cat_channel, 8)
        self.pool_conv3_5=Pool_Conv(filters[2], cat_channel, 4)
        self.pool_conv4_5=Pool_Conv(filters[3], cat_channel, 2)
        self.pool_conv5_5=Pool_Conv(filters[4], cat_channel, 1)
        self.pool_conv1_4=Pool_Conv(filters[0],cat_channel,8)
        self.pool_conv2_4=Pool_Conv(filters[1],cat_channel,4)
        self.pool_conv3_4=Pool_Conv(filters[2],cat_channel,2)
        self.pool_conv4_4=Pool_Conv(filters[3],cat_channel,1)
        self.up_conv5_4_scal=Up_Conv_scal(filters[4],cat_channel,2)
        self.pool_conv1_3=Pool_Conv(filters[0],cat_channel,4)
        self.pool_conv2_3=Pool_Conv(filters[1],cat_channel,2)
        self.pool_conv3_3=Pool_Conv(filters[2],cat_channel,1)
        self.up_conv4_3_scal=Up_Conv_scal(filters[3],cat_channel,2)
        self.up_conv5_3_scal=Up_Conv_scal(filters[4],cat_channel,4)
        self.pool_conv1_2=Pool_Conv(filters[0],cat_channel,2)
        self.pool_conv2_2=Pool_Conv(filters[1],cat_channel,1)
        self.up_conv3_2_scal=Up_Conv_scal(filters[2],cat_channel,2)
        self.up_conv4_2_scal=Up_Conv_scal(filters[3],cat_channel,4)
        self.up_conv5_2_scal=Up_Conv_scal(filters[4],cat_channel,8)
        self.pool_conv1_1=Pool_Conv(filters[0],cat_channel,1)
        self.up_conv2_1_scal=Up_Conv_scal(filters[1],cat_channel,2)
        self.up_conv3_1_scal=Up_Conv_scal(filters[2],cat_channel,4)
        self.up_conv4_1_scal=Up_Conv_scal(filters[3],cat_channel,8)
        self.up_conv5_1_scal=Up_Conv_scal(filters[4],cat_channel,16)

        self.hdconv5=Hd_Conv(cat_channel*5,cat_channel)
        self.hdconv4=Hd_Conv(cat_channel*4,cat_channel)
        self.hdconv3=Hd_Conv(cat_channel*4,cat_channel)
        self.hdconv2=Hd_Conv(cat_channel*3,cat_channel)
        self.hdconv1=Hd_Conv(cat_channel*2,cat_channel)

        self.gate_hdconv4_3=Up_Conv_scal(cat_channel,cat_channel,2)
        self.gate_hdconv3_2=Up_Conv_scal(cat_channel, cat_channel, 2)
        self.gate_hdconv2_1=Up_Conv_scal(cat_channel, cat_channel, 2)

        self.edge4=Edge_Conv2(cat_channel,1,1)
        self.edge3=Edge_Conv2(cat_channel,1,2)
        self.edge2=Edge_Conv2(cat_channel,1,3)
        # self.edge1=Edge_Conv(cat_channel,1,4)

        self.up_conv5_4=Up_Conv_scal(cat_channel,cat_channel,stride=2)
        self.up_conv4_3=Up_Conv_scal(cat_channel,cat_channel,stride=2)
        self.up_conv3_2=Up_Conv_scal(cat_channel,cat_channel,stride=2)
        self.up_conv2_1=Up_Conv_scal(cat_channel,cat_channel,stride=2)

        self.gate4=Gate(cat_channel,cat_channel)
        self.gate3=Gate(cat_channel,cat_channel)
        self.gate2=Gate(cat_channel,cat_channel)
        self.gate1=Gate(cat_channel,cat_channel)
        self.edge_out2=nn.Sequential(
            nn.Conv2d(cat_channel,cat_channel,3,1,1),
            nn.BatchNorm2d(cat_channel),
            nn.ReLU(True),
            nn.Conv2d(cat_channel,1,3,1,1)
        )
        self.edge_out3=nn.Sequential(
            nn.Conv2d(cat_channel, cat_channel, 3, 1, 1),
            nn.BatchNorm2d(cat_channel),
            nn.ReLU(True),
            nn.Conv2d(cat_channel,1,3,1,1)
        )
        self.edge_out4=nn.Sequential(
            nn.Conv2d(cat_channel, cat_channel, 3, 1, 1),
            nn.BatchNorm2d(cat_channel),
            nn.ReLU(True),
            nn.Conv2d(cat_channel, 1, 3, 1, 1)

        )

        self.out=nn.Sequential(
            nn.Conv2d(cat_channel*2,cat_channel,3,1,1),
            nn.BatchNorm2d(cat_channel),
            nn.ReLU(True),
            nn.Conv2d(cat_channel,out_channels,1,1,0),
        )

        self.gate_up2 = nn.Sequential(
            nn.Conv2d(cat_channel, cat_channel, 3, 1, 1),
            nn.BatchNorm2d(cat_channel),
            nn.ReLU(True),
            nn.Upsample(scale_factor=2, mode='bilinear'),
            nn.Conv2d(cat_channel, cat_channel, 1, 1, 0)
        )
        self.gate_up3 = nn.Sequential(
            nn.Conv2d(cat_channel, cat_channel, 3, 1, 1),
            nn.BatchNorm2d(cat_channel),
            nn.ReLU(True),
            nn.Upsample(scale_factor=4, mode='bilinear'),
            nn.Conv2d(cat_channel, cat_channel, 1, 1, 0)
        )
        self.gate_up4 = nn.Sequential(
            nn.Conv2d(cat_channel, cat_channel, 3, 1, 1),
            nn.BatchNorm2d(cat_channel),
            nn.ReLU(True),
            nn.Upsample(scale_factor=8, mode='bilinear'),
            nn.Conv2d(cat_channel, cat_channel, 1, 1, 0)
        )

        self.conv1_d1 = nn.Sequential(
            nn.Conv2d(cat_channel, cat_channel, 3, 1, 1),
            nn.BatchNorm2d(cat_channel),
            nn.ReLU(True)
        )
        self.conv1_d2 = nn.Sequential(
            # nn.Upsample(scale_factor=2,mode='bilinear'),
            nn.Conv2d(cat_channel, cat_channel, 3, 1, 4, dilation=4),
            nn.BatchNorm2d(cat_channel),
            nn.ReLU(True)
        )
        self.convd2 = nn.Sequential(
            nn.Conv2d(cat_channel * 2, cat_channel, 3, 1, 1),
            nn.BatchNorm2d(cat_channel),
            # nn.ReLU(True)
        )
        self.conv1_d3 = nn.Sequential(
            # nn.Upsample(scale_factor=4,mode='bilinear'),
            nn.Conv2d(cat_channel, cat_channel, 3, 1, 8, dilation=8),
            nn.BatchNorm2d(cat_channel),
            nn.ReLU(True)
        )
        self.convd3 = nn.Sequential(
            nn.Conv2d(cat_channel * 2, cat_channel, 3, 1, 1),
            nn.BatchNorm2d(cat_channel),
            # nn.ReLU(True)
        )
        self.conv1_d4 = nn.Sequential(
            # nn.Upsample(scale_factor=8,mode='bilinear'),
            nn.Conv2d(cat_channel, cat_channel, 3, 1, 12, dilation=12),
            nn.BatchNorm2d(cat_channel),
            nn.ReLU(True)
        )
        self.convd4 = nn.Sequential(
            nn.Conv2d(cat_channel * 2, cat_channel, 3, 1, 1),
            nn.BatchNorm2d(cat_channel),
            # nn.ReLU(True)
        )
        self.conv1_d5 = nn.Sequential(
            # nn.Upsample(scale_factor=16,mode='bilinear'),
            nn.Conv2d(cat_channel, cat_channel, 3, 1, 16, dilation=16),
            nn.BatchNorm2d(cat_channel),
            nn.ReLU(True)
        )
        self.convd5 = nn.Sequential(
            nn.Conv2d(cat_channel , cat_channel, 3, 1, 1),
            nn.BatchNorm2d(cat_channel),
            nn.ReLU(True)
        )
        self.convd1=nn.Sequential(
            nn.Conv2d(cat_channel*5,cat_channel,3,1,1),
            nn.BatchNorm2d(cat_channel),
            nn.ReLU(True)
        )

        self.relu=nn.ReLU(True)
        self.up_2 = nn.Upsample(scale_factor=2, mode='bilinear')
        self.up_4 = nn.Upsample(scale_factor=4, mode='bilinear')
        self.up_8 = nn.Upsample(scale_factor=8, mode='bilinear')
        self.up_16 = nn.Upsample(scale_factor=16, mode='bilinear')
        self.aspp = ASPP(cat_channel, cat_channel)
        self.edge_out=nn.Sequential(
            nn.Conv2d(cat_channel*3,cat_channel,3,1,1),
            nn.BatchNorm2d(cat_channel),
            nn.ReLU(True),
            nn.Conv2d(cat_channel,cat_channel,1,1,0),
            nn.BatchNorm2d(cat_channel),
            nn.ReLU(True)
        )
        self.sigmoid=nn.Sigmoid()

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init_weights(m, init_type='kaiming')
            elif isinstance(m, nn.BatchNorm2d):
                init_weights(m, init_type='kaiming')

    def forward(self, x, train=False):
        x1 = self.down_conv1(x)
        p1 = self.maxpool2(x1)
        x2 = self.down_conv2(p1)
        p2 = self.maxpool2(x2)
        x3 = self.down_conv3(p2)
        p3 = self.maxpool2(x3)
        x4 = self.down_conv4(p3)
        p4 = self.maxpool2(x4)
        x5 = self.down_conv5(p4)

        h1_5 = self.pool_conv1_5(x1)
        h2_5 = self.pool_conv2_5(x2)
        h3_5 = self.pool_conv3_5(x3)
        h4_5 = self.pool_conv4_5(x4)
        h5_5 = self.pool_conv5_5(x5)
        hd5 = torch.cat((h1_5, h2_5, h3_5, h4_5, h5_5), dim=1)
        hd5_=self.hdconv5(hd5)
        # hd5_=self.aspp(hd5_)
        h1_4 = self.pool_conv1_4(x1)
        h2_4 = self.pool_conv2_4(x2)
        h3_4 = self.pool_conv3_4(x3)
        h4_4 = self.pool_conv4_4(x4)
        # h5_4=self.up_conv5_4_scal(x5)
        hd4 = torch.cat((h1_4, h2_4, h3_4, h4_4), dim=1)
        # hd4 = torch.cat((h1_4,h2_4,h3_4, h4_4,h5_4), dim=1)
        hd4 = self.hdconv4(hd4)
        edge4 = self.edge4(hd4,(hd5_))
        d5_4 = self.up_conv5_4(hd5_)
        gate4 = self.gate4(d5_4, edge4)

        h1_3 = self.pool_conv1_3(x1)
        h2_3 = self.pool_conv2_3(x2)
        h3_3 = self.pool_conv3_3(x3)
        # h4_3=self.up_conv4_3_scal(x4)
        # h5_3=self.up_conv5_3_scal(x5)
        hd3 = torch.cat((h1_3, h2_3, h3_3,self.gate_hdconv4_3(gate4)), dim=1)
        hd3 = self.hdconv3(hd3)
        edge3 = self.edge3(hd3,(hd4,hd5_))
        d4_3 = self.up_conv4_3(hd4)
        gate3 = self.gate3(d4_3, edge3)

        h1_2 = self.pool_conv1_2(x1)
        h2_2 = self.pool_conv2_2(x2)
        # h3_2=self.up_conv3_2_scal(x3)
        # h4_2=self.up_conv4_2_scal(x4)
        # h5_2=self.up_conv5_2_scal(x5)
        # hd2=torch.cat((h2_2,h3_2,h4_2,h5_2),dim=1)
        hd2 = torch.cat((h1_2, h2_2,self.gate_hdconv3_2(gate3)), dim=1)
        hd2 = self.hdconv2(hd2)
        edge2 = self.edge2(hd2,(hd3,hd4,hd5_))
        d3_2 = self.up_conv3_2(hd3)
        gate2 = self.gate2(d3_2, edge2)

        h1_1 = self.pool_conv1_1(x1)
        hd1 = self.hdconv1(torch.cat((h1_1,self.gate_hdconv2_1(gate2)),dim=1))
        # edge1=self.edge1(hd1)

        # a1_5 = self.conv1_d5(hd1)
        a1_4 = self.conv1_d4(hd1)
        a1_3 = self.conv1_d3(hd1)
        a1_2 = self.conv1_d2(hd1)
        a1_1=self.conv1_d1(hd1)
        # a1_5 = self.convd5(a1_5)
        # a1_4=self.relu(a1_4+self.up_8(gate4))
        # a1_3=self.relu(a1_3+self.up_4(gate3))
        # a1_2=self.relu(a1_2+self.up_2(gate2))

        gate4_up=self.gate_up4(gate4)
        gate3_up=self.gate_up3(gate3)
        gate2_up=self.gate_up2(gate2)

        # print(a1_4.size(),gate4_up.size())
        a1_4 = self.convd4(torch.cat((a1_4, gate4_up), dim=1))
        a1_3 = self.convd3(torch.cat((a1_3, gate3_up), dim=1))
        a1_2 = self.convd2(torch.cat((a1_2, gate2_up), dim=1))
        # a1 = self.convd1(torch.cat((a1_4 , a1_3 , a1_2 , hd1 , a1_5),dim=1))
        # alpha=self.sigmoid(a1_4+a1_3+a1_2+a1_5+a1_1)
        # a1=hd1*(alpha+1)
        a1=self.relu(a1_4+a1_3+a1_2+a1_1)

        # out=self.out(a1)

        edge=self.edge_out(torch.cat((gate2_up,gate3_up,gate4_up),dim=1))
        out=self.out(torch.cat((a1,edge),dim=1))
        edge_out=self.edge_out2(edge)
        # edge2_out=self.edge_out2(gate2_up)
        # edge3_out=self.edge_out3(gate3_up)
        # edge4_out=self.edge_out4(gate4_up)

        if train:
            return out, edge_out
        return out


class M7(Module):
    def __init__(self,inchannels,out_channels,filters=(64,128,256,512,1024)):
        super(M7, self).__init__()
        cat_channel=filters[0]*4
        self.down_conv1=Down_Conv(inchannels,filters[0])
        self.down_conv2=Down_Conv(filters[0],filters[1])
        self.down_conv3=Down_Conv(filters[1],filters[2])
        self.down_conv4=Down_Conv(filters[2],filters[3])
        self.down_conv5=Down_Conv(filters[3],filters[4])
        self.maxpool2=nn.MaxPool2d(2,2)
        self.maxpool4=nn.MaxPool2d(4,4)
        self.maxpool8=nn.MaxPool2d(8,8)
        self.maxpool16=nn.MaxPool2d(16,16)

        self.pool_conv1_5=Pool_Conv(filters[0], cat_channel, 16)
        self.pool_conv2_5=Pool_Conv(filters[1], cat_channel, 8)
        self.pool_conv3_5=Pool_Conv(filters[2], cat_channel, 4)
        self.pool_conv4_5=Pool_Conv(filters[3], cat_channel, 2)
        self.pool_conv5_5=Pool_Conv(filters[4], cat_channel, 1)
        self.pool_conv1_4=Pool_Conv(filters[0],cat_channel,8)
        self.pool_conv2_4=Pool_Conv(filters[1],cat_channel,4)
        self.pool_conv3_4=Pool_Conv(filters[2],cat_channel,2)
        self.pool_conv4_4=Pool_Conv(filters[3],cat_channel,1)
        self.up_conv5_4_scal=Up_Conv_scal(filters[4],cat_channel,2)
        self.pool_conv1_3=Pool_Conv(filters[0],cat_channel,4)
        self.pool_conv2_3=Pool_Conv(filters[1],cat_channel,2)
        self.pool_conv3_3=Pool_Conv(filters[2],cat_channel,1)
        self.up_conv4_3_scal=Up_Conv_scal(filters[3],cat_channel,2)
        self.up_conv5_3_scal=Up_Conv_scal(filters[4],cat_channel,4)
        self.pool_conv1_2=Pool_Conv(filters[0],cat_channel,2)
        self.pool_conv2_2=Pool_Conv(filters[1],cat_channel,1)
        self.up_conv3_2_scal=Up_Conv_scal(filters[2],cat_channel,2)
        self.up_conv4_2_scal=Up_Conv_scal(filters[3],cat_channel,4)
        self.up_conv5_2_scal=Up_Conv_scal(filters[4],cat_channel,8)
        self.pool_conv1_1=Pool_Conv(filters[0],cat_channel,1)
        self.up_conv2_1_scal=Up_Conv_scal(filters[1],cat_channel,2)
        self.up_conv3_1_scal=Up_Conv_scal(filters[2],cat_channel,4)
        self.up_conv4_1_scal=Up_Conv_scal(filters[3],cat_channel,8)
        self.up_conv5_1_scal=Up_Conv_scal(filters[4],cat_channel,16)

        self.hdconv5=Hd_Conv(cat_channel*5,cat_channel)
        self.hdconv4=Hd_Conv(cat_channel*4,cat_channel)
        self.hdconv3=Hd_Conv(cat_channel*4,cat_channel)
        self.hdconv2=Hd_Conv(cat_channel*3,cat_channel)
        self.hdconv1=Hd_Conv(cat_channel*2,cat_channel)

        self.gate_hdconv4_3=Up_Conv_scal(cat_channel,cat_channel,2)
        self.gate_hdconv3_2=Up_Conv_scal(cat_channel, cat_channel, 2)
        self.gate_hdconv2_1=Up_Conv_scal(cat_channel, cat_channel, 2)

        self.edge4=Edge_Conv2(cat_channel,1,1)
        self.edge3=Edge_Conv2(cat_channel,1,2)
        self.edge2=Edge_Conv2(cat_channel,1,3)
        # self.edge1=Edge_Conv(cat_channel,1,4)

        self.up_conv5_4=Up_Conv_scal(cat_channel,cat_channel,stride=2)
        self.up_conv4_3=Up_Conv_scal(cat_channel,cat_channel,stride=4)
        self.up_conv3_2=Up_Conv_scal(cat_channel,cat_channel,stride=8)
        self.up_conv2_1=Up_Conv_scal(cat_channel,cat_channel,stride=16)

        self.gate4=Gate(cat_channel,cat_channel)
        self.gate3=Gate(cat_channel,cat_channel)
        self.gate2=Gate(cat_channel,cat_channel)
        self.gate1=Gate(cat_channel,cat_channel)
        self.edge_out2=nn.Sequential(
            nn.Conv2d(cat_channel,cat_channel,3,1,1),
            nn.BatchNorm2d(cat_channel),
            nn.ReLU(True),
            nn.Conv2d(cat_channel,1,3,1,1)
        )
        self.edge_out3=nn.Sequential(
            nn.Conv2d(cat_channel, cat_channel, 3, 1, 1),
            nn.BatchNorm2d(cat_channel),
            nn.ReLU(True),
            nn.Conv2d(cat_channel,1,3,1,1)
        )
        self.edge_out4=nn.Sequential(
            nn.Conv2d(cat_channel, cat_channel, 3, 1, 1),
            nn.BatchNorm2d(cat_channel),
            nn.ReLU(True),
            nn.Conv2d(cat_channel, 1, 3, 1, 1)

        )

        self.out=nn.Sequential(
            nn.Conv2d(cat_channel,cat_channel,3,1,1),
            nn.BatchNorm2d(cat_channel),
            nn.ReLU(True),
            nn.Conv2d(cat_channel,out_channels,1,1,0),
        )

        self.gate_up2 = nn.Sequential(
            nn.Conv2d(cat_channel, cat_channel, 3, 1, 1),
            nn.BatchNorm2d(cat_channel),
            nn.ReLU(True),
            nn.Upsample(scale_factor=2, mode='bilinear'),
            nn.Conv2d(cat_channel, cat_channel, 1, 1, 0)
        )
        self.gate_up3 = nn.Sequential(
            nn.Conv2d(cat_channel, cat_channel, 3, 1, 1),
            nn.BatchNorm2d(cat_channel),
            nn.ReLU(True),
            nn.Upsample(scale_factor=4, mode='bilinear'),
            nn.Conv2d(cat_channel, cat_channel, 1, 1, 0)
        )
        self.gate_up4 = nn.Sequential(
            nn.Conv2d(cat_channel, cat_channel, 3, 1, 1),
            nn.BatchNorm2d(cat_channel),
            nn.ReLU(True),
            nn.Upsample(scale_factor=8, mode='bilinear'),
            nn.Conv2d(cat_channel, cat_channel, 1, 1, 0)
        )

        self.conv1_d1 = nn.Sequential(
            nn.Conv2d(cat_channel, cat_channel, 3, 1, 1),
            nn.BatchNorm2d(cat_channel),
            nn.ReLU(True)
        )
        self.conv1_d2 = nn.Sequential(
            # nn.Upsample(scale_factor=2,mode='bilinear'),
            nn.Conv2d(cat_channel, cat_channel, 3, 1, 4, dilation=4),
            nn.BatchNorm2d(cat_channel),
            nn.ReLU(True)
        )
        self.convd2 = nn.Sequential(
            nn.Conv2d(cat_channel * 2, cat_channel, 3, 1, 1),
            nn.BatchNorm2d(cat_channel),
            # nn.ReLU(True)
        )
        self.conv1_d3 = nn.Sequential(
            # nn.Upsample(scale_factor=4,mode='bilinear'),
            nn.Conv2d(cat_channel, cat_channel, 3, 1, 8, dilation=8),
            nn.BatchNorm2d(cat_channel),
            nn.ReLU(True)
        )
        self.convd3 = nn.Sequential(
            nn.Conv2d(cat_channel * 2, cat_channel, 3, 1, 1),
            nn.BatchNorm2d(cat_channel),
            # nn.ReLU(True)
        )
        self.conv1_d4 = nn.Sequential(
            # nn.Upsample(scale_factor=8,mode='bilinear'),
            nn.Conv2d(cat_channel, cat_channel, 3, 1, 12, dilation=12),
            nn.BatchNorm2d(cat_channel),
            nn.ReLU(True)
        )
        self.convd4 = nn.Sequential(
            nn.Conv2d(cat_channel * 2, cat_channel, 3, 1, 1),
            nn.BatchNorm2d(cat_channel),
            # nn.ReLU(True)
        )
        self.conv1_d5 = nn.Sequential(
            # nn.Upsample(scale_factor=16,mode='bilinear'),
            nn.Conv2d(cat_channel, cat_channel, 3, 1, 16, dilation=16),
            nn.BatchNorm2d(cat_channel),
            nn.ReLU(True)
        )
        self.convd5 = nn.Sequential(
            nn.Conv2d(cat_channel , cat_channel, 3, 1, 1),
            nn.BatchNorm2d(cat_channel),
            nn.ReLU(True)
        )
        self.convd1=nn.Sequential(
            nn.Conv2d(cat_channel*5,cat_channel,3,1,1),
            nn.BatchNorm2d(cat_channel),
            nn.ReLU(True)
        )

        self.relu=nn.ReLU(True)
        self.up_2 = nn.Upsample(scale_factor=2, mode='bilinear')
        self.up_4 = nn.Upsample(scale_factor=4, mode='bilinear')
        self.up_8 = nn.Upsample(scale_factor=8, mode='bilinear')
        self.up_16 = nn.Upsample(scale_factor=16, mode='bilinear')
        self.aspp = ASPP(cat_channel, cat_channel)
        self.sigmoid=nn.Sigmoid()
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init_weights(m, init_type='kaiming')
            elif isinstance(m, nn.BatchNorm2d):
                init_weights(m, init_type='kaiming')

    def forward(self, x, train=False):
        x1 = self.down_conv1(x)
        p1 = self.maxpool2(x1)
        x2 = self.down_conv2(p1)
        p2 = self.maxpool2(x2)
        x3 = self.down_conv3(p2)
        p3 = self.maxpool2(x3)
        x4 = self.down_conv4(p3)
        p4 = self.maxpool2(x4)
        x5 = self.down_conv5(p4)

        h1_5 = self.pool_conv1_5(x1)
        h2_5 = self.pool_conv2_5(x2)
        h3_5 = self.pool_conv3_5(x3)
        h4_5 = self.pool_conv4_5(x4)
        h5_5 = self.pool_conv5_5(x5)
        hd5 = torch.cat((h1_5, h2_5, h3_5, h4_5, h5_5), dim=1)
        hd5_=self.hdconv5(hd5)
        # hd5_=self.aspp(hd5_)
        h1_4 = self.pool_conv1_4(x1)
        h2_4 = self.pool_conv2_4(x2)
        h3_4 = self.pool_conv3_4(x3)
        h4_4 = self.pool_conv4_4(x4)
        # h5_4=self.up_conv5_4_scal(x5)
        hd4 = torch.cat((h1_4, h2_4, h3_4, h4_4), dim=1)
        # hd4 = torch.cat((h1_4,h2_4,h3_4, h4_4,h5_4), dim=1)
        hd4 = self.hdconv4(hd4)
        edge4 = self.edge4(hd4,(hd5_))
        d5_4 = self.up_conv5_4(hd5_)
        gate4 = self.gate4(d5_4, edge4)

        h1_3 = self.pool_conv1_3(x1)
        h2_3 = self.pool_conv2_3(x2)
        h3_3 = self.pool_conv3_3(x3)
        # h4_3=self.up_conv4_3_scal(x4)
        # h5_3=self.up_conv5_3_scal(x5)
        hd3 = torch.cat((h1_3, h2_3, h3_3,self.gate_hdconv4_3(gate4)), dim=1)
        hd3 = self.hdconv3(hd3)
        edge3 = self.edge3(hd3,(hd4,hd5_))
        d4_3 = self.up_conv4_3(hd5_)
        gate3 = self.gate3(d4_3, edge3)

        h1_2 = self.pool_conv1_2(x1)
        h2_2 = self.pool_conv2_2(x2)
        # h3_2=self.up_conv3_2_scal(x3)
        # h4_2=self.up_conv4_2_scal(x4)
        # h5_2=self.up_conv5_2_scal(x5)
        # hd2=torch.cat((h2_2,h3_2,h4_2,h5_2),dim=1)
        hd2 = torch.cat((h1_2, h2_2,self.gate_hdconv3_2(gate3)), dim=1)
        hd2 = self.hdconv2(hd2)
        edge2 = self.edge2(hd2,(hd3,hd4,hd5_))
        d3_2 = self.up_conv3_2(hd5_)
        gate2 = self.gate2(d3_2, edge2)

        h1_1 = self.pool_conv1_1(x1)
        hd1 = self.hdconv1(torch.cat((h1_1,self.gate_hdconv2_1(gate2)),dim=1))
        # edge1=self.edge1(hd1)

        # a1_5 = self.conv1_d5(hd1)
        a1_4 = self.conv1_d4(hd1)
        a1_3 = self.conv1_d3(hd1)
        a1_2 = self.conv1_d2(hd1)
        a1_1=self.conv1_d1(hd1)
        # a1_5 = self.convd5(a1_5)
        # a1_4=self.relu(a1_4+self.up_8(gate4))
        # a1_3=self.relu(a1_3+self.up_4(gate3))
        # a1_2=self.relu(a1_2+self.up_2(gate2))

        gate4_up=self.gate_up4(gate4)
        gate3_up=self.gate_up3(gate3)
        gate2_up=self.gate_up2(gate2)

        # print(a1_4.size(),gate4_up.size())
        a1_4 = self.convd4(torch.cat((a1_4, gate4_up), dim=1))
        a1_3 = self.convd3(torch.cat((a1_3, gate3_up), dim=1))
        a1_2 = self.convd2(torch.cat((a1_2, gate2_up), dim=1))
        # a1 = self.convd1(torch.cat((a1_4 , a1_3 , a1_2 , hd1 , a1_5),dim=1))
        # alpha=self.sigmoid(a1_4+a1_3+a1_2+a1_5+a1_1)
        # a1=hd1*(alpha+1)
        a1=self.relu(a1_4+a1_3+a1_2+a1_1)

        out=self.out(a1)

        edge2_out=self.edge_out2(gate2_up)
        edge3_out=self.edge_out3(gate3_up)
        edge4_out=self.edge_out4(gate4_up)

        if train:
            return out, (edge2_out, edge3_out, edge4_out)
        return out

if __name__=='__main__':
    net=M6(3,1,(4,32,64,98,96)).cuda()
    summary(net,(3,512,512))



