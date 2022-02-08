import torch
from torch.nn import Module
import torch.nn as nn
from net.unet3_plus import Unet_3plus
import numpy as np
from torchsummary import summary
from torch.nn import functional as F

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

class EDGE_Net(Module):
    def __init__(self,in_channels,out_channels,filters=(32,64,96)):
        super(EDGE_Net, self).__init__()
        self.conv1=nn.Sequential(
            nn.Conv2d(in_channels,filters[0],3,1,1),
            nn.BatchNorm2d(filters[0]),
            nn.ReLU(True)
        )
        self.conv2=nn.Sequential(
            nn.Conv2d(filters[0],filters[1],3,1,1),
            nn.BatchNorm2d(filters[1]),
            nn.ReLU(True)
        )
        self.conv3=nn.Sequential(
            nn.Conv2d(filters[1],filters[2],3,1,1),
            nn.BatchNorm2d(filters[2]),
            nn.ReLU(True)
        )
        self.conv_up2=nn.Sequential(
            nn.Conv2d(filters[1]+filters[2],filters[1],3,1,1),
            nn.BatchNorm2d(filters[1]),
            nn.ReLU(True)
        )
        self.conv_up1=nn.Sequential(
            nn.Conv2d(filters[0]+filters[1],filters[0],3,1,1),
            nn.BatchNorm2d(filters[0]),
            nn.ReLU(True)
        )
        self.out=nn.Conv2d(filters[0],out_channels,3,1,1)
        self.pool=nn.MaxPool2d(2,2)
        self.up=nn.Upsample(scale_factor=2,mode='bilinear')
    def forward(self,x,res=True):
        x1=self.conv1(x)
        p1=self.pool(x1)
        x2=self.conv2(p1)
        p2=self.pool(x2)
        x3=self.conv3(p2)
        x3_2=self.up(x3)
        h2=torch.cat((x2,x3_2),dim=1)
        h2=self.conv_up2(h2)
        x2_1=self.up(h2)
        h1=torch.cat((x1,x2_1),dim=1)
        h1=self.conv_up1(h1)
        out=self.out(h1)
        if res :
            return out
        return h1,out


class MSBlock(nn.Module):
    def __init__(self, c_in, rate=4):
        super(MSBlock, self).__init__()
        c_out = c_in
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

        self._initialize_weights()

    def forward(self, x):
        o = self.relu(self.conv(x))
        o1 = self.relu1(self.conv1(o))
        o2 = self.relu2(self.conv2(o))
        o3 = self.relu3(self.conv3(o))
        out = o + o1 + o2 + o3
        return out

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                m.weight.data.normal_(0, 0.01)
                if m.bias is not None:
                    m.bias.data.zero_()

class bdsn_Unet(Module):
    def __init__(self,in_channels=3,out_channels=1,filters=(64,128,256,512,1024),rate=4,logger=None):
        super(bdsn_Unet, self).__init__()
        t=1
        cat_channels=5*filters[0]
        self.unet=Unet_3plus(in_channels,out_channels,filters)
        self.msblock5_1 = MSBlock(filters[4], rate)
        self.conv5_1_down = nn.Conv2d(32 * t, 21, (1, 1), stride=1)
        self.score_dsn5 = nn.Conv2d(21, 1, (1, 1), stride=1)
        self.score_dsn5_1 = nn.Conv2d(21, 1, 1, stride=1)
        self.msblock4_1 = MSBlock(cat_channels, rate)
        self.conv4_1_down = nn.Conv2d(32 * t, 21, (1, 1), stride=1)
        self.score_dsn4 = nn.Conv2d(21, 1, (1, 1), stride=1)
        self.score_dsn4_1 = nn.Conv2d(21, 1, (1, 1), stride=1)
        self.msblock3_1 = MSBlock(cat_channels, rate)
        self.conv3_1_down = nn.Conv2d(32 * t, 21, (1, 1), stride=1)
        self.score_dsn3 = nn.Conv2d(21, 1, (1, 1), stride=1)
        self.score_dsn3_1 = nn.Conv2d(21, 1, (1, 1), stride=1)
        self.msblock2_1 = MSBlock(cat_channels, rate)
        self.conv2_1_down = nn.Conv2d(32 * t, 21, (1, 1), stride=1)
        self.score_dsn2 = nn.Conv2d(21, 1, (1, 1), stride=1)
        self.score_dsn2_1 = nn.Conv2d(21, 1, (1, 1), stride=1)
        self.msblock1_1 = MSBlock(filters[0], rate)
        self.conv1_1_down = nn.Conv2d(32 * t, 21, (1, 1), stride=1)
        self.score_dsn1 = nn.Conv2d(21, 1, (1, 1), stride=1)
        self.score_dsn1_1 = nn.Conv2d(21, 1, (1, 1), stride=1)
        self.upsample_2 = nn.ConvTranspose2d(1, 1, 4, stride=2, bias=False)
        self.upsample_4 = nn.ConvTranspose2d(1, 1, 8, stride=4, bias=False)
        self.upsample_8 = nn.ConvTranspose2d(1, 1, 16, stride=8, bias=False)
        self.upsample_16 = nn.ConvTranspose2d(1, 1, 32, stride=16, bias=False)
        self.fuse = nn.Conv2d(10, 1, 1, stride=1)
        self.sigoid = nn.Sigmoid()
        # self._initialize_weights(logger)

    def forward(self,x):
        features = self.unet(x,res=False,res2=True)
        x5,x4,x3,x2,x1=features
        # print(x5.size(),x4.size(),x3.size(),x2.size(),x1.size())
        sum5 = self.conv5_1_down(self.msblock5_1(x5))
        s5 = self.score_dsn1(sum5)
        s51 = self.score_dsn1_1(sum5)
        s5 = self.upsample_16(s5)
        s51 = self.upsample_16(s51)
        s5=crop(s5,x,0,0)
        s51 = crop(s51, x, 0, 0)

        sum4 = self.conv4_1_down(self.msblock4_1(x4))
        s4 = self.score_dsn4(sum4)
        s4 = self.upsample_8(s4)
        # print(s4.data.shape)
        s4 = crop(s4, x, 4, 4)
        s41 = self.score_dsn4_1(sum4)
        s41 = self.upsample_8(s41)
        s41 = crop(s41, x, 4, 4)
        # print(s4.size(),s41.size())
        sum3 = self.conv3_1_down(self.msblock3_1(x3))
        s3 = self.score_dsn3(sum3)
        s3 = self.upsample_4(s3)
        # print(s3.data.shape)
        s3 = crop(s3, x, 2, 2)
        s31 = self.score_dsn3_1(sum3)
        s31 = self.upsample_4(s31)
        # print(s31.data.shape)
        s31 = crop(s31, x, 2, 2)

        sum2 = self.conv2_1_down(self.msblock2_1(x2))
        s2 = self.score_dsn2(sum2)
        s21 = self.score_dsn2_1(sum2)
        s2 = self.upsample_2(s2)
        s21 = self.upsample_2(s21)
        # print(s2.data.shape, s21.data.shape)
        s2 = crop(s2, x, 1, 1)
        s21 = crop(s21, x, 1, 1)

        sum1 = self.conv1_1_down(self.msblock1_1(x1))
        s1 = self.score_dsn1(sum1)
        s11 = self.score_dsn1_1(sum1)
        # print(s11.size(),s21.size(),s31.size(),s41.size(),s51.size())
        o1, o2, o3, o4, o5 = s1.detach(), s2.detach(), s3.detach(), s4.detach(), s5.detach()
        o11, o21, o31, o41, o51 = s11.detach(), s21.detach(), s31.detach(), s41.detach(), s51.detach()
        p1_1 = s1
        p2_1 = s2 + o1
        p3_1 = s3 + o2 + o1
        p4_1 = s4 + o3 + o2 + o1
        p5_1 = s5 + o4 + o3 + o2 + o1
        p1_2 = s11 + o21 + o31 + o41 + o51
        p2_2 = s21 + o31 + o41 + o51
        p3_2 = s31 + o41 + o51
        p4_2 = s41 + o51
        p5_2 = s51
        fuse = self.fuse(torch.cat([p1_1, p2_1, p3_1, p4_1, p5_1, p1_2, p2_2, p3_2, p4_2, p5_2], 1))
        self.res = torch.sigmoid(fuse)

        return [p1_1, p2_1, p3_1, p4_1, p5_1, p1_2, p2_2, p3_2, p4_2, p5_2, fuse]


    def _initialize_weights(self, logger=None):
        for name, param in self.state_dict().items():
            # if self.pretrain and 'features' in name:
            #     continue
            # elif 'down' in name:
            #     param.zero_()
            if 'upsample' in name:
                if logger:
                    logger.info('init upsamle layer %s ' % name)
                k = int(name.split('.')[0].split('_')[1])
                param.copy_(get_upsampling_weight(1, 1, k*2))
            elif 'fuse' in name:
                if logger:
                    logger.info('init params %s ' % name)
                if 'bias' in name:
                    param.zero_()
                else:
                    nn.init.constant(param, 0.080)
            else:
                if logger:
                    logger.info('init params %s ' % name)
                if 'bias' in name:
                    param.zero_()
                else:
                    param.normal_(0, 0.01)

def crop(data1, data2, crop_h, crop_w):
    _, _, h1, w1 = data1.size()
    _, _, h2, w2 = data2.size()
    assert(h2 <= h1 and w2 <= w1)
    data = data1[:, :, crop_h:crop_h+h2, crop_w:crop_w+w2]
    return data

def get_upsampling_weight(in_channels, out_channels, kernel_size):
    """Make a 2D bilinear kernel suitable for upsampling"""
    factor = (kernel_size + 1) // 2
    if kernel_size % 2 == 1:
        center = factor - 1
    else:
        center = factor - 0.5
    og = np.ogrid[:kernel_size, :kernel_size]
    filt = (1 - abs(og[0] - center) / factor) * \
           (1 - abs(og[1] - center) / factor)
    weight = np.zeros((in_channels, out_channels, kernel_size, kernel_size),
                      dtype=np.float64)
    weight[range(in_channels), range(out_channels), :, :] = filt
    return torch.from_numpy(weight).float()


class UpConv(Module):
    def __init__(self,in_channel,out_channel):
        super(UpConv, self).__init__()
        self.conv1=nn.Upsample(scale_factor=2,mode='bilinear')
        self.conv2=DoubleConv(in_channel,out_channel)
    def forward(self,x):
        x=self.conv1(x)
        y=self.conv2(x)
        return y


class EdgeUnet(Module):
    def __init__(self,in_channels=3,out_channels=1,filters=(64,128,256,512,1024),rate=4):
        super(EdgeUnet, self).__init__()
        self.conv1=DoubleConv(in_channels,filters[0])
        self.conv2=DoubleConv(filters[0],filters[1])
        self.conv3=DoubleConv(filters[1],filters[2])
        self.conv4=DoubleConv(filters[2],filters[3])
        self.conv5=DoubleConv(filters[3],filters[4])
        self.maxpool=nn.MaxPool2d(2,2)
        self.up_conv5_4=UpConv(filters[4],filters[3])
        self.conv4_1=DoubleConv(filters[3]*2,filters[3])
        self.up_conv4_3=UpConv(filters[3],filters[2])
        self.conv3_1=DoubleConv(filters[2]*2,filters[2])
        self.up_conv3_2=UpConv(filters[2],filters[1])
        self.conv2_1=DoubleConv(filters[1]*2,filters[1])
        self.up_conv2_1=UpConv(filters[1],filters[0])
        self.conv1_1=DoubleConv(filters[0]*2,filters[0])

        t=1
        self.msblock5_1 = MSBlock(filters[4], rate)
        self.conv5_1_down = nn.Conv2d(32 * t, 21, (1, 1), stride=1)
        self.score_dsn5 = nn.Conv2d(21, 1, (1, 1), stride=1)
        self.score_dsn5_1 = nn.Conv2d(21, 1, 1, stride=1)
        self.msblock4_1 = MSBlock(filters[3], rate)
        self.msblock4_2 = MSBlock(filters[3],rate)
        self.conv4_1_down = nn.Conv2d(32 * t, 21, (1, 1), stride=1)
        self.conv4_2_down = nn.Conv2d(32 * t, 21, (1, 1), stride=1)
        self.score_dsn4 = nn.Conv2d(21, 1, (1, 1), stride=1)
        self.score_dsn4_1 = nn.Conv2d(21, 1, (1, 1), stride=1)
        self.msblock3_1 = MSBlock(filters[2], rate)
        self.msblock3_2 = MSBlock(filters[2],rate)
        self.conv3_1_down = nn.Conv2d(32 * t, 21, (1, 1), stride=1)
        self.conv3_2_down = nn.Conv2d(32 * t, 21, (1, 1), stride=1)
        self.score_dsn3 = nn.Conv2d(21, 1, (1, 1), stride=1)
        self.score_dsn3_1 = nn.Conv2d(21, 1, (1, 1), stride=1)
        self.msblock2_1 = MSBlock(filters[1], rate)
        self.msblock2_2 = MSBlock(filters[1],rate)
        self.conv2_1_down = nn.Conv2d(32 * t, 21, (1, 1), stride=1)
        self.conv2_2_down=nn.Conv2d(32 * t, 21, (1, 1), stride=1)
        self.score_dsn2 = nn.Conv2d(21, 1, (1, 1), stride=1)
        self.score_dsn2_1 = nn.Conv2d(21, 1, (1, 1), stride=1)
        self.msblock1_1 = MSBlock(filters[0], rate)
        self.msblock1_2 =MSBlock(filters[0],rate)
        self.conv1_1_down = nn.Conv2d(32 * t, 21, (1, 1), stride=1)
        self.conv1_2_down =nn.Conv2d(32*t,21,(1,1),stride=1)
        self.score_dsn1 = nn.Conv2d(21, 1, (1, 1), stride=1)
        self.score_dsn1_1 = nn.Conv2d(21, 1, (1, 1), stride=1)
        self.upsample_2 = nn.ConvTranspose2d(32, 32, 2, stride=2, bias=False)
        self.upsample_4 = nn.ConvTranspose2d(32, 32, 4, stride=4, bias=False)
        self.upsample_8 = nn.ConvTranspose2d(32, 32, 8, stride=8, bias=False)
        self.upsample_16 = nn.ConvTranspose2d(32, 32, 16, stride=16, bias=False)
        self.fuse = nn.Conv2d(5, 1, 1, stride=1)
        self.sigoid = nn.Sigmoid()
        # self.weights_init()
        initialize_weights(self)


    def forward(self,x,train=False):
        x1=self.conv1(x)
        x2=self.maxpool(x1)
        x2=self.conv2(x2)
        x3=self.maxpool(x2)
        x3=self.conv3(x3)
        x4=self.maxpool(x3)
        x4=self.conv4(x4)
        x5=self.maxpool(x4)
        hd5=self.conv5(x5)
        h5_4=self.up_conv5_4(hd5)
        hd4=torch.cat((h5_4,x4),dim=1)
        hd4=self.conv4_1(hd4)
        h4_3=self.up_conv4_3(hd4)
        hd3=torch.cat((x3,h4_3),dim=1)
        hd3=self.conv3_1(hd3)
        h3_2=self.up_conv3_2(hd3)
        hd2=torch.cat((x2,h3_2),dim=1)
        hd2=self.conv2_1(hd2)
        h2_1=self.up_conv2_1(hd2)
        hd1=torch.cat((x1,h2_1),dim=1)
        hd1=self.conv1_1(hd1)

        s5_1=self.msblock5_1(hd5)
        s5_1 = self.upsample_16(s5_1)
        s5_1=self.conv5_1_down(s5_1)
        s5_1_out=self.score_dsn5_1(s5_1)
        s4_1 = self.msblock4_1(hd4)
        s4_1 = self.upsample_8(s4_1)
        s4_1=self.conv4_1_down(s4_1)
        s4_1_out=self.score_dsn4_1(s4_1)
        s3_1=self.msblock3_1(hd3)
        s3_1=self.upsample_4(s3_1)
        s3_1=self.conv3_1_down(s3_1)
        s3_1_out=self.score_dsn3_1(s3_1)
        s2_1=self.msblock2_1(hd2)
        s2_1=self.upsample_2(s2_1)
        s2_1=self.conv2_1_down(s2_1)
        s2_1_out=self.score_dsn2_1(s2_1)
        s1_1=self.msblock1_1(hd1)
        s1_1=self.conv1_1_down(s1_1)
        s1_1_out=self.score_dsn1_1(s1_1)

        fuse=torch.cat((s5_1_out,s4_1_out,s3_1_out,s2_1_out,s1_1_out),dim=1)
        fuse=self.fuse(fuse)
        # print(s5_1_out.size(),s4_1_out.size(),s3_1_out.size(),s2_1_out.size(),s1_1_out.size(),fuse.size())
        if train:
            return (s5_1_out,s4_1_out,s3_1_out,s2_1_out,s1_1_out,fuse)
        else:
            return fuse

    def weights_init(m):
        if isinstance(m, nn.Conv2d):
            # xavier(m.weight.data)
            m.weight.data.normal_(0, 0.01)
            if m.bias is not None:
                m.bias.data.zero_()


class MSBlock2(nn.Module):
    def __init__(self, c_in,c_out=21, rate=4):
        super(MSBlock2, self).__init__()
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

class ddUnet(Module):
    def __init__(self,in_channels,out_channels,filters=(64,128,256,512,1024)):
        super(ddUnet, self).__init__()
        mse_out=21
        up=filters[0]*5
        self.unet_model=Unet_3plus(in_channels,out_channels,filters)
        self.mse1=MSBlock2(filters[0],mse_out)
        self.mse2=MSBlock2(up,mse_out)
        self.mse3=MSBlock2(up,mse_out)
        self.mse4=MSBlock2(up,mse_out)


        self.edge4=nn.Conv2d(mse_out,1,1,1,0)
        self.edge3=nn.Conv2d(mse_out,1,1,1,0)
        self.edge2=nn.Conv2d(mse_out,1,1,1,0)
        self.edge1=nn.Conv2d(mse_out,1,1,1,0)

        self.out_4=nn.Conv2d(filters[3],1,3,1,1)
        self.out_3=nn.Conv2d(filters[2],1,3,1,1)
        self.out_2=nn.Conv2d(filters[1],1,3,1,1)
        self.out_1=nn.Conv2d(filters[0],1,3,1,1)
        self.fuse=nn.Conv2d(4, 1, 1, stride=1)

        initialize_weights(self)

    def forward(self,x,train=False):
        x5,x4,x3,x2,x1=self.unet_model(x,res=False,res2=True)
        hd4=self.mse4(x4)
        hd3=self.mse3(x3)
        hd2=self.mse2(x2)
        hd1=self.mse1(x1)
        # print(hd1.size())

        edge_size=x.size()[2:]
        hd1_=F.interpolate(hd1,edge_size,mode='bilinear')
        hd2_=F.interpolate(hd2,edge_size,mode='bilinear')
        hd3_=F.interpolate(hd3,edge_size,mode='bilinear')
        hd4_=F.interpolate(hd4,edge_size,mode='bilinear')
        # print(hd1_.size(),hd2_.size(),hd3_.size(),hd4_.size())
        edge1=self.edge1(hd1_)
        edge2=self.edge2(hd2_)
        edge3=self.edge3(hd3_)
        edge4=self.edge4(hd4_)
        fuse=torch.cat((edge1,edge2,edge3,edge4),dim=1)
        fuse=self.fuse(fuse)
        # print(edge1.size(),edge2.size(),edge3.size(),edge4.size())
        if train:
            return (edge1,edge2,edge3,edge4,fuse)
        return fuse

if __name__=='__main__':
    model=ddUnet(3,1,filters=(12,14,16,18,20)).cuda()
    summary(model,(3,32,32))