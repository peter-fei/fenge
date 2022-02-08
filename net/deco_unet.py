import torch
import torch.nn as nn
from torch.nn import Module
from torch.nn import functional as F
from torchsummary import summary

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
        self.out_conv=nn.Conv2d(up_channel,out_channel,1,1,0)

        self.h2_conv=nn.Sequential(
            nn.Conv2d(filters[1],48,3,1,1),
            nn.BatchNorm2d(48),
            nn.ReLU(True)
        )
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
        h2=self.h2_conv(h2)
        return out,h2

class SqueezeBodyEdge(Module):
    def __init__(self,inplane):
        super(SqueezeBodyEdge, self).__init__()
        self.down = nn.Sequential(
            nn.Conv2d(inplane, inplane, kernel_size=3, groups=inplane, stride=2),
            nn.BatchNorm2d(inplane),
            nn.ReLU(inplace=True),
            nn.Conv2d(inplane, inplane, kernel_size=3, groups=inplane, stride=2),
            nn.BatchNorm2d(inplane),
            nn.ReLU(inplace=True)
        )
        self.flow_make = nn.Conv2d(inplane *2 , 2, kernel_size=3, padding=1, bias=False)

    def forward(self,x):
        size=x.size()[2:]
        # print(x.size())
        seg_down=self.down(x)
        # print('sss',seg_down.size())
        seg_down=F.upsample(seg_down,size=size,mode='bilinear',align_corners=True)
        flow=self.flow_make(torch.cat((x,seg_down),dim=1))
        seg_flow_warp=self.flow_warp(x, flow, size)
        seg_edge = x - seg_flow_warp

        return seg_flow_warp, seg_edge

    def flow_warp(self,input,flow,size):
        out_h, out_w = size
        n, c, h, w = input.size()
        norm=torch.Tensor([[[out_w,out_h]]]).type_as(input).to(input.device)
        h_grid=torch.linspace(-1,1,out_h).reshape(-1,1).repeat(1,out_w)
        w_gird = torch.linspace(-1.0, 1.0, out_w).repeat(out_h, 1)
        grid = torch.cat((w_gird.unsqueeze(2), h_grid.unsqueeze(2)), 2)
        grid = grid.repeat(n, 1, 1, 1).type_as(input).to(input.device)
        grid = grid + flow.permute(0, 2, 3, 1) / norm
        output = F.grid_sample(input, grid)

        return output


class decou_unet(Module):
    def __init__(self,in_channel,out_channel):
        super(decou_unet, self).__init__()
        self.unet=Unet_3plus(in_channel,64,filters=(12,16,18,32,64))
        self.squeeze_body_edge = SqueezeBodyEdge(64)
        # self.edge_fusion = nn.Conv2d(256 + 48, 256, 1, bias=False)
        self.sigmoid_edge = nn.Sigmoid()
        self.edge_fusion = nn.Conv2d(64 + 48, 64, 1, bias=False)
        self.edge_out = nn.Sequential(
            nn.Conv2d(64, 48, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(48),
            nn.ReLU(inplace=True),
            nn.Conv2d(48, 1, kernel_size=1, bias=False)
        )

        self.dsn_seg_body = nn.Sequential(
            nn.Conv2d(64, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, out_channel, kernel_size=1, bias=False)
        )

        self.final_seg = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, out_channel, kernel_size=1, bias=False))



    def forward(self,x):
        x_size=x.size()
        x,m2=self.unet(x)
        fine_size=m2.size()

        seg_body, seg_edge = self.squeeze_body_edge(x)

        # may add canny edge
        # canny_edge = self.edge_canny(inp, x_size)
        # add low-level feature
        dec0_fine = m2
        # print(F.interpolate(seg_edge, fine_size[2:],mode='bilinear').shape,fine_size,dec0_fine.shape)
        seg_edge = self.edge_fusion(torch.cat([F.interpolate(seg_edge, fine_size[2:],mode='bilinear'), dec0_fine], dim=1))
        seg_edge_out = self.edge_out(seg_edge)
        # print(seg_edge.shape,F.interpolate(seg_body, fine_size[2:],mode='bilinear').shape)
        seg_out = seg_edge + F.interpolate(seg_body, fine_size[2:],mode='bilinear')
        aspp = F.interpolate(x, fine_size[2:],mode='bilinear')
        # print('ssss',x.size(), fine_size,aspp.shape)
        seg_out = torch.cat([aspp, seg_out], dim=1)
        seg_final = self.final_seg(seg_out)

        seg_edge_out = F.interpolate(seg_edge_out, x_size[2:],mode='bilinear')
        seg_edge_out = self.sigmoid_edge(seg_edge_out)

        seg_final_out = F.interpolate(seg_final, x_size[2:],mode='bilinear')

        seg_body_out = F.interpolate(self.dsn_seg_body(seg_body), x_size[2:],mode='bilinear')

        return seg_final_out
if __name__=='__main__':
    net=decou_unet(3,1).cuda()
    summary(net,(3,32,32))

