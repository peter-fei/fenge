import torch
from torch.nn import Module
import torch.nn as nn
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
        x=self.conv1(x)
        y=self.conv2(x)
        return y

class UPConv(Module):
    def __init__(self,in_channels,out_channels):
        super(UPConv, self).__init__()
        self.up=nn.Upsample(scale_factor=2,mode='bilinear')
        self.conv=nn.Sequential(
            nn.Conv2d(in_channels,out_channels,3,1,1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(True)
        )
        self.conv2=DoubleConv(2*out_channels,out_channels)
    def forward(self,x):
        x=self.up(x)
        y=self.conv(x)
        # y=self.conv2(x)
        return y

class Decouple(Module):
    def __init__(self, inplane):
        super(Decouple, self).__init__()
        self.down = nn.Sequential(
            nn.Conv2d(inplane, inplane, kernel_size=3, groups=inplane, stride=2),
            nn.BatchNorm2d(inplane),
            nn.ReLU(inplace=True),
            nn.Conv2d(inplane, inplane, kernel_size=3, groups=inplane, stride=2),
            nn.BatchNorm2d(inplane),
            nn.ReLU(inplace=True)
        )
        self.flow_make = nn.Conv2d(inplane * 2, 2, kernel_size=3, padding=1, bias=False)

    def forward(self, x):
        size = x.size()[2:]
        # print(x.size())
        seg_down = self.down(x)
        # print('sss',seg_down.size())
        seg_down = F.upsample(seg_down, size=size, mode='bilinear', align_corners=True)
        flow = self.flow_make(torch.cat((x, seg_down), dim=1))
        seg_flow_warp = self.flow_warp(x, flow, size)
        seg_edge = x - seg_flow_warp
        # print(x.size(),seg_flow_warp.size(),seg_edge.size())

        return seg_flow_warp, seg_edge

    def flow_warp(self, input, flow, size):
        out_h, out_w = size
        n, c, h, w = input.size()
        norm = torch.Tensor([[[out_w, out_h]]]).type_as(input).to(input.device)
        h_grid = torch.linspace(-1, 1, out_h).reshape(-1, 1).repeat(1, out_w)
        w_gird = torch.linspace(-1.0, 1.0, out_w).repeat(out_h, 1)
        grid = torch.cat((w_gird.unsqueeze(2), h_grid.unsqueeze(2)), 2)
        grid = grid.repeat(n, 1, 1, 1).type_as(input).to(input.device)
        grid = grid + flow.permute(0, 2, 3, 1) / norm
        output = F.grid_sample(input, grid)

        return output

class D_UNet(Module):
    def __init__(self,in_channels,out_channels,filters=[64,128,256,512,1024]):
        super(D_UNet, self).__init__()
        self.down_conv1 = DoubleConv(in_channels, filters[0])
        self.down_conv2 = DoubleConv(filters[0], filters[1])
        self.down_conv3 = DoubleConv(filters[1], filters[2])
        self.down_conv4 = DoubleConv(filters[2], filters[3])
        self.down_conv5 = DoubleConv(filters[3], filters[4])
        self.maxpool = nn.MaxPool2d(2, 2)

        self.decouple1=Decouple(filters[0])
        self.decouple2=Decouple(filters[1])
        self.decouple3=Decouple(filters[2])
        self.decouple4=Decouple(filters[3])
        self.edge_conv=nn.Sequential(
            nn.Conv2d(sum(filters),filters[0],3,1,1),
            nn.BatchNorm2d(filters[0]),
            nn.ReLU(True)
        )

        self.up=nn.Upsample(scale_factor=2,mode='bilinear')
        self.up_conv5_4=UPConv(filters[4],filters[3])
        self.conv4=DoubleConv(filters[3]*2,filters[3])
        self.up_conv4_3=UPConv(filters[3],filters[2])
        self.conv3=DoubleConv(filters[2]*2,filters[2])
        self.up_conv3_2=UPConv(filters[2],filters[1])
        self.conv2=DoubleConv(filters[1]*2,filters[1])
        self.up_conv2_1=UPConv(filters[1],filters[0])
        self.conv1=nn.Sequential(
            nn.Conv2d(filters[0]*2,filters[0],3,1,1),
            nn.BatchNorm2d(filters[0]),
            nn.ReLU(True)
        )
        self.out_conv=nn.Sequential(
            nn.Conv2d(filters[0],out_channels,3,1,1)
        )

    def forward(self,x):
        h1=self.down_conv1(x)
        h2=self.down_conv2(self.maxpool(h1))
        h3=self.down_conv3(self.maxpool(h2))
        h4=self.down_conv4(self.maxpool(h3))
        h5=self.down_conv5(self.maxpool(h4))

        body1,edge1=self.decouple1(h1)
        body2,edge2=self.decouple2(h2)
        body3,edge3=self.decouple3(h3)
        body4,edge4=self.decouple4(h4)

        hd4=self.up_conv5_4(h5)
        hd4=torch.cat((body4,hd4),dim=1)
        hd4=self.conv4(hd4)
        hd3=self.up_conv4_3(hd4)
        hd3=torch.cat((body3,hd3),dim=1)
        hd3=self.conv3(hd3)
        hd2=self.up_conv3_2(hd3)
        hd2=torch.cat((body2,hd2),dim=1)
        hd2=self.conv2(hd2)
        hd1=self.up_conv2_1(hd2)
        hd1=torch.cat((body1,hd1),dim=1)
        hd1=self.conv1(hd1)

        edge_size=body1.size()[2:]
        edge2_up=F.interpolate(edge2,size=edge_size,mode='bilinear')
        edge3_up=F.interpolate(edge3,size=edge_size,mode='bilinear')
        edge4_up=F.interpolate(edge4,size=edge_size,mode='bilinear')
        body5_up=F.interpolate(h5,size=edge_size,mode='bilinear')
        edge=torch.cat((edge1,edge2_up,edge3_up,edge4_up,body5_up),dim=1)
        edge=self.edge_conv(edge)
        out=4+hd1
        out=self.out_conv(out)
        # print(edge1.size(),edge2_up.size(),edge3_up.size(),edge4_up.size(),egde5_up.size())
        return out


class D_Net2(Module):
    def __init__(self,in_channels,out_channels,filters=[64,128,256,512,1024]):
        super(D_Net2, self).__init__()
        self.down_conv1 = DoubleConv(in_channels, filters[0])
        self.down_conv2 = DoubleConv(filters[0], filters[1])
        self.down_conv3 = DoubleConv(filters[1], filters[2])
        self.down_conv4 = DoubleConv(filters[2], filters[3])
        self.down_conv5 = DoubleConv(filters[3], filters[4])
        self.maxpool = nn.MaxPool2d(2, 2)

        self.decouple1 = Decouple(filters[0])
        self.decouple2 = Decouple(filters[1])
        self.decouple3 = Decouple(filters[2])
        self.decouple4 = Decouple(filters[3])
        self.edge_conv = nn.Sequential(
            nn.Conv2d(sum(filters), filters[0], 3, 1, 1),
            nn.BatchNorm2d(filters[0]),
            nn.ReLU(True)
        )

        self.up = nn.Upsample(scale_factor=2, mode='bilinear')
        self.up_conv5_4 = UPConv(filters[4], filters[3])
        self.conv4 = DoubleConv(filters[3] * 2, filters[3])
        self.up_conv4_3 = UPConv(filters[3], filters[2])
        self.conv3 = DoubleConv(filters[2] * 2, filters[2])
        self.up_conv3_2 = UPConv(filters[2], filters[1])
        self.conv2 = DoubleConv(filters[1] * 2, filters[1])
        self.up_conv2_1 = UPConv(filters[1], filters[0])
        self.conv1 = nn.Sequential(
            nn.Conv2d(filters[0] * 2, filters[0], 3, 1, 1),
            nn.BatchNorm2d(filters[0]),
            nn.ReLU(True)
        )
        self.out_conv = nn.Sequential(
            nn.Conv2d(filters[0], out_channels, 3, 1, 1)
        )
        self.edge_out=nn.Sequential(
            nn.Conv2d(filters[0],1,1,1,0)
        )
        self.body_out=nn.Sequential(
            nn.Conv2d(filters[0],out_channels,1,1,0)
        )

    def forward(self, x,train=False):
        h1 = self.down_conv1(x)
        h2 = self.down_conv2(self.maxpool(h1))
        h3 = self.down_conv3(self.maxpool(h2))
        h4 = self.down_conv4(self.maxpool(h3))
        h5 = self.down_conv5(self.maxpool(h4))

        body1, edge1 = self.decouple1(h1)
        body2, edge2 = self.decouple2(h2)
        body3, edge3 = self.decouple3(h3)
        body4, edge4 = self.decouple4(h4)

        hd4 = self.up_conv5_4(h5)
        hd4 = torch.cat((body4, hd4), dim=1)
        hd4 = self.conv4(hd4)
        hd3 = self.up_conv4_3(hd4)
        hd3 = torch.cat((body3, hd3), dim=1)
        hd3 = self.conv3(hd3)
        hd2 = self.up_conv3_2(hd3)
        hd2 = torch.cat((body2, hd2), dim=1)
        hd2 = self.conv2(hd2)
        hd1 = self.up_conv2_1(hd2)
        hd1 = torch.cat((body1, hd1), dim=1)
        hd1 = self.conv1(hd1)

        edge_size = body1.size()[2:]
        edge2_up = F.interpolate(edge2, size=edge_size, mode='bilinear')
        edge3_up = F.interpolate(edge3, size=edge_size, mode='bilinear')
        edge4_up = F.interpolate(edge4, size=edge_size, mode='bilinear')
        body5_up = F.interpolate(h5, size=edge_size, mode='bilinear')
        edge = torch.cat((edge1, edge2_up, edge3_up, edge4_up, body5_up), dim=1)
        edge = self.edge_conv(edge)

        out = edge + hd1
        out = self.out_conv(out)
        edge_out=self.edge_out(edge)
        body_out=self.body_out(hd1)
        # print(edge1.size(),edge2_up.size(),edge3_up.size(),edge4_up.size(),egde5_up.size())
        if train:
            # print(out.shape,body_out.shape,edge_out.shape)
            return (out,body_out,edge_out)
        return out




if __name__=='__main__':
    # model=Decouple(3).cuda()
    model=D_Net2(3,1,(16,32,64,72,81)).cuda()
    summary(model,(3,512,512))


