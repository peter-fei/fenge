import torch
import torch.nn as nn
from torch.nn import Module,functional as F
from torchsummary import summary
from net.unet3_plus import Unet_3plus
from net.edge_net import EDGE_Net
from net.bdsn import BDCN

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
            nn.Conv2d(len(rates)*out_channels+in_channels,out_channels,1,1,0),
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

class Down(Module):
    def __init__(self,in_channels,out_channels,down_num):
        super(Down, self).__init__()
        self.conv=nn.Sequential(
            nn.MaxPool2d(down_num, down_num),
            nn.Conv2d(in_channels,out_channels,1,1,0),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(True)
        )
    def forward(self,x):
        x=self.conv(x)
        return x

class UP(Module):
    def __init__(self,in_channels,out_channels,up_num):
        super(UP, self).__init__()
        self.conv=nn.Sequential(
            nn.Upsample(scale_factor=up_num,mode='bilinear'),
            nn.Conv2d(in_channels,out_channels,3,1,1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(True)
        )
    def forward(self,x):
        x=self.conv(x)
        return x

class Doubel_Net(Module):
    def __init__(self,in_channels=3,out_channels=1,filters1=(64,128,256,512,1024),filters2=(64,128,256,512,1024),filters_edge=(32,64,96),device='cuda',edge_save_path='checkpoints/save_edge_black.pth.tar'):
        super(Doubel_Net, self).__init__()
        self.unet1=Unet_3plus(in_channels,1,filters=filters1).to(device)
        self.aspp=ASPP(filters1[0],64)
        self.unet2=Unet_3plus(64,out_channels,filters=filters2).to(device)
        self.edge_net=EDGE_Net(1,1)
        # self.edge_net.load_state_dict(torch.load(edge_save_path)['state_dict'])
        self.edge_conv1=nn.Sequential(
            nn.Conv2d(filters_edge[0], filters1[0], 3, 1, 1),
            nn.BatchNorm2d(filters1[0]),
            nn.ReLU(True)
        )
        self.edge_conv2=nn.Sequential(
            nn.Conv2d(filters_edge[0],filters2[0],3,1,1),
            nn.BatchNorm2d(filters2[0]),
            nn.ReLU(True)
        )
        self.final_conv=nn.Sequential(
            nn.Conv2d(filters2[0]+filters1[0],filters2[0],3,1,1),
            nn.BatchNorm2d(filters2[0]),
            nn.ReLU(True)
        )
        # self.edge_out=nn.Conv2d(filters_edge[0],1,3,1,1)
        self.body_out=nn.Conv2d(filters1[0],out_channels,3,1,1)
        self.out_conv=nn.Conv2d(filters2[0],out_channels,3,1,1)
        self.x_out=nn.Sequential(
            nn.Conv2d(in_channels,filters1[0],3,1,1),
            nn.BatchNorm2d(filters1[0]),
            nn.ReLU(True)
        )

    def forward(self,x,train=False):
        x1=self.unet1(x,res=False)
        x_out=self.x_out(x)
        x_body = self.body_out(x1)
        x1_1=(torch.sigmoid(x_body)>0.5).float()
        x_edge0,edge_out = self.edge_net(x1_1,res=False)
        x_edge1=self.edge_conv1(x_edge0)
        # edge_out=self.edge_out(x_edge0)
        x1=x_out*(x1-x_edge1)
        x2=self.aspp(x1)
        x2=self.unet2(x2,res=False)
        x_edge=F.interpolate(x_edge0,x2.size()[2:],mode='bilinear')
        x_edge=self.edge_conv2(x_edge)
        # print(x_edge.size(),x2.size())
        x_final=x_edge+x2
        x_final=torch.cat((x_final,x1),dim=1)
        out=self.final_conv(x_final)
        out=self.out_conv(out)
        # print(x_edge.size(),x2.size())
        # out=x_edge+x2
        if train:
            return out,x_body,edge_out
        return out

class DD_Net(Module):
    def __init__(self, in_channels=3, out_channels=1, filters1=(64, 128, 256, 512, 1024),
                 filters2=(64, 128, 256, 512, 1024), filters_edge=(32, 64, 96), device='cuda',
                 edge_save_path='checkpoints/save_edge_black.pth.tar'):
        super(DD_Net, self).__init__()
        self.unet1 = Unet_3plus(in_channels, 1, filters=filters1).to(device)
        self.edge_net = EDGE_Net(1, 1)
        self.edge_out = nn.Conv2d(filters_edge[0], 1, 3, 1, 1)
        self.body_out = nn.Conv2d(filters1[0], out_channels, 3, 1, 1)
    def forward(self, x, train=False):
        x1=self.unet1(x,res=False)
        x_out=self.body_out(x1)
        x_1=(torch.sigmoid(x_out)>0.5).float()
        edge,out=self.edge_net(x_1,res=False)
        x_edge=self.edge_out(edge)
        if train:
            return x_out,x_edge
        return x_out


class DU_Net(Module):
    def __init__(self, in_channels=3, out_channels=1, filters1=(64, 128, 256, 512, 1024), filters_edge=(32, 64, 96), device='cuda',):
        super(DU_Net, self).__init__()
        self.unet1 = Unet_3plus(in_channels, 1, filters=filters1).to(device)
        down_chanel=filters1[0]*5
        self.edge_1=nn.Sequential(
            nn.Conv2d(filters_edge[0],filters1[0],1,1,0),
            nn.BatchNorm2d(filters1[0]),
            nn.ReLU(True)
        )
        self.down1_2=Down(filters_edge[0],down_chanel,2)
        self.down1_3=Down(filters_edge[0],down_chanel,4)
        self.down1_4=Down(filters_edge[0],down_chanel,8)
        self.up5_1=UP(filters1[4],filters1[0],up_num=16)
        self.up4_1=UP(down_chanel,filters1[0],up_num=8)
        self.up3_1=UP(down_chanel,filters1[0],up_num=4)
        self.up2_1=UP(down_chanel,filters1[0],up_num=2)
        self.conv1_1=nn.Sequential(
            nn.Conv2d(filters1[0],filters1[0],3,1,1),
            nn.BatchNorm2d(filters1[0]),
            nn.ReLU(True)
        )
        # self.down1_5=Down(filters_edge[0],filters1[4],16)
        self.edge_net = EDGE_Net(filters1[0], 1,filters=filters_edge)
        self.aspp=ASPP(filters1[0]*5,64)
        self.body_out = nn.Conv2d(64, out_channels, 3, 1, 1)
        self.edge_conv=nn.Sequential(
            nn.Conv2d(filters_edge[0],64,3,1,1),
            nn.BatchNorm2d(64),
            nn.ReLU(True)
        )
        self.final_conv=nn.Sequential(
            nn.Conv2d(64,64,3,1,1),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.Conv2d(64,out_channels,1,1,0)
        )
        self.body_out=nn.Sequential(
            nn.Conv2d(filters1[0], 64, 3, 1, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.Conv2d(64, out_channels, 1, 1, 0)
        )
        initialize_weights(self)

    def forward(self, x, train=False):
        x5,x4,x3,x2,x1=self.unet1(x,res=False,res2=True)
        x_=(torch.sigmoid(x1)>0.5).float()
        edge,edge_out=self.edge_net(x_,res=False)
        # edge=torch.sigmoid(edge)
        # edge=(edge>0.5).float()
        edge_1=self.edge_1(edge)
        edge_2=self.down1_2(edge)
        edge_3=self.down1_3(edge)
        edge_4=self.down1_4(edge)
        x1=x1-edge_1
        x2=x2-edge_2
        x3=x3-edge_3
        x4=x4-edge_4
        x5_1=self.up5_1(x5)
        x4_1=self.up4_1(x4)
        x3_1=self.up3_1(x3)
        x2_1=self.up2_1(x2)
        x1_1=self.conv1_1(x1)
        x_body=torch.cat((x1_1,x2_1,x3_1,x4_1,x5_1),dim=1)
        # x_body=x1_1+x2_1+x3_1+x4_1+x5_1
        x_body=self.aspp(x_body)
        body_out=self.body_out(x1)
        x_edge=self.edge_conv(edge)
        final=x_body+x_edge
        out=self.final_conv(final)
        if train:
            return out,body_out,edge_out
        return out


class DB_Net(Module):
    def __init__(self, in_channels=3, out_channels=1, filters1=(64, 128, 256, 512, 1024), filters_edge=(32, 64, 96), device='cuda',):
        super(DB_Net, self).__init__()
        self.unet1 = Unet_3plus(in_channels, 1, filters=filters1).to(device)
        down_chanel=filters1[0]*5
        self.edge_1=nn.Sequential(
            nn.Conv2d(1,filters1[0],1,1,0),
            nn.BatchNorm2d(filters1[0]),
            nn.ReLU(True)
        )
        self.down1_2=Down(1,down_chanel,2)
        self.down1_3=Down(1,down_chanel,4)
        self.down1_4=Down(1,down_chanel,8)
        self.up5_1=UP(filters1[4],filters1[0],up_num=16)
        self.up4_1=UP(down_chanel,filters1[0],up_num=8)
        self.up3_1=UP(down_chanel,filters1[0],up_num=4)
        self.up2_1=UP(down_chanel,filters1[0],up_num=2)
        self.conv1_1=nn.Sequential(
            nn.Conv2d(filters1[0],filters1[0],3,1,1),
            nn.BatchNorm2d(filters1[0]),
            nn.ReLU(True)
        )
        # self.down1_5=Down(filters_edge[0],filters1[4],16)
        self.edge_net = BDCN(filters1[0])
        self.aspp=ASPP(filters1[0]*5,64)
        self.body_out = nn.Conv2d(64, out_channels, 3, 1, 1)
        self.edge_conv=nn.Sequential(
            nn.Conv2d(1,64,3,1,1),
            nn.BatchNorm2d(64),
            nn.ReLU(True)
        )
        self.final_conv=nn.Sequential(
            nn.Conv2d(64,64,3,1,1),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.Conv2d(64,out_channels,1,1,0)
        )
        self.body_out=nn.Sequential(
            nn.Conv2d(64, 64, 3, 1, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.Conv2d(64, out_channels, 1, 1, 0)
        )


    def forward(self, x, train=False):
        x5,x4,x3,x2,x1=self.unet1(x,res=False,res2=True)
        # edge,edge_out=self.edge_net(x1,res=False)
        x1_=(torch.sigmoid(x1)>0).float()
        features = self.edge_net(x1)
        edge=self.edge_net.res
        edge_1=self.edge_1(edge)
        edge_2=self.down1_2(edge)
        edge_3=self.down1_3(edge)
        edge_4=self.down1_4(edge)
        x1=x1-edge_1
        x2=x2-edge_2
        x3=x3-edge_3
        x4=x4-edge_4
        x5_1=self.up5_1(x5)
        x4_1=self.up4_1(x4)
        x3_1=self.up3_1(x3)
        x2_1=self.up2_1(x2)
        x1_1=self.conv1_1(x1)
        x_body=torch.cat((x1_1,x2_1,x3_1,x4_1,x5_1),dim=1)
        x_body=self.aspp(x_body)
        body_out=self.body_out(x_body)
        x_edge=self.edge_conv(edge)
        final=x_body+x_edge
        out=self.final_conv(final)
        if train:
            return out,body_out,features
        return out

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
if __name__=='__main__':
    model=DU_Net(3,1,(2,4,8,16,32)).cuda()
    summary(model,(3,64,64))
