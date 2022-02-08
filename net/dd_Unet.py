from net.unet3_plus import Unet_3plus
import torch
import torch.nn as nn
from torch.nn import Module
from torchsummary import summary
from torch.nn import functional as F
from net.unet import Unet
from net.modules import Atten_ASPP

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

class MSBlock(nn.Module):
    def __init__(self, c_in,c_out=21, rate=4):
        super(MSBlock, self).__init__()
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
            # nn.ReLU(True)
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

class ddUnet(Module):
    def __init__(self,in_channels,out_channels,filters=(64,128,256,512,1024)):
        super(ddUnet, self).__init__()
        mse_out=21
        up=filters[0]*5
        self.unet_model=Unet_3plus(in_channels,out_channels,filters)
        self.mse1=MSBlock(filters[0],mse_out)
        self.mse2=MSBlock(up,mse_out)
        self.mse3=MSBlock(up,mse_out)
        self.mse4=MSBlock(up,mse_out)
        self.aspp=ASPP(filters[4],filters[4])

        self.edge4=nn.Conv2d(mse_out,1,1,1,0)
        self.edge3=nn.Conv2d(mse_out,1,1,1,0)
        self.edge2=nn.Conv2d(mse_out,1,1,1,0)
        self.edge1=nn.Conv2d(mse_out,1,1,1,0)
        self.edge_out = nn.Conv2d(4, 1, 1, 1, 0)

        self.out_4=nn.Conv2d(filters[3],1,1,1,0)
        self.out_3=nn.Conv2d(filters[2],1,1,1,0)
        self.out_2=nn.Conv2d(filters[1],1,1,1,0)
        self.out_1=nn.Conv2d(filters[0],1,1,1,0)
        self.out=nn.Conv2d(4,1,1,1,0)

        self.up5_4=nn.Sequential(
            nn.ConvTranspose2d(filters[4],mse_out,2,2,bias=False),
            nn.BatchNorm2d(mse_out),
            # nn.ReLU(True)
        )
        self.conv_cat4=nn.Sequential(
            nn.Conv2d(mse_out,filters[3],3,1,1),
            nn.BatchNorm2d(filters[3]),
            nn.ReLU(True)
        )
        self.up4_3=nn.Sequential(
            nn.ConvTranspose2d(filters[3],mse_out,2,2,bias=False),
            nn.BatchNorm2d(mse_out),
            # nn.ReLU(True)
        )
        self.conv_cat3=nn.Sequential(
            nn.Conv2d(mse_out,filters[2],3,1,1),
            nn.BatchNorm2d(filters[2]),
            nn.ReLU(True)
        )
        self.up3_2=nn.Sequential(
            nn.ConvTranspose2d(filters[2],mse_out,2,2),
            nn.BatchNorm2d(mse_out),
            # nn.ReLU(True)
        )
        self.conv_cat2=nn.Sequential(
            nn.ConvTranspose2d(mse_out,filters[1],2,2),
            nn.BatchNorm2d(filters[1]),
            nn.ReLU(True)
        )
        self.up2_1=nn.Sequential(
            nn.ConvTranspose2d(filters[1],mse_out,3,1,1),
            nn.BatchNorm2d(mse_out),
            # nn.ReLU(True)
        )
        self.conv_cat1=nn.Sequential(
            nn.Conv2d(mse_out,filters[0],3,1,1),
            nn.BatchNorm2d(filters[0]),
            nn.ReLU(True)
        )
        self.edge_up2=nn.Sequential(
            nn.ConvTranspose2d(mse_out, mse_out, 2, 2),
            nn.BatchNorm2d(mse_out),
            nn.ReLU(True)
        )
        self.edge_up3=nn.Sequential(
            nn.ConvTranspose2d(mse_out,mse_out,4,4),
            nn.BatchNorm2d(mse_out),
            nn.ReLU(True)
        )
        self.edge_up4=nn.Sequential(
            nn.ConvTranspose2d(mse_out,mse_out,8,8),
            nn.BatchNorm2d(mse_out),
            nn.ReLU(True)
        )
        self.base_out=nn.Sequential(
            nn.ConvTranspose2d(filters[4],filters[4],16,16),
            nn.BatchNorm2d(filters[4]),
            nn.ReLU(True),
            nn.Conv2d(filters[4],32,3,1,1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32,out_channels,1,1,0)
        )
        self.relu=nn.ReLU(True)
        initialize_weights(self)

    def forward(self,x,train=False):
        x5,x4,x3,x2,x1=self.unet_model(x,res=False,res2=True)
        hd4=self.mse4(x4)
        hd3=self.mse3(x3)
        hd2=self.mse2(x2)
        hd1=self.mse1(x1)
        hd5=self.aspp(x5)
        s5_4=self.up5_4(hd5)
        s4=s5_4+hd4
        s4=self.conv_cat4(self.relu(s4))
        s4_3=self.up4_3(s4)
        s3=s4_3+hd3
        s3=self.conv_cat3(self.relu(s3))
        s3_2=self.up3_2(s3)
        s2=s3_2+hd2
        s2=self.conv_cat2(self.relu(s2))
        s2_1=self.up2_1(s2)
        s1=s2_1+hd1
        s1 = self.conv_cat1(self.relu(s1))

        x_size = x.size()[2:]
        s1_=F.interpolate(s1,x_size,mode='bilinear')
        s2_ = F.interpolate(s2, x_size, mode='bilinear')
        s3_ = F.interpolate(s3, x_size, mode='bilinear')
        s4_ = F.interpolate(s4, x_size, mode='bilinear')

        s4_out=self.out_4(s4_)
        s3_out=self.out_3(s3_)
        s2_out=self.out_2(s2_)
        s1_out=self.out_1(s1_)

        # print(s4_out.size(),s3_out.size(),s2_out.size(),s1_out.size())

        # hd1_=F.interpolate(hd1,x_size,mode='bilinear').detach()
        # hd2_=F.interpolate(hd2,x_size,mode='bilinear').detach()
        # hd3_=F.interpolate(hd3,x_size,mode='bilinear').detach()
        # hd4_=F.interpolate(hd4,x_size,mode='bilinear').detach()
        hd1_=hd1.detach()
        hd2_=self.edge_up2(hd2).detach()
        hd3_=self.edge_up3(hd3).detach()
        hd4_=self.edge_up4(hd4).detach()
        edge1=self.edge1(hd1_)
        edge2=self.edge2(hd2_)
        edge3=self.edge3(hd3_)
        edge4=self.edge4(hd4_)
        edge_out_=torch.cat((edge1,edge2,edge3,edge4),dim=1)
        edge_out=self.edge_out(edge_out_)

        s_out=torch.cat((s1_out,s2_out,s3_out,s4_out),dim=1)
        out=s_out+edge_out_
        out=self.out(out)
        base_out=self.base_out(hd5)
        # print(edge1.size(),edge2.size(),edge3.size(),edge4.size())
        if train:
            return out,base_out,(edge1,edge2,edge3,edge4,edge_out)
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



class ddUnet2(Module):
    def __init__(self,in_channels,out_channels,filters,device='cuda'):
        super(ddUnet2, self).__init__()
        self.unet_model=Unet(in_channels,out_channels,filters).to(device)
        mse_out=21
        self.mse1 = MSBlock(filters[0], mse_out)
        self.mse2 = MSBlock(filters[1], mse_out)
        self.mse3 = MSBlock(filters[2], mse_out)
        self.mse4 = MSBlock(filters[3], mse_out)
        self.aspp = ASPP(filters[4], filters[4])

        self.edge4 = nn.Conv2d(mse_out, 1, 1, 1, 0)
        self.edge3 = nn.Conv2d(mse_out, 1, 1, 1, 0)
        self.edge2 = nn.Conv2d(mse_out, 1, 1, 1, 0)
        self.edge1 = nn.Conv2d(mse_out, 1, 1, 1, 0)
        self.x5_out=nn.Sequential(
            nn.ConvTranspose2d(filters[4],filters[4],16,16),
            nn.BatchNorm2d(filters[4]),
            nn.ReLU(True),
            nn.Conv2d(filters[4],1,1,1,0)
        )
        self.edge_conv=nn.Sequential(
            nn.Conv2d(mse_out * 4, mse_out, 3, 1, 1),
            nn.BatchNorm2d(mse_out),
            nn.ReLU(True)
        )
        self.edge_out = nn.Conv2d(mse_out, 1, 1, 1, 0)

        # self.out_4 = nn.Conv2d(filters[3], 1, 1, 1, 0)
        # self.out_3 = nn.Conv2d(filters[2], 1, 1, 1, 0)
        # self.out_2 = nn.Conv2d(filters[1], 1, 1, 1, 0)
        # self.out_1 = nn.Conv2d(filters[0], 1, 1, 1, 0)
        # self.out = nn.Conv2d(4, 1, 1, 1, 0)

        self.up5_4 = nn.Sequential(
            nn.ConvTranspose2d(filters[4], filters[3], 2, 2, bias=False),
            nn.BatchNorm2d(filters[3]),
            nn.ReLU(True)
        )
        self.up_conv4=nn.Sequential(
            nn.Conv2d(filters[3]*2,filters[3],3,1,1),
            nn.BatchNorm2d(filters[3]),
            nn.ReLU(True)
        )
        self.edge_conv4=nn.Sequential(
            nn.Conv2d(mse_out,filters[3],3,1,1),
            nn.BatchNorm2d(filters[3]),
            nn.ReLU(True)
        )
        self.up4_3=nn.Sequential(
            nn.ConvTranspose2d(filters[3],filters[2],2,2,bias=False),
            nn.BatchNorm2d(filters[2]),
            nn.ReLU(True)
        )
        self.up_conv3=nn.Sequential(
            nn.Conv2d(filters[2]*2,filters[2],3,1,1),
            nn.BatchNorm2d(filters[2]),
            nn.ReLU(True)
        )
        self.edge_conv3=nn.Sequential(
            nn.Conv2d(mse_out,filters[2],3,1,1),
            nn.BatchNorm2d(filters[2]),
            nn.ReLU(True)
        )
        self.up3_2 = nn.Sequential(
            nn.ConvTranspose2d(filters[2], filters[1], 2, 2, bias=False),
            nn.BatchNorm2d(filters[1]),
            nn.ReLU(True)
        )
        self.up_conv2 = nn.Sequential(
            nn.Conv2d(filters[1] * 2, filters[1], 3, 1, 1),
            nn.BatchNorm2d(filters[1]),
            nn.ReLU(True)
        )
        self.edge_conv2 = nn.Sequential(
            nn.Conv2d(mse_out, filters[1], 3, 1, 1),
            nn.BatchNorm2d(filters[1]),
            nn.ReLU(True)
        )
        self.up2_1 = nn.Sequential(
            nn.ConvTranspose2d(filters[1], filters[0], 2, 2, bias=False),
            nn.BatchNorm2d(filters[0]),
            nn.ReLU(True)
        )
        self.up_conv1 = nn.Sequential(
            nn.Conv2d(filters[0] * 2, filters[0], 3, 1, 1),
            nn.BatchNorm2d(filters[0]),
            nn.ReLU(True)
        )
        self.edge_conv1 = nn.Sequential(
            nn.Conv2d(mse_out, filters[0], 3, 1, 1),
            nn.BatchNorm2d(filters[0]),
            nn.ReLU(True)
        )
        self.body_conv=nn.Sequential(
            nn.Conv2d(filters[0],filters[0],3,1,1),
            nn.BatchNorm2d(filters[0]),
            nn.ReLU(True)
        )
        self.final_edge=nn.Sequential(
            nn.Conv2d(mse_out, filters[0], 3, 1, 1),
            nn.BatchNorm2d(filters[0]),
            nn.ReLU(True)
        )
        self.final_conv=nn.Sequential(
            nn.Conv2d(filters[0],filters[0],3,1,1),
            nn.BatchNorm2d(filters[0]),
            nn.ReLU(True)
        )
        self.final_aspp=ASPP(filters[0],filters[0])
        self.final_out = nn.Conv2d(filters[0], out_channels, 1, 1, 0)

        self.edge_up2 = nn.Sequential(
            nn.ConvTranspose2d(mse_out, mse_out, 2, 2),
            nn.BatchNorm2d(mse_out),
            nn.ReLU(True)
        )
        self.edge_up3 = nn.Sequential(
            nn.ConvTranspose2d(mse_out, mse_out, 4, 4),
            nn.BatchNorm2d(mse_out),
            nn.ReLU(True)
        )
        self.edge_up4 = nn.Sequential(
            nn.ConvTranspose2d(mse_out, mse_out, 8, 8),
            nn.BatchNorm2d(mse_out),
            nn.ReLU(True)
        )
        initialize_weights(self)

    def forward(self,x,train=False):
        res=self.unet_model(x,res1=True)
        x1,x2,x3,x4,x5,hd4,hd3,hd2,hd1=res

        hd4=self.mse4(hd4)
        hd3=self.mse3(hd3)
        hd2=self.mse2(hd2)
        hd1=self.mse1(hd1)
        hd5 = self.aspp(x5)
        x5_out=self.x5_out(hd5)
        hd4_up=self.edge_up4(hd4)
        hd3_up=self.edge_up3(hd3)
        hd2_up=self.edge_up2(hd2)
        edge=torch.cat((hd4_up,hd3_up,hd2_up,hd1),dim=1)
        edge=self.edge_conv(edge)
        edge_1out=self.edge1(hd1)
        edge_2out=self.edge2(hd2_up)
        edge_3out=self.edge3(hd3_up)
        edge_4out=self.edge4(hd4_up)
        edge_out=self.edge_out(edge)

        # hd5_=hd5
        # hd4_=hd4
        # hd3_=hd3
        # hd2_=hd2
        # hd1_=hd1
        hd5_=hd5.detach()
        hd4_=hd4.detach()
        hd3_=hd3.detach()
        hd2_=hd2.detach()
        hd1_=hd1.detach()
        d5_4=self.up5_4(hd5_)
        d4=torch.cat((d5_4,x4),dim=1)
        d4=self.up_conv4(d4)
        d4_edge=self.edge_conv4(hd4_)
        d4=d4+d4_edge
        d4_3=self.up4_3(d4)
        d3=torch.cat((d4_3,x3),dim=1)
        d3=self.up_conv3(d3)
        d3_edge=self.edge_conv3(hd3_)
        d3=d3+d3_edge
        d3_2=self.up3_2(d3)
        d2=torch.cat((d3_2,x2),dim=1)
        d2=self.up_conv2(d2)
        d2_edge=self.edge_conv2(hd2_)
        d2=d2+d2_edge
        d2_1=self.up2_1(d2)
        d1=torch.cat((d2_1,x1),dim=1)
        d1=self.up_conv1(d1)
        d1_edge=self.edge_conv1(hd1_)
        d1=d1+d1_edge
        # body=self.body_conv(d1)
        # final_edge=self.final_edge(edge)
        # body=body+final_edge
        final_out=self.final_aspp(d1)
        final_out=self.final_out(final_out)
        if train:
            return final_out,x5_out,(edge_4out,edge_3out,edge_2out,edge_1out,edge_out)
        else:
            return edge_out


class ddUnet3(Module):
    def __init__(self,in_channels,out_channels,filters=(64,128,256,512,1024)):
        super(ddUnet3, self).__init__()
        mse_out=21
        up=filters[0]*5
        self.unet_model=Unet_3plus(in_channels,out_channels,filters)
        self.mse1=MSBlock(filters[0],mse_out)
        self.mse2=MSBlock(up,mse_out)
        self.mse3=MSBlock(up,mse_out)
        self.mse4=MSBlock(up,mse_out)
        self.aspp=ASPP(filters[4],filters[4])

        self.edge4=nn.Conv2d(mse_out,1,1,1,0)
        self.edge3=nn.Conv2d(mse_out,1,1,1,0)
        self.edge2=nn.Conv2d(mse_out,1,1,1,0)
        self.edge1=nn.Conv2d(mse_out,1,1,1,0)
        self.edge_out = nn.Conv2d(4, 1, 1, 1, 0)

        self.out_4=nn.Conv2d(filters[3],1,1,1,0)
        self.out_3=nn.Conv2d(filters[2],1,1,1,0)
        self.out_2=nn.Conv2d(filters[1],1,1,1,0)
        self.out_1=nn.Conv2d(filters[0],1,1,1,0)
        self.out=nn.Conv2d(4,1,1,1,0)

        self.up5_4=nn.Sequential(
            nn.ConvTranspose2d(filters[4],mse_out,2,2,bias=False),
            nn.BatchNorm2d(mse_out),
            nn.ReLU(True)
        )
        self.conv_cat4=nn.Sequential(
            nn.Conv2d(mse_out,filters[3],3,1,1),
            nn.BatchNorm2d(filters[3]),
            nn.ReLU(True)
        )
        self.up4_3=nn.Sequential(
            nn.ConvTranspose2d(filters[3],mse_out,2,2,bias=False),
            nn.BatchNorm2d(mse_out),
            nn.ReLU(True)
        )
        self.conv_cat3=nn.Sequential(
            nn.Conv2d(mse_out,filters[2],3,1,1),
            nn.BatchNorm2d(filters[2]),
            nn.ReLU(True)
        )
        self.up3_2=nn.Sequential(
            nn.ConvTranspose2d(filters[2],mse_out,2,2),
            nn.BatchNorm2d(mse_out),
            nn.ReLU(True)
        )
        self.conv_cat2=nn.Sequential(
            nn.ConvTranspose2d(mse_out,filters[1],2,2),
            nn.BatchNorm2d(filters[1]),
            nn.ReLU(True)
        )
        self.up2_1=nn.Sequential(
            nn.ConvTranspose2d(filters[1],mse_out,3,1,1),
            nn.BatchNorm2d(mse_out),
            nn.ReLU(True)
        )
        self.conv_cat1=nn.Sequential(
            nn.Conv2d(mse_out,filters[0],3,1,1),
            nn.BatchNorm2d(filters[0]),
            nn.ReLU(True)
        )
        self.edge_up2=nn.Sequential(
            nn.ConvTranspose2d(mse_out, mse_out, 2, 2),
            nn.BatchNorm2d(mse_out),
            nn.ReLU(True)
        )
        self.edge_up3=nn.Sequential(
            nn.ConvTranspose2d(mse_out,mse_out,4,4),
            nn.BatchNorm2d(mse_out),
            nn.ReLU(True)
        )
        self.edge_up4=nn.Sequential(
            nn.ConvTranspose2d(mse_out,mse_out,8,8),
            nn.BatchNorm2d(mse_out),
            nn.ReLU(True)
        )
        self.base_out=nn.Sequential(
            nn.ConvTranspose2d(filters[4],filters[4],16,16),
            nn.BatchNorm2d(filters[4]),
            nn.ReLU(True),
            nn.Conv2d(filters[4],32,3,1,1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32,out_channels,1,1,0)
        )
        self.aspp2=ASPP(mse_out*4,4)
        initialize_weights(self)

    def forward(self,x,train=False):
        x5,x4,x3,x2,x1=self.unet_model(x,res=False,res2=True)
        hd4=self.mse4(x4)
        hd3=self.mse3(x3)
        hd2=self.mse2(x2)
        hd1=self.mse1(x1)
        hd5=self.aspp(x5)
        s5_4=self.up5_4(hd5)
        s4=s5_4+hd4
        s4=self.conv_cat4(s4)
        s4_3=self.up4_3(s4)
        s3=s4_3+hd3
        s3=self.conv_cat3(s3)
        s3_2=self.up3_2(s3)
        s2=s3_2+hd2
        s2=self.conv_cat2(s2)
        s2_1=self.up2_1(s2)
        s1=s2_1+hd1
        s1 = self.conv_cat1(s1)

        x_size = x.size()[2:]
        s1_=F.interpolate(s1,x_size,mode='bilinear')
        s2_ = F.interpolate(s2, x_size, mode='bilinear')
        s3_ = F.interpolate(s3, x_size, mode='bilinear')
        s4_ = F.interpolate(s4, x_size, mode='bilinear')

        s4_out=self.out_4(s4_)
        s3_out=self.out_3(s3_)
        s2_out=self.out_2(s2_)
        s1_out=self.out_1(s1_)

        # print(s4_out.size(),s3_out.size(),s2_out.size(),s1_out.size())

        # hd1_=F.interpolate(hd1,x_size,mode='bilinear').detach()
        # hd2_=F.interpolate(hd2,x_size,mode='bilinear').detach()
        # hd3_=F.interpolate(hd3,x_size,mode='bilinear').detach()
        # hd4_=F.interpolate(hd4,x_size,mode='bilinear').detach()
        hd1_=hd1.detach()
        hd2_=self.edge_up2(hd2).detach()
        hd3_=self.edge_up3(hd3).detach()
        hd4_=self.edge_up4(hd4).detach()
        edge1=self.edge1(hd1_)
        edge2=self.edge2(hd2_)
        edge3=self.edge3(hd3_)
        edge4=self.edge4(hd4_)
        edge_out_=torch.cat((edge1,edge2,edge3,edge4),dim=1)
        edge_out=self.edge_out(edge_out_)
        s_out=torch.cat((hd1_,hd2_,hd3_,hd4_),dim=1)
        s_out=self.aspp2(s_out)
        out=s_out+edge_out_
        out=self.out(out)
        base_out=self.base_out(hd5)
        # print(edge1.size(),edge2.size(),edge3.size(),edge4.size())
        if train:
            return out,base_out,(edge1,edge2,edge3,edge4,edge_out)
        return out





class ddUnet4(Module):
    def __init__(self,in_channels,out_channels,filters=(64,128,256,512,1024)):
        super(ddUnet4, self).__init__()
        mse_out=21
        # up=filters[0]*5
        self.unet_model=Unet(in_channels,out_channels,filters)
        self.mse1=Atten_ASPP(filters[0],mse_out)
        self.mse2=Atten_ASPP(filters[1],mse_out)
        self.mse3=Atten_ASPP(filters[2],mse_out)
        self.mse4=Atten_ASPP(filters[3],mse_out)
        self.aspp=ASPP(filters[4],filters[4])

        self.edge4=nn.Conv2d(mse_out,1,1,1,0)
        self.edge3=nn.Conv2d(mse_out,1,1,1,0)
        self.edge2=nn.Conv2d(mse_out,1,1,1,0)
        self.edge1=nn.Conv2d(mse_out,1,1,1,0)
        self.edge_out = nn.Conv2d(4, 1, 1, 1, 0)

        self.out_4=nn.Conv2d(filters[3],1,1,1,0)
        self.out_3=nn.Conv2d(filters[2],1,1,1,0)
        self.out_2=nn.Conv2d(filters[1],1,1,1,0)
        self.out_1=nn.Sequential(
            nn.Conv2d(filters[0], 1, 1, 1, 0),
            nn.BatchNorm2d(1),
            nn.ReLU(True),
            nn.Conv2d(1,1,3,1,1)
        )

        self.up5_4=nn.Sequential(
            nn.ConvTranspose2d(filters[4],mse_out,2,2,bias=False),
            nn.BatchNorm2d(mse_out),
            # nn.ReLU(True)
        )
        self.conv_cat4=nn.Sequential(
            nn.Conv2d(mse_out,filters[3],3,1,1),
            nn.BatchNorm2d(filters[3]),
            nn.ReLU(True)
        )
        self.up4_3=nn.Sequential(
            nn.ConvTranspose2d(filters[3],mse_out,2,2,bias=False),
            nn.BatchNorm2d(mse_out),
            # nn.ReLU(True)
        )
        self.conv_cat3=nn.Sequential(
            nn.Conv2d(mse_out,filters[2],3,1,1),
            nn.BatchNorm2d(filters[2]),
            nn.ReLU(True)
        )
        self.up3_2=nn.Sequential(
            nn.ConvTranspose2d(filters[2],mse_out,2,2),
            nn.BatchNorm2d(mse_out),
            # nn.ReLU(True)
        )
        self.conv_cat2=nn.Sequential(
            nn.ConvTranspose2d(mse_out,filters[1],2,2),
            nn.BatchNorm2d(filters[1]),
            nn.ReLU(True)
        )
        self.up2_1=nn.Sequential(
            nn.ConvTranspose2d(filters[1],mse_out,3,1,1),
            nn.BatchNorm2d(mse_out),
            # nn.ReLU(True)
        )
        self.conv_cat1=nn.Sequential(
            nn.Conv2d(mse_out,filters[0],3,1,1),
            nn.BatchNorm2d(filters[0]),
            nn.ReLU(True)
        )
        self.edge_up2=nn.Sequential(
            nn.ConvTranspose2d(mse_out, mse_out, 2, 2),
            nn.BatchNorm2d(mse_out),
            nn.ReLU(True)
        )
        self.edge_up3=nn.Sequential(
            nn.ConvTranspose2d(mse_out,mse_out,4,4),
            nn.BatchNorm2d(mse_out),
            nn.ReLU(True)
        )
        self.edge_up4=nn.Sequential(
            nn.ConvTranspose2d(mse_out,mse_out,8,8),
            nn.BatchNorm2d(mse_out),
            nn.ReLU(True)
        )
        self.base_out=nn.Sequential(
            nn.ConvTranspose2d(filters[4],filters[4],16,16),
            nn.BatchNorm2d(filters[4]),
            nn.ReLU(True),
            nn.Conv2d(filters[4],out_channels,1,1,0),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(True),
            nn.Conv2d(out_channels,out_channels,3,1,1)
        )
        self.x1_out=nn.Sequential(
            nn.Conv2d(filters[0],out_channels,1,1,0),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(True),
            nn.Conv2d(out_channels,out_channels,3,1,1)
        )
        self.relu=nn.ReLU(True)
        self.out=nn.Sequential(
            nn.Conv2d(2,out_channels,1,1,0),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(True),
            nn.Conv2d(out_channels,out_channels,3,1,1)
        )
        self.edge_out=nn.Sequential(
            nn.Conv2d(4,out_channels,1,1,0),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(True),
            nn.Conv2d(out_channels,out_channels,3,1,1)
        )
        initialize_weights(self)

    def forward(self,x,train=False):
        _,_,_,_,x5,x4,x3,x2,x1=self.unet_model(x,res1=True)
        hd4=self.mse4(x4)
        hd3=self.mse3(x3)
        hd2=self.mse2(x2)
        hd1=self.mse1(x1)
        hd5=self.aspp(x5)
        s5_4=self.up5_4(hd5)
        s4=self.relu(s5_4+hd4)
        s4=self.conv_cat4(s4)
        s4_3=self.up4_3(s4)
        s3=self.relu(s4_3+hd3)
        s3=self.conv_cat3(s3)
        s3_2=self.up3_2(s3)
        s2=self.relu(s3_2+hd2)
        s2=self.conv_cat2(s2)
        s2_1=self.up2_1(s2)
        s1=self.relu(s2_1+hd1)
        s1 = self.conv_cat1(s1)

        x_size = x.size()[2:]
        s1_=F.interpolate(s1,x_size,mode='bilinear')
        s2_ = F.interpolate(s2, x_size, mode='bilinear')
        s3_ = F.interpolate(s3, x_size, mode='bilinear')
        s4_ = F.interpolate(s4, x_size, mode='bilinear')

        s4_out=self.out_4(s4_)
        s3_out=self.out_3(s3_)
        s2_out=self.out_2(s2_)
        s1_out=self.out_1(s1_)

        hd1_=hd1.detach()
        hd2_=self.edge_up2(hd2).detach()
        hd3_=self.edge_up3(hd3).detach()
        hd4_=self.edge_up4(hd4).detach()
        edge1=self.edge1(hd1_)
        edge2=self.edge2(hd2_)
        edge3=self.edge3(hd3_)
        edge4=self.edge4(hd4_)
        # edge_out_=torch.cat((edge1,edge2,edge3,edge4),dim=1)
        # edge_out=self.edge_out(edge_out_)

        # s_out=torch.cat((s1_out,s2_out,s3_out,s4_out),dim=1)
        # out=s_out+edge_out_
        # out=self.out(out)
        base_out=self.base_out(hd5)
        # x1_out=self.x1_out(x1)
        out=torch.cat((s1_out,base_out),dim=1)
        out=self.out(out)
        edge_out=self.edge_out(torch.cat((edge1,edge2,edge3,edge4),dim=1))
        # print(edge1.size(),edge2.size(),edge3.size(),edge4.size())
        if train:
            return (edge1,edge2,edge3,edge4,edge_out)
        return edge_out

if __name__=='__main__':
    model=ddUnet4(3,1,(16,16,32,64,96)).cuda()
    summary(model,(3,32,32))