import torch
import torch.nn as nn
from torch.nn import Module
from torchsummary import summary
from torch.nn import functional as F
from net.unet import Unet


class PreActBlock(nn.Module):
    def __init__(self, in_planes, planes, stride=1):
        super(PreActBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)

        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride, bias=False)
            )

        # SE layers
        self.fc1 = nn.Conv2d(planes, planes//4, kernel_size=1)
        self.fc2 = nn.Conv2d(planes//4, planes, kernel_size=1)

    def forward(self, x):
        out = F.relu(self.bn1(x))
        shortcut = self.shortcut(out) if hasattr(self, 'shortcut') else x
        out = self.conv1(out)
        out = self.conv2(F.relu(self.bn2(out)))

        # Squeeze
        w = F.avg_pool2d(out, out.size(2))
        w = F.relu(self.fc1(w))
        w = F.sigmoid(self.fc2(w))
        # Excitation
        out = out * w

        # out += shortcut
        return out

class Atten_ASPP(Module):
    def __init__(self,in_channels,out_channels,rates=(6,12,18)):
        super(Atten_ASPP, self).__init__()
        self.conv_list=nn.ModuleList()
        for r in rates:
            conv_module=nn.Sequential(
                nn.Conv2d(in_channels,in_channels,3,1,padding=r,dilation=r),
                nn.BatchNorm2d(in_channels),
                nn.ReLU(True),
                PreActBlock(in_channels,in_channels)
            )
            self.conv_list.append(conv_module)
            self.relu=nn.ReLU(True)
            self.out_conv=nn.Sequential(
                nn.Conv2d(in_channels,out_channels,3,1,1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(True)
            )
    def forward(self,x):
        res=[]
        for module in self.conv_list:
            res.append(module(x))
        y=sum(res)
        y=self.relu(y+x)
        out=self.out_conv(y)
        return out







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
            # nn.ReLU(True)
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
        seg_down=self.down(x)
        seg_down=F.upsample(seg_down,size=size,mode='bilinear',align_corners=True)
        flow=self.flow_make(torch.cat((x,seg_down),dim=1))
        seg_flow_warp=self.flow_warp(x, flow, size)

        return seg_flow_warp

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

class UpConvBlock(nn.Module):
    def __init__(self, in_features, up_scale):
        super(UpConvBlock, self).__init__()
        self.up_factor = 2
        self.constant_features = 16

        layers = self.make_deconv_layers(in_features, up_scale)
        assert layers is not None, layers
        self.features = nn.Sequential(*layers)

    def make_deconv_layers(self, in_features, up_scale):
        layers = []
        all_pads=[0,0,1,3,7]
        for i in range(up_scale):
            kernel_size = 2 ** up_scale
            pad = all_pads[up_scale]  # kernel_size-1
            out_features = self.compute_out_features(i, up_scale)
            layers.append(nn.Conv2d(in_features, out_features, 1))
            layers.append(nn.ReLU(inplace=True))
            layers.append(nn.ConvTranspose2d(
                out_features, out_features, kernel_size, stride=2, padding=pad))
            in_features = out_features
        return layers

    def compute_out_features(self, idx, up_scale):
        return 1 if idx == up_scale - 1 else self.constant_features

    def forward(self, x):
        return self.features(x)


class DE_UNet(Module):
    def __init__(self,in_channels,out_channels,filters):
        super(DE_UNet, self).__init__()
        unet_model = Unet(in_channels, out_channels, filters)
        self.conv1 = unet_model.conv1
        self.conv2 = unet_model.conv2
        self.conv3 = unet_model.conv3
        self.conv4 = unet_model.conv4
        # self.conv5=unet_model.conv5
        self.up5 = unet_model.up5
        self.up4 = unet_model.up4
        self.up3 = unet_model.up3
        self.up2 = unet_model.up2
        self.up_conv4 = unet_model.up_conv4
        self.up_conv3 = unet_model.up_conv3
        self.up_conv2 = unet_model.up_conv2
        self.up_conv1 = unet_model.up_conv1
        self.maxpool=unet_model.maxpool
        self.relu=nn.ReLU()

        mse_channels=21
        self.aspp = ASPP(filters[3], filters[4])
        self.msb1 = MSBlock(filters[0],filters[0])
        self.msb2 = MSBlock(filters[1],filters[1])
        self.msb3 = MSBlock(filters[2],filters[2])
        self.msb4 = MSBlock(filters[3],filters[3])

        self.edge1_conv=nn.Sequential(
            nn.Conv2d(filters[0],1,3,1,1),
            nn.ReLU(True),
            nn.Conv2d(1,1,3,1,1),
            nn.ReLU(True)
        )
        self.edge_up2=UpConvBlock(filters[1],1)
        self.edge_up3=UpConvBlock(filters[2],2)
        self.edge_up4=UpConvBlock(filters[3],3)
        self.base_out=UpConvBlock(filters[4],4)
        self.edge_out=nn.Conv2d(5,1,1,1,0)

        self.out=nn.Sequential(
            nn.Conv2d(filters[0],filters[0],3,1,1),
            nn.BatchNorm2d(filters[0]),
            nn.ReLU(True),
            nn.Conv2d(filters[0],out_channels,1,1,0)
        )
        # initialize_weights(self)

    def forward(self,x,train=False):
        x1=self.conv1(x)
        edge1=self.msb1(x1)
        x1_=x1-edge1
        x1_=self.relu(x1_)
        p1=self.maxpool(x1_)
        x2=self.conv2(p1)
        edge2=self.msb2(x2)
        x2_=self.relu(x2-edge2)
        p2=self.maxpool(x2_)
        x3=self.conv3(p2)
        edge3=self.msb3(x3)
        x3_=self.relu(x3-edge3)
        p3=self.maxpool(x3_)
        x4=self.conv4(p3)
        edge4=self.msb4(x4)
        x4_=self.relu(x4-edge4)
        p4=self.maxpool(x4_)
        x5=self.aspp(p4)
        edge1_=self.relu(edge1)
        edge2_=self.relu(edge2)
        edge3_=self.relu(edge3)
        edge4_=self.relu(edge4)
        edge1_out=self.edge1_conv(edge1_)
        edge2_out=self.edge_up2(edge2_)
        edge3_out=self.edge_up3(edge3_)
        edge4_out=self.edge_up4(edge4_)
        base_out=self.base_out(x5)
        edge_out=torch.cat((edge1_out,edge2_out,edge3_out,edge4_out,base_out),dim=1)
        edge_out=self.edge_out(edge_out)
        # print(edge1_out.size(),edge2_out.size(),edge3_out.size(),edge4_out.size(),edge_out.size())

        h5_d4=self.relu(self.up5(x5)+edge4)
        hd4=torch.cat((h5_d4,x4_),dim=1)
        hd4=self.up_conv4(hd4)
        h4_d3=self.relu(self.up4(hd4)+edge3)
        hd3=torch.cat((h4_d3,x3_),dim=1)
        hd3=self.up_conv3(hd3)
        h3_d2=self.relu(self.up3(hd3)+edge2)
        hd2=torch.cat((h3_d2,x2_),dim=1)
        hd2=self.up_conv2(hd2)
        h2_d1=self.relu(self.up2(hd2)+edge1)
        hd1=torch.cat((h2_d1,x1_),dim=1)
        hd1=self.up_conv1(hd1)
        out=self.out(hd1)

        if train:
            return out,(edge1_out,edge2_out,edge3_out,edge4_out,edge_out)
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


class UpConv(Module):
    def __init__(self,in_channels,out_channels):
        super(UpConv, self).__init__()
        self.up_conv=nn.Sequential(
            nn.ConvTranspose2d(in_channels,out_channels,2,2),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(True)
        )
    def forward(self,x):
        x=self.up_conv(x)
        return x

class CatConv(Module):
    def __init__(self,in_channels,out_channels):
        super(CatConv, self).__init__()
        self.conv=nn.Sequential(
            nn.Conv2d(in_channels,out_channels,3,1,1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(True)
        )
    def forward(self,x):
        x=self.conv(x)
        return x

class DcoUnet(Module):
    def __init__(self,in_channels,out_channels,filters=(64,128,256,512,1024)):
        super(DcoUnet, self).__init__()
        self.unet_model=Unet(in_channels,out_channels,filters)
        self.aspp=ASPP(filters[0],filters[0])
        ms_out=21

        self.maxpool=nn.MaxPool2d(2,2)
        self.relu=nn.ReLU()
        # self.msb2=MSBlock(filters[0],filters[1])
        self.msb1_2=MSBlock(filters[0],filters[1])
        self.msb2_3=MSBlock(filters[1],filters[2])
        self.msb3_4=MSBlock(filters[2],filters[3])
        self.aspp5=ASPP(filters[3],filters[4])

        self.up5=UpConv(filters[4]*2,filters[3])
        self.up4=UpConv(filters[3],filters[2])
        self.up3=UpConv(filters[2],filters[1])
        self.up2=UpConv(filters[1],filters[0])
        self.cat_conv4=CatConv(filters[3]*2,filters[3])
        self.cat_conv3=CatConv(filters[2]*2,filters[2])
        self.cat_conv2=CatConv(filters[1]*2,filters[1])
        self.cat_conv1=CatConv(filters[0]*2,filters[0])

        self.edge_out1=nn.Conv2d(filters[0],out_channels,1,1,0)
        self.edge_out2=UpConvBlock(filters[1],1)
        self.edge_out3=UpConvBlock(filters[2],2)
        self.edge_out4=UpConvBlock(filters[3],3)
        self.edge_out=nn.Conv2d(filters[0],out_channels,1,1,0)
        self.body_out=nn.Sequential(
            nn.Conv2d(filters[0] , out_channels, 1, 1, 0),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(True),
            nn.Conv2d(out_channels,out_channels,3,1,1)
        )

        self.out=nn.Sequential(
            nn.Conv2d(filters[0]+filters[4], out_channels, 3, 1, 1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(True),
            nn.Conv2d(out_channels,out_channels,3,1,1)
        )

        self.squeeze_body=SqueezeBodyEdge(filters[0])
        self.edge_conv=nn.Sequential(
            nn.Conv2d(sum(filters)-filters[4],filters[0],3,1,1),
            nn.BatchNorm2d(filters[0]),
        )
        self.body_conv=nn.Sequential(
            nn.Conv2d(filters[0]*2,filters[0],3,1,1),
            nn.BatchNorm2d(filters[0]),
        )
        self.sigmoid=nn.Sigmoid()
        initialize_weights(self)

    def forward(self,x,train=False):
        outs=self.unet_model(x,res1=True)
        x1,x2,x3,x4,x5,hd4,hd3,hd2,hd1=outs
        hd1_aspp=self.aspp(hd1)
        hd1_aspp_pool=self.maxpool(hd1_aspp)
        ms2=self.msb1_2(hd1_aspp_pool)
        ms2_=self.relu(ms2+hd2)
        # ms2_=self.relu(self.sigmoid(hd2)*ms2)
        ms2_pool=self.maxpool(ms2_)
        ms3=self.msb2_3(ms2_pool)
        ms3_=self.relu(ms3+hd3)
        # ms3_=self.relu(self.sigmoid(hd3)*ms3)
        ms3_pool=self.maxpool(ms3_)
        ms4=self.msb3_4(ms3_pool)
        ms4_=self.relu(ms4+hd4)
        # ms4_=self.relu(self.sigmoid(hd4)*ms4)
        md4_pool=self.maxpool(ms4_)
        x5_aspp=self.aspp5(md4_pool)
        x5_=torch.cat((self.relu(x5_aspp),x5),dim=1)

        h5_4=self.up5(x5_)
        h4=self.cat_conv4(torch.cat((h5_4,ms4_),dim=1))
        h4_3=self.up4(h4)
        h3=self.cat_conv3(torch.cat((h4_3,ms3_),dim=1))
        h3_2=self.up3(h3)
        h2=self.cat_conv2(torch.cat((h3_2,ms2_),dim=1))
        h2_1=self.up2(h2)
        h1=self.cat_conv1(torch.cat((h2_1,self.relu(hd1_aspp)),dim=1))
        # print(h1.size(),h2.size(),h3.size(),h4.size())
        edge_out1=self.edge_out1(h1)
        edge_out2=self.edge_out2(h2)
        edge_out3=self.edge_out3(h3)
        edge_out4=self.edge_out4(h4)

        body = self.squeeze_body(hd1_aspp)
        body=self.body_conv(torch.cat((body,hd1_aspp),dim=1))
        body_out=self.body_out(body)
        x_size=x.size()[2:]
        h2_up=F.interpolate(h2,x_size,mode='bilinear')
        h3_up=F.interpolate(h3,x_size,mode='bilinear')
        h4_up=F.interpolate(h4,x_size,mode='bilinear')
        h5_up=F.interpolate(x5,x_size,mode='bilinear')
        edge=torch.cat((h1,h2_up,h3_up,h4_up),dim=1)
        edge=self.edge_conv(edge)
        edge_=self.relu(edge)
        edge_out=self.edge_out(edge_)
        final_out=self.out(torch.cat((body+edge_,h5_up),dim=1))
        if train:
            return final_out,body_out,edge_out

        return final_out


class DC_UNet(Module):
    def __init__(self,in_channels,out_channels,filters=(64,128,256,512,1024)):
        super(DC_UNet, self).__init__()
        self.unet_model = Unet(in_channels, out_channels, filters)
        self.aspp = ASPP(filters[0], filters[0])
        ms_out = 21

        self.maxpool = nn.MaxPool2d(2, 2)
        self.relu = nn.ReLU()
        self.att1=Atten_ASPP(filters[0],filters[0])
        self.att2=Atten_ASPP(filters[1],filters[1])
        self.att3=Atten_ASPP(filters[2],filters[2])
        self.att4=Atten_ASPP(filters[3],filters[3])
        self.att5=Atten_ASPP(filters[4],filters[4])
        self.edge1=nn.Sequential(
            nn.Conv2d(filters[0],out_channels,3,1,1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(True),
            nn.Conv2d(out_channels,out_channels,3,1,1)
        )
        self.edge2=nn.Sequential(
            nn.ConvTranspose2d(filters[1],out_channels,2,2),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(True),
            nn.Conv2d(out_channels,out_channels,3,1,1)

        )
        self.edge3=nn.Sequential(
            nn.ConvTranspose2d(filters[2],out_channels,4,4),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(True),
            nn.Conv2d(out_channels,out_channels,3,1,1)
        )
        self.edge4=nn.Sequential(
            nn.ConvTranspose2d(filters[3],out_channels,8,8),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(True),
            nn.Conv2d(out_channels,out_channels,3,1,1)
        )
        self.base=nn.Sequential(
            nn.ConvTranspose2d(filters[4], out_channels, 16, 16),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(True),
            nn.Conv2d(out_channels, out_channels, 3, 1, 1)
        )
        self.edge=nn.Sequential(
            nn.Conv2d(4,out_channels,1,1,0),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(True),
            nn.Conv2d(out_channels,out_channels,3,1,1)
        )

        self.sigmoid = nn.Sigmoid()
        initialize_weights(self)

    def forward(self,x,train=False):
        outs = self.unet_model(x, res1=True)
        x1, x2, x3, x4, x5, hd4, hd3, hd2, hd1 = outs
        h1=self.att1(hd1)
        h2=self.att2(hd2)
        h3=self.att3(hd3)
        h4=self.att4(hd4)
        x5=self.att5(x5)

        edge1=self.edge1(h1)
        edge2=self.edge2(h2)
        edge3=self.edge3(h3)
        edge4=self.edge4(h4)
        base_5=self.base(x5)
        edge=torch.cat((edge1,edge2,edge3,edge4),dim=1)
        edge=self.edge(edge)
        if train:
            return (edge1,edge2,edge3,edge4,edge)
        else:
            return edge


class DCC_UNet(Module):
    def __init__(self,in_channels,out_channels,filters=(64,128,256,512,1024)):
        super(DCC_UNet, self).__init__()
        self.unet_model = Unet(in_channels, out_channels, filters)
        self.edge_model=Edge_model(filters)
        self.aspp = ASPP(filters[0], filters[0])
        ms_out = 21

        self.maxpool = nn.MaxPool2d(2, 2)
        self.relu = nn.ReLU()
        # self.msb2=MSBlock(filters[0],filters[1])
        self.msb1_2 = Atten_ASPP(filters[0], filters[1])
        self.msb2_3 = Atten_ASPP(filters[1], filters[2])
        self.msb3_4 = Atten_ASPP(filters[2], filters[3])
        self.aspp5 = ASPP(filters[3], filters[4])

        self.up5 = UpConv(filters[4] * 2, filters[3])
        self.up4 = UpConv(filters[3], filters[2])
        self.up3 = UpConv(filters[2], filters[1])
        self.up2 = UpConv(filters[1], filters[0])
        self.cat_conv4 = CatConv(filters[3] * 2, filters[3])
        self.cat_conv3 = CatConv(filters[2] * 2, filters[2])
        self.cat_conv2 = CatConv(filters[1] * 2, filters[1])
        self.cat_conv1 = CatConv(filters[0] * 2, filters[0])

        self.edge_out1 = nn.Conv2d(filters[0], out_channels, 1, 1, 0)
        self.edge_out2 = UpConvBlock(filters[1], 1)
        self.edge_out3 = UpConvBlock(filters[2], 2)
        self.edge_out4 = UpConvBlock(filters[3], 3)
        self.edge_out = nn.Conv2d(filters[0], out_channels, 1, 1, 0)
        self.body_out = nn.Sequential(
            nn.Conv2d(filters[0], out_channels, 1, 1, 0),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(True),
            nn.Conv2d(out_channels, out_channels, 3, 1, 1)
        )
        self.edge_up2=nn.Sequential(
            nn.ConvTranspose2d(filters[1], filters[1], 2, 2),
            nn.BatchNorm2d(filters[1]),
            nn.ReLU(True),
            nn.Conv2d(filters[1], filters[0], 3, 1, 1),
            nn.BatchNorm2d(filters[0])
        )
        self.edge_up3 = nn.Sequential(
            nn.ConvTranspose2d(filters[2], filters[2], 4, 4),
            nn.BatchNorm2d(filters[2]),
            nn.ReLU(True),
            nn.Conv2d(filters[2], filters[0], 3, 1, 1),
            nn.BatchNorm2d(filters[0])
        )
        self.edge_up4 = nn.Sequential(
            nn.ConvTranspose2d(filters[3], filters[3], 8, 8),
            nn.BatchNorm2d(filters[3]),
            nn.ReLU(True),
            nn.Conv2d(filters[3], filters[0], 3, 1, 1),
            nn.BatchNorm2d(filters[0])
        )
        self.base_up=nn.Sequential(
            nn.ConvTranspose2d(filters[4],filters[4],16,16),
            nn.BatchNorm2d(filters[4]),
            nn.ReLU(True),
            nn.Conv2d(filters[4],filters[0],3,1,1),
            nn.BatchNorm2d(filters[0])
        )
        self.out = nn.Sequential(
            nn.Conv2d(filters[0], out_channels, 3, 1, 1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(True),
            nn.Conv2d(out_channels, out_channels, 3, 1, 1)
        )

        self.squeeze_body = SqueezeBodyEdge(filters[0])
        self.edge_conv = nn.Sequential(
            nn.Conv2d(filters[0]*4, filters[0], 3, 1, 1),
            nn.BatchNorm2d(filters[0]),
        )
        self.body_conv = nn.Sequential(
            nn.Conv2d(filters[0] * 2, filters[0], 3, 1, 1),
            nn.BatchNorm2d(filters[0]),
        )
        self.sigmoid = nn.Sigmoid()
        initialize_weights(self)

    def forward(self, x, train=False):
        outs = self.unet_model(x, res1=True)
        x1, x2, x3, x4, x5, hd4, hd3, hd2, hd1 = outs
        hd1_aspp = self.aspp(hd1)
        hd1_aspp_pool = self.maxpool(hd1_aspp)
        ms2 = self.msb1_2(hd1_aspp_pool)
        ms2_ = self.relu(ms2 + hd2)
        # ms2_=self.relu(self.sigmoid(hd2)*ms2)
        ms2_pool = self.maxpool(ms2_)
        ms3 = self.msb2_3(ms2_pool)
        ms3_ = self.relu(ms3 + hd3)
        # ms3_=self.relu(self.sigmoid(hd3)*ms3)
        ms3_pool = self.maxpool(ms3_)
        ms4 = self.msb3_4(ms3_pool)
        ms4_ = self.relu(ms4 + hd4)
        # ms4_=self.relu(self.sigmoid(hd4)*ms4)
        md4_pool = self.maxpool(ms4_)
        x5_aspp = self.aspp5(md4_pool)
        x5_ = torch.cat((self.relu(x5_aspp), x5), dim=1)

        h5_4 = self.up5(x5_)
        h4 = self.cat_conv4(torch.cat((h5_4, ms4_), dim=1))
        h4_3 = self.up4(h4)
        h3 = self.cat_conv3(torch.cat((h4_3, ms3_), dim=1))
        h3_2 = self.up3(h3)
        h2 = self.cat_conv2(torch.cat((h3_2, ms2_), dim=1))
        h2_1 = self.up2(h2)
        h1 = self.cat_conv1(torch.cat((h2_1, self.relu(hd1_aspp)), dim=1))
        # print(h1.size(),h2.size(),h3.size(),h4.size())
        edge_out1 = self.edge_out1(h1)
        edge_out2 = self.edge_out2(h2)
        edge_out3 = self.edge_out3(h3)
        edge_out4 = self.edge_out4(h4)

        x_size = x.size()[2:]
        h2_up =self.edge_up2(h2)
        h3_up = self.edge_up3(h3)
        h4_up = self.edge_up4(h4)
        h5_up = self.base_up(x5_aspp)

        body = self.squeeze_body(hd1_aspp)
        body = self.relu(self.body_conv(torch.cat((body, hd1_aspp),dim=1))+h5_up)
        body_out = self.body_out(body)
        edge = torch.cat((h1, h2_up, h3_up, h4_up), dim=1)
        edge = self.edge_conv(edge)
        edge_ = self.relu(edge)
        edge_out = self.edge_out(edge_)
        final_out = self.out(body + edge_)
        if train:
            return final_out, body_out, (edge_out1,edge_out2,edge_out3,edge_out4,edge_out)

        return final_out

class Edge_model(Module):
    def __init__(self,filters,in_channels=3,out_channels=1):
        super(Edge_model, self).__init__()
        self.unet_model = Unet(in_channels, out_channels, filters)
        self.msb1_2 = Atten_ASPP(filters[0], filters[1])
        self.msb2_3 = Atten_ASPP(filters[1], filters[2])
        self.msb3_4 = Atten_ASPP(filters[2], filters[3])
        self.aspp5 = ASPP(filters[3], filters[4])

        self.up5 = UpConv(filters[4] * 2, filters[3])
        self.up4 = UpConv(filters[3], filters[2])
        self.up3 = UpConv(filters[2], filters[1])
        self.up2 = UpConv(filters[1], filters[0])
        self.cat_conv4 = CatConv(filters[3] * 2, filters[3])
        self.cat_conv3 = CatConv(filters[2] * 2, filters[2])
        self.cat_conv2 = CatConv(filters[1] * 2, filters[1])
        self.cat_conv1 = CatConv(filters[0] * 2, filters[0])

        self.edge_out1 = nn.Conv2d(filters[0], out_channels, 1, 1, 0)
        self.edge_out2 = UpConvBlock(filters[1], 1)
        self.edge_out3 = UpConvBlock(filters[2], 2)
        self.edge_out4 = UpConvBlock(filters[3], 3)
        self.edge_out = nn.Conv2d(filters[0], out_channels, 1, 1, 0)

        self.edge_up2 = nn.Sequential(
            nn.ConvTranspose2d(filters[1], filters[1], 2, 2),
            nn.BatchNorm2d(filters[1]),
            nn.ReLU(),
            nn.Conv2d(filters[1], filters[0], 3, 1, 1),
            nn.BatchNorm2d(filters[0])
        )
        self.edge_up3 = nn.Sequential(
            nn.ConvTranspose2d(filters[2], filters[2], 4, 4),
            nn.BatchNorm2d(filters[2]),
            nn.ReLU(),
            nn.Conv2d(filters[2], filters[0], 3, 1, 1),
            nn.BatchNorm2d(filters[0])
        )
        self.edge_up4 = nn.Sequential(
            nn.ConvTranspose2d(filters[3], filters[3], 8, 8),
            nn.BatchNorm2d(filters[3]),
            nn.ReLU(),
            nn.Conv2d(filters[3], filters[0], 3, 1, 1),
            nn.BatchNorm2d(filters[0])
        )
        self.base_up = nn.Sequential(
            nn.ConvTranspose2d(filters[4], filters[4], 16, 16),
            nn.BatchNorm2d(filters[4]),
            nn.ReLU(),
            nn.Conv2d(filters[4], filters[0], 3, 1, 1),
            nn.BatchNorm2d(filters[0])
        )
        self.squeeze_body = SqueezeBodyEdge(filters[0])
        self.edge_conv = nn.Sequential(
            nn.Conv2d(filters[0] * 4, filters[0], 3, 1, 1),
            nn.BatchNorm2d(filters[0]),
        )
        self.sigmoid = nn.Sigmoid()
        self.aspp = ASPP(filters[0], filters[0])
        self.maxpool=nn.MaxPool2d(2,2)
        self.relu=nn.ReLU()


    def forward(self, x):
        outs = self.unet_model(x, res1=True)
        x1, x2, x3, x4, x5, hd4, hd3, hd2, hd1 = outs
        hd1_aspp = self.aspp(hd1)
        hd1_aspp_pool = self.maxpool(hd1_aspp)
        ms2 = self.msb1_2(hd1_aspp_pool)
        ms2_ = self.relu(ms2 + hd2)
        # ms2_=self.relu(self.sigmoid(hd2)*ms2)
        ms2_pool = self.maxpool(ms2_)
        ms3 = self.msb2_3(ms2_pool)
        ms3_ = self.relu(ms3 + hd3)
        # ms3_=self.relu(self.sigmoid(hd3)*ms3)
        ms3_pool = self.maxpool(ms3_)
        ms4 = self.msb3_4(ms3_pool)
        ms4_ = self.relu(ms4 + hd4)
        # ms4_=self.relu(self.sigmoid(hd4)*ms4)
        md4_pool = self.maxpool(ms4_)
        x5_aspp = self.aspp5(md4_pool)
        x5_ = torch.cat((self.relu(x5_aspp), x5), dim=1)

        h5_4 = self.up5(x5_)
        h4 = self.cat_conv4(torch.cat((h5_4, ms4_), dim=1))
        h4_3 = self.up4(h4)
        h3 = self.cat_conv3(torch.cat((h4_3, ms3_), dim=1))
        h3_2 = self.up3(h3)
        h2 = self.cat_conv2(torch.cat((h3_2, ms2_), dim=1))
        h2_1 = self.up2(h2)
        h1 = self.cat_conv1(torch.cat((h2_1, self.relu(hd1_aspp)), dim=1))


        h2_up = self.edge_up2(h2)
        h3_up = self.edge_up3(h3)
        h4_up = self.edge_up4(h4)
        h5_up = self.base_up(x5_aspp)


        edge = torch.cat((h1, h2_up, h3_up, h4_up), dim=1)
        edge = self.edge_conv(edge)
        edge_ = self.relu(edge)
        edge_out = self.edge_out(edge_)
        return edge_out,hd1_aspp,h5_up

class Body_model(Module):
    def __init__(self,filters,out_channels=1):
        super(Body_model, self).__init__()
        self.base_up = nn.Sequential(
            nn.ConvTranspose2d(filters[4], filters[4], 16, 16),
            nn.BatchNorm2d(filters[4]),
            nn.ReLU(),
            nn.Conv2d(filters[4], filters[0], 3, 1, 1),
            nn.BatchNorm2d(filters[0])
        )
        self.out = nn.Sequential(
            nn.Conv2d(filters[0], out_channels, 3, 1, 1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, 3, 1, 1)
        )

        self.squeeze_body = SqueezeBodyEdge(filters[0])
        self.edge_conv = nn.Sequential(
            nn.Conv2d(filters[0] * 4, filters[0], 3, 1, 1),
            nn.BatchNorm2d(filters[0]),
        )
        self.body_conv = nn.Sequential(
            nn.Conv2d(filters[0] * 2, filters[0], 3, 1, 1),
            nn.BatchNorm2d(filters[0]),
        )
        self.body_out = nn.Sequential(
            nn.Conv2d(filters[0], out_channels, 1, 1, 0),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, 3, 1, 1)
        )
        self.relu=nn.ReLU()

    def forward(self,hd1,hd5):
        body = self.squeeze_body(hd1)
        body = self.relu(self.body_conv(torch.cat((body, hd1), dim=1)) + hd5)
        body_out = self.body_out(body)
        return body_out


class DCCC_Net(Module):
    def __init__(self,in_channels,out_channels,filters=(64,128,256,512,1024)):
        super(DCCC_Net, self).__init__()

        self.edge_model=Edge_model(filters)
        self.body_model=Body_model(filters)
        self.out = nn.Sequential(
            nn.Conv2d(1, out_channels, 1, 1, 0),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, 3, 1, 1)
        )
        initialize_weights(self)
    def forward(self,x,train=False):

        outs=self.edge_model(x)
        # print(len(a))
        edge,hd1,hd5=outs
        body=self.body_model(hd1,hd5)
        final_out = self.out(body + edge)
        if train:
            return (final_out,body,edge)
        return final_out



if __name__=='__main__':
    # model=DcoUnet(3,1,(4,8,16,64,128)).cuda()
    # summary(model,(3,512,512))
    model=DCCC_Net(3,1,(16,32,64,72,128)).cuda()
    summary(model,(3,32,32))




