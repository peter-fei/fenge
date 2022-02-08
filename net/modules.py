import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.nn import Module

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

class Atten_ASPP(Module):
    def __init__(self,in_channels,out_channels,rates=(6,12,18)):
        super(Atten_ASPP, self).__init__()
        self.conv_list=nn.ModuleList()
        self.atten_list=nn.ModuleList()
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
        self.fc1 = nn.Conv2d(planes, planes//16, kernel_size=1)
        self.fc2 = nn.Conv2d(planes//16, planes, kernel_size=1)

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



