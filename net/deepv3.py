import logging
import torch
from torch.nn import functional as F
import torch.nn as nn
from torch.nn import Module
from collections import OrderedDict
from functools import partial
# from torchsummary import summary

def initialize_weights(*models):
    for model in models:
        for module in model.modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                nn.init.kaiming_normal_(module.weight)
                if module.bias is not None:
                    module.bias.data.zero_()
            elif isinstance(module, nn.BatchNorm2d):
                module.weight.data.fill_(1)
                module.bias.data.zero_()

class _AtrousSpatialPyramidPoolingModule(Module):
    def __init__(self, in_dim, reduction_dim=256, output_stride=16, rates=(6, 12, 18)):
        super(_AtrousSpatialPyramidPoolingModule, self).__init__()
        if output_stride==8:
            rates=[2*r for r in rates]
        elif output_stride == 16:
            pass
        else:
            raise 'output stride of {} not supported'.format(output_stride)
        self.features=nn.ModuleList()
        self.features.append(
            nn.Sequential(
                nn.Conv2d(in_dim,reduction_dim,1,1,0),
                nn.BatchNorm2d(reduction_dim),
                nn.ReLU(True)
            )
        )
        for r in rates:
            self.features.append(
                nn.Sequential(
                    nn.Conv2d(in_dim,reduction_dim,3,dilation=r,padding=r,bias=False),
                    nn.BatchNorm2d(reduction_dim),
                    nn.ReLU(True)
                )
            )
        self.img_pool=nn.AdaptiveAvgPool2d(1)
        self.img_conv=nn.Sequential(
            nn.Conv2d(in_dim,reduction_dim,1,1,0),
            # nn.BatchNorm2d(reduction_dim),
            # nn.ReLU(True)
        )
    def forward(self,x):
        x_size=x.size()
        img_features=self.img_pool(x)
        img_features=self.img_conv(img_features)
        img_features=F.interpolate(img_features,x_size[2:],mode='bilinear')
        out = img_features
        # print(out.shape)
        for f in self.features:
            y=f(x)
            out=torch.cat((out,y),dim=1)
        #     print(y.shape)
        # print(out.shape)
        return out



class GlobalAvgPool2d(nn.Module):
    """
    Global average pooling over the input's spatial dimensions
    """

    def __init__(self):
        super(GlobalAvgPool2d, self).__init__()
        logging.info("Global Average Pooling Initialized")

    def forward(self, inputs):
        in_size = inputs.size()
        return inputs.view((in_size[0], in_size[1], -1)).mean(dim=2)

class IdentityResidualBlock(Module):
    def __init__(self,in_channels,channels,stride=1,dilation=1,groups=1,dropout=None,dist_bn=False):
        super(IdentityResidualBlock, self).__init__()
        self.dist_bn=dist_bn
        if len(channels) != 2 and len(channels) != 3:
            raise ValueError("channels must contain either two or three values")
        if len(channels) == 2 and groups != 1:
            raise ValueError("groups > 1 are only valid if len(channels) == 3")
        is_bottlenck=len(channels)==3
        need_proj_conv = stride != 1 or in_channels != channels[-1]
        self.bn1=nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.ReLU(True)
        )
        if not is_bottlenck:
            layers=[
                ('conv1',nn.Conv2d(in_channels,channels[0],3,stride=stride,padding=dilation,dilation=dilation,bias=False)
                ),
                ('bn2',nn.BatchNorm2d(channels[0])),
                ('relu2',nn.ReLU(True)),
                ("conv2", nn.Conv2d(channels[0], channels[1],
                                    3,
                                    stride=1,
                                    padding=dilation,
                                    bias=False,
                                    dilation=dilation))
            ]
            if dropout is not None:
                layers=layers[0:2]+['dropout',dropout()]+layers[2:]
        else:
            layers = [
                ("conv1",
                 nn.Conv2d(in_channels,
                           channels[0],
                           1,
                           stride=stride,
                           padding=0,
                           bias=False)),
                ("bn2", nn.BatchNorm2d(channels[0])),
                ('relu2', nn.ReLU(True)),
                ("conv2", nn.Conv2d(channels[0],
                                    channels[1],
                                    3, stride=1,
                                    padding=dilation, bias=False,
                                    groups=groups,
                                    dilation=dilation)),
                ("bn3", nn.BatchNorm2d(channels[1])),
                ("conv3", nn.Conv2d(channels[1], channels[2],
                                    1, stride=1, padding=0, bias=False))
            ]
            if dropout is not None:
                layers = layers[0:4] + [("dropout", dropout())] + layers[4:]

        self.conv=nn.Sequential(OrderedDict(layers))
        if need_proj_conv:
            self.proj_conv = nn.Conv2d(
                in_channels, channels[-1], 1, stride=stride, padding=0, bias=False)

    def forward(self,x):
        if hasattr(self,'proj_conv'):
            bn1=self.bn1(x)
            shortcut=self.proj_conv(bn1)
        else:
            shortcut=x.clone()
            bn1=self.bn1(x)
        # print(bn1.shape)
        out=self.conv(bn1)
        out.add_(shortcut)
        return out




class WiderResNetA2(Module):
    def __init__(self,structure,classes=0,dilation=False,dist_bn=False):
        super(WiderResNetA2, self).__init__()
        self.structure = structure
        self.dilation = dilation
        self.dist_bn=dist_bn
        if len(structure)!=6:
            raise ValueError("Expected a structure with six values")
        self.mod1 = nn.Sequential(OrderedDict([
            ("conv1", nn.Conv2d(3, 64, 3, stride=1, padding=1, bias=False))
        ]))
        in_channels=64
        # [(128, 128), (256, 256), (512, 512), (512, 1024),
        #  (512, 1024, 2048), (1024, 2048, 4096)]
        channels = [(128, 128), (128, 128), (128, 128), (128, 128),
                    (128, 128, 2048), (128, 2048, 4096)]
        for mod_id,num in enumerate(structure):
            blocks=[]
            for block_id in range(num):
                if not dilation:
                    stride=2 if block_id==0 and 2<=mod_id<=4 else 1
                else:
                    if mod_id==3:
                        dil=2
                    elif mod_id>4:
                        dil=4
                    else:
                        dil=1
                    stride=2 if block_id==0 and mod_id==2 else 1
                if mod_id==4:
                    drop=partial(nn.Dropout,p=0.3)
                elif mod_id==5:
                    drop=partial(nn.Dropout,p=0.5)
                else:
                    drop=None
                print(in_channels,channels[mod_id])
                blocks.append((
                    f'blocks{block_id+1}',
                    IdentityResidualBlock(in_channels,channels[mod_id],stride=stride,dilation=dil,dropout=drop,dist_bn=self.dist_bn)
                ))
                in_channels=channels[mod_id][-1]
            if mod_id<=4:
                self.add_module(f'pool{mod_id+2}',nn.MaxPool2d(3,2,1))
            self.add_module(f'mod{mod_id+2}',nn.Sequential(OrderedDict(blocks)))
        self.bn_out=nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.ReLU(True)
        )
        if classes != 0:
            self.classifier = nn.Sequential(OrderedDict([
                ("avg_pool", GlobalAvgPool2d()),
                ("fc", nn.Linear(in_channels, classes))
            ]))
    def forward(self,x):
        out=self.mod1(x)
        out=self.mod2(self.pool2(out))
        out=self.mod3(self.pool3(out))
        out=self.mod4(out)
        out=self.mod5(out)
        out=self.mod6(out)
        out=self.mod7(out)
        out = self.bn_out(out)
        if hasattr(self,'classifier'):
            return self.classifier(out)
        return out


class DeepWV3Plus(Module):
    def __init__(self,num_classes,trunk='WideResnet38', criterion=None):
        super(DeepWV3Plus, self).__init__()
        self.criterion=criterion
        wide_resnet = WiderResNetA2(structure=[1, 1, 1, 1, 1, 1],classes=1000, dilation=True)

        # wide_resnet=wide_resnet.Module
        self.mod1 = wide_resnet.mod1
        self.mod2 = wide_resnet.mod2
        self.mod3 = wide_resnet.mod3
        self.mod4 = wide_resnet.mod4
        self.mod5 = wide_resnet.mod5
        self.mod6 = wide_resnet.mod6
        self.mod7 = wide_resnet.mod7
        self.pool2 = wide_resnet.pool2
        self.pool3 = wide_resnet.pool3

        self.aspp = _AtrousSpatialPyramidPoolingModule(4096, 256,
                                                       output_stride=8)
        self.bot_fine = nn.Conv2d(128, 48, kernel_size=1, bias=False)
        self.bot_aspp = nn.Conv2d(1280, 256, kernel_size=1, bias=False)
        self.final=nn.Sequential(
            nn.Conv2d(48+256,256,3,1,1),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.Conv2d(256,256,3,1,1),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            # nn.Upsample(scale_factor=2,mode='bilinear'),
            # nn.Conv2d(256, num_classes, kernel_size=1, bias=False)
        )
        self.out_conv=nn.Sequential(
            nn.Upsample(scale_factor=2,mode='bilinear'),
            nn.Conv2d(256,num_classes,3,1,1,bias=False),
            nn.BatchNorm2d(num_classes),
            nn.ReLU(True),
            nn.Conv2d(num_classes,num_classes,1,1,0)

        )
        initialize_weights(self.final)

    def forward(self, inp, gts=None):
        x_size=inp.size()
        x=self.mod1(inp)
        m2=self.mod2(self.pool2(x))
        x=self.mod3(self.pool3(m2))
        x=self.mod4(x)
        x=self.mod5(x)
        x=self.mod6(x)
        x=self.mod7(x)
        x=self.aspp(x)
        dec0_up=self.bot_aspp(x)
        dec0_fine=self.bot_fine(m2)
        dec0_up=F.interpolate(dec0_up,m2.size()[2:],mode='bilinear')
        dec0 = [dec0_fine, dec0_up]
        dec0 = torch.cat(dec0, 1)
        dec1 = self.final(dec0)
        # print(dec1.shape,x_size)
        # out = F.interpolate(dec1, scale_factor=2,mode='bilinear')
        # print(out.shape)
        out=self.out_conv(dec1)
        return out



if __name__=='__main__':
    # net=WiderResNetA2(structure=[1, 1, 1, 1, 1, 1],classes=1, dilation=True).cuda()
    # summary(net,(3,32,32))
    # print(net)
    net=DeepWV3Plus(1).cuda()
    summary(net, (3, 32, 32))