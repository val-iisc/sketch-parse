import torch.nn as nn
import math
import torch.utils.model_zoo as model_zoo
import torch
import torch.nn.init as init
__all__ = ['Res_Deeplab_forward','Res_Deeplab','ResNet', 'resnet18', 'resnet34', 'resnet50', 'resnet101',
           'resnet152']

affine_par = True


def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1,  dilation_ = 1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, stride=stride, bias=False) # change
        self.bn1 = nn.BatchNorm2d(planes,affine = affine_par)
        for i in self.bn1.parameters():
            i.requires_grad = False
        padding = 1
        if dilation_ == 2:
            padding = 2
        elif dilation_ == 4:
            padding = 4
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, # change
                               padding=padding, bias=False, dilation = dilation_)
        self.bn2 = nn.BatchNorm2d(planes,affine = affine_par)
        for i in self.bn2.parameters():
            i.requires_grad = False
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4, affine = affine_par)
        for i in self.bn3.parameters():
            i.requires_grad = False
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride



    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

class Classifier_Module(nn.Module):
    def __init__(self, dilation_series, padding_series):
        super(Classifier_Module, self).__init__()
        self.conv2d_list = nn.ModuleList()
        self.conv2d_aux_list = nn.ModuleList()

        for dilation,padding in zip(dilation_series,padding_series):
            self.conv2d_list.append(nn.Conv2d(2048,5,kernel_size=3,stride=1, padding =padding, dilation = dilation,bias = True))
            self.conv2d_aux_list.append(nn.Conv2d(2048,2,kernel_size=3,stride=1, padding =padding, dilation = dilation,bias = True))

        for m in self.conv2d_list:
            n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            m.weight.data.normal_(0, 0.01)
            m.bias.data.fill_(0)

    def forward(self, x):
        out = self.conv2d_list[0](x)
        for i in range(len(self.conv2d_list)-1):
            out += self.conv2d_list[i+1](x)

        return out

class View(nn.Module):
    def __init__(self, *shape):
        super(View, self).__init__()
        self.shape = shape
    def forward(self, x):
        return x.view(*shape)

class PoseModule(nn.Module):
    def __init__(self):
        super(PoseModule, self).__init__()
        
        # TODO: Train BatchNorm ?
        self.PoseC1 = nn.Sequential(
                nn.Conv2d(5,  128, kernel_size=3, stride=2, padding=1, dilation=2, bias=True),
                nn.ReLU(),
                nn.Conv2d(128, 64, kernel_size=3, stride=2, padding=1, dilation=2, bias=True),
                nn.ReLU(),
                nn.Conv2d(64, 32, kernel_size=9, stride=1, padding=0, bias=True),
                nn.ReLU(),
                )
        self.PoseC2 = nn.Sequential(
                nn.Linear(32,8),
                )
        init.normal(self.PoseC1[0].weight, mean=0, std=0.01)
        init.constant(self.PoseC1[0].bias, 0)
        init.normal(self.PoseC1[2].weight, mean=0, std=0.01)
        init.constant(self.PoseC1[2].bias, 0)
        init.normal(self.PoseC1[4].weight, mean=0, std=0.01)
        init.constant(self.PoseC1[4].bias, 0)
        init.normal(self.PoseC2[0].weight, mean=0, std=0.01)
        init.constant(self.PoseC2[0].bias, 0)

    def forward(self, x):
        # Put in the convs here
        out = self.PoseC1(x)
        out = out.view(x.size(0),-1)
        out = self.PoseC2(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, layers):
        self.inplanes = 64
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64,affine = affine_par)
        for i in self.bn1.parameters():
            i.requires_grad = False
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1, ceil_mode=True) # change
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=1, dilation__ = 2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=1, dilation__ = 4)
        self.layer5 = self._make_pred_layer(Classifier_Module, [6,12,18,24],[6,12,18,24])

        #self.avgpool = nn.AvgPool2d(7)
        #self.fc = nn.Linear(512 * 59*59, num_classes)
        #self.layer5 = self._make_pred_layer()
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, 0.01)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1,dilation__ = 1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion or dilation__ == 2 or dilation__ == 4:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion,affine = affine_par),
            )
        for i in downsample._modules['1'].parameters():
            i.requires_grad = False
        layers = []
        layers.append(block(self.inplanes, planes, stride,dilation_=dilation__, downsample = downsample ))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes,dilation_=dilation__))
        return nn.Sequential(*layers)

    def _make_pred_layer(self, block, dilation_series, padding_series):
        return block(dilation_series, padding_series)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        #x = self.avgpool(x)
        #x = x.view(x.size(0), -1)
        #x = self.fc(x)
        return x

class MS_Deeplab(nn.Module):
    def __init__(self,block):
        super(MS_Deeplab,self).__init__()
        self.Scale1 = ResNet(block,[3, 4, 23, 3])   # for original scale
        self.Scale2 = ResNet(block,[3, 4, 23, 3])   # for 0.75x scale
        self.Scale3 = ResNet(block,[3, 4, 23, 3])   # for 0.5x scale
        self.pose = PoseModule() # pose classifier for max loss output
        self.interp1 = nn.UpsamplingBilinear2d(size = (241,241))
        self.interp2 = nn.UpsamplingBilinear2d(size = (161,161))
        self.interp3 = nn.UpsamplingBilinear2d(size = (41,41))
        self.interp4 = nn.UpsamplingBilinear2d(size = (41,41))
        return

    def forward(self,x):
        out = []
        x2 = self.interp1(x)
        x3 = self.interp2(x)
        out.append(self.Scale1(x))              # for original scale
        out.append(self.interp3(self.Scale2(x2)))# for 0.75x scale
        out.append(self.Scale3(x3))             # for 0.5x scale
        x2Out_interp = out[1]
        x3Out_interp = self.interp4(out[2])

        #cat = torch.cat((out[0],x2Out_interp,x3Out_interp),dimension = 4)
        #out.append(torch.max(cat,dim = 4))
        temp1 = torch.max(out[0], x2Out_interp)
        temp2 = torch.max(temp1, x3Out_interp)
        out.append(temp2)
        x = self.pose(temp2)
        out.append(x)

        return out

def Res_Deeplab():
    model = MS_Deeplab(Bottleneck)
    return model
