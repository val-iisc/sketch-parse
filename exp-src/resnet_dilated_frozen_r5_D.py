import torch.nn as nn
import math
import torch.utils.model_zoo as model_zoo
import torch

__all__ = ['Res_Deeplab_forward','Res_Deeplab','ResNet', 'resnet18', 'resnet34', 'resnet50', 'resnet101',
           'resnet152']

affine_par = True
model_urls = {
    'resnet18': 'https://s3.amazonaws.com/pytorch/models/resnet18-5c106cde.pth',
    'resnet34': 'https://s3.amazonaws.com/pytorch/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://s3.amazonaws.com/pytorch/models/resnet50-19c8e357.pth',
    'resnet101': 'https://s3.amazonaws.com/pytorch/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://s3.amazonaws.com/pytorch/models/resnet152-b121ed2d.pth',
}


def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes, affine = affine_par)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes, affine = affine_par)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


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

    def __init__(self,no_outs,dilation_series,padding_series):
        super(Classifier_Module, self).__init__()
	self.conv2d_list = nn.ModuleList()
	for dilation,padding in zip(dilation_series,padding_series):
	    self.conv2d_list.append(nn.Conv2d(2048,no_outs,kernel_size=3,stride=1, padding =padding, dilation = dilation,bias = True))

        for m in self.conv2d_list:
            n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            #m.weight.data.normal_(0, math.sqrt(2. / n))
	    m.weight.data.normal_(0,0.01)


    def forward(self, x):
	out = self.conv2d_list[0](x)
	for i in range(len(self.conv2d_list)-1):
	    out += self.conv2d_list[i+1](x)
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
        self.layer4 = self._make_layer(block, 512, layers[3]-1, stride=1, dilation__ = 4,change=False)

	self.layer4_5_r0 = self._make_layer(block, 512, 1, stride=1, dilation__ = 4,change=False,downsample_needed = False)
	self.layer5_r0 = self._make_pred_layer(Classifier_Module,5, [6,12,18,24],[6,12,18,24])

	self.layer4_5_r1 = self._make_layer(block, 512, 1, stride=1, dilation__ = 4,change=False,downsample_needed = False)
	self.layer5_r1 = self._make_pred_layer(Classifier_Module,5, [6,12,18,24],[6,12,18,24])

	self.layer4_5_r2 = self._make_layer(block, 512, 1, stride=1, dilation__ = 4,change=False,downsample_needed = False)
	self.layer5_r2 = self._make_pred_layer(Classifier_Module,7, [6,12,18,24],[6,12,18,24])

	self.layer4_5_r3 = self._make_layer(block, 512, 1, stride=1, dilation__ = 4,change=False,downsample_needed = False)
	self.layer5_r3 = self._make_pred_layer(Classifier_Module,5, [6,12,18,24],[6,12,18,24])

	self.layer4_5_r4 = self._make_layer(block, 512, 1, stride=1, dilation__ = 4,change=False,downsample_needed = False)
	self.layer5_r4 = self._make_pred_layer(Classifier_Module,9, [6,12,18,24],[6,12,18,24])




        #self.avgpool = nn.AvgPool2d(7)
        #self.fc = nn.Linear(512 * 59*59, num_classes)
	#self.layer5 = self._make_pred_layer()
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                #m.weight.data.normal_(0, math.sqrt(2. / n))
                m.weight.data.normal_(0,0.01 )
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
        #        for i in m.parameters():
        #            i.requires_grad = False

    def _make_layer(self, block, planes, blocks, stride=1,dilation__ = 1,change=True,downsample_needed = True):
        downsample = None
        if (stride != 1 or self.inplanes != planes * block.expansion or dilation__ == 2 or dilation__ == 4) and downsample_needed:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion,affine = affine_par),
            )
            for i in downsample._modules['1'].parameters():
                i.requires_grad = False
        layers = []
	if downsample_needed:
	    layers.append(block(self.inplanes, planes, stride,dilation_=dilation__, downsample = downsample ))
	else:
	    #print 'downsampled', planes
	    layers.append(block(planes*block.expansion,planes,dilation_=dilation__))
#	    layers.append(block(512,planes,dilation_=dilation__))

        #if not dilation__ == 4:
	self.inplanes_old = self.inplanes
	self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes,dilation_=dilation__))
        if not change:
	    self.inplanes = self.inplanes_old

        return nn.Sequential(*layers)



    def _make_pred_layer(self,block,no_outs, dilation_series, padding_series):
	return block(no_outs,dilation_series,padding_series)

    def forward(self, x):
        x[0] = self.conv1(x[0])
        x[0] = self.bn1(x[0])
        x[0] = self.relu(x[0])
        x[0] = self.maxpool(x[0])

        x[0] = self.layer1(x[0])
        x[0] = self.layer2(x[0])
        x[0] = self.layer3(x[0])
	x[0] = self.layer4(x[0])

	if x[1] == 0:
	    #print 'route 0'
            x[0] = self.layer4_5_r0(x[0])
	    x[0] = self.layer5_r0(x[0])

	elif x[1] == 1:
	    #print 'route 1'
            x[0] = self.layer4_5_r1(x[0])
	    x[0] = self.layer5_r1(x[0])

	elif x[1] == 2:
            x[0] = self.layer4_5_r2(x[0])
	    x[0] = self.layer5_r2(x[0])

	elif x[1] == 3:
            x[0] = self.layer4_5_r3(x[0])
	    x[0] = self.layer5_r3(x[0])

	elif x[1] == 4:
            x[0] = self.layer4_5_r4(x[0])
	    x[0] = self.layer5_r4(x[0])

        return x[0]

class MS_Deeplab(nn.Module):
    def __init__(self,block):
	super(MS_Deeplab,self).__init__()
	self.Scale1 = ResNet(block,[3, 4, 23, 3])   # for original scale
	self.Scale2 = ResNet(block,[3, 4, 23, 3])   # for 0.75x scale
	self.Scale3 = ResNet(block,[3, 4, 23, 3])   # for 0.5x scale
	self.interp1 = nn.UpsamplingBilinear2d(size = (241,241))
        self.interp2 = nn.UpsamplingBilinear2d(size = (161,161))
        self.interp3 = nn.UpsamplingBilinear2d(size = (41,41))
        self.interp4 = nn.UpsamplingBilinear2d(size = (41,41))

    def forward(self,x):
        out = []

	x2 = []
        x2.append(self.interp1(x[0]))
	x2.append(x[1])

	x3 = []
        x3.append(self.interp2(x[0]))
	x3.append(x[1])

	out.append(self.Scale1(x))	# for original scale
	out.append(self.interp3(self.Scale2(x2)))	# for 0.75x scale but interped to original scale
	out.append(self.Scale3(x3))	# for 0.5x scale


        x2Out_interp = out[1]
        x3Out_interp = self.interp4(out[2])
        #cat = torch.cat((out[0],x2Out_interp,x3Out_interp),dimension = 4)
        #out.append(torch.max(cat,dim = 4))
        temp1 = torch.max(out[0],x2Out_interp)
        out.append(torch.max(temp1,x3Out_interp))
	return out



def Res_Deeplab():
    model = MS_Deeplab(Bottleneck)
    return model

