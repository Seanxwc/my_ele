import torch
import torch.nn as nn


class BasicBlock(nn.Module):
    expansion: int = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):

        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError("BasicBlock only supports groups=1 and base_width=64")
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=3, stride=stride,
                               padding=dilation, groups=groups, bias=False, dilation=dilation)

        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=dilation, groups=groups, bias=False, dilation=dilation)

        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None,
                 groups=1, base_width=64, dilation=1, norm_layer=None, ):
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.0)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = nn.Conv2d(inplanes, width, kernel_size=1, stride=1, bias=False)
        self.bn1 = norm_layer(width)
        self.conv2 = nn.Conv2d(width, width, kernel_size=3, stride=stride, bias=False, padding=dilation,
                               dilation=dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = nn.Conv2d(width, planes * self.expansion, kernel_size=1, stride=1, bias=False)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(
            self, block, layers, num_classes=1000, zero_init_residual=False, groups=1,
            width_per_group=64, replace_stride_with_dilation=None, norm_layer=None):
        super(ResNet, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer
        self.inplanes = 64
        self.dilation = 2
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]

        if len(replace_stride_with_dilation) != 3:
            raise ValueError(
                "replace_stride_with_dilation should be None "
                f"or a 3-element tuple, got {replace_stride_with_dilation}"
            )
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=1, dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2, dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=1, dilate=replace_stride_with_dilation[2])
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)  # type: ignore[arg-type]
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)  # type: ignore[arg-type]

    def _make_layer(
            self,
            block,
            planes,
            blocks,
            stride=1,
            dilate=False,
    ):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = stride

        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False),
                norm_layer(planes * block.expansion))

        layers = []
        layers.append(
            block(
                self.inplanes, planes, stride, downsample, self.groups, self.base_width, previous_dilation, norm_layer
            )
        )
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(
                block(
                    self.inplanes,
                    planes,
                    groups=self.groups,
                    base_width=self.base_width,
                    dilation=self.dilation,
                    norm_layer=norm_layer,
                )
            )
        return nn.Sequential(*layers)

    def _forward_impl(self, x):

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        return x

    def forward(self, x):
        return self._forward_impl(x)

    def _resnet(block, layers, pretrained_path=None, **kwargs, ):
        model = ResNet(block, layers, **kwargs)
        if pretrained_path is not None:
            model.load_state_dict(torch.load(pretrained_path), strict=False)
        return model

    def resnet50(pretrained_path=None, **kwargs):
        return ResNet._resnet(Bottleneck, [3, 4, 6, 3], pretrained_path, **kwargs)

    def resnet101(pretrained_path=None, **kwargs):
        return ResNet._resnet(Bottleneck, [3, 4, 23, 3], pretrained_path, **kwargs)


import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.transforms import Resize


class DCMModle(nn.Module):
    def __init__(self, in_channels=2048, channels=512, filter_size=1, fusion=True):
        super(DCMModle, self).__init__()
        self.filter_size = filter_size
        self.in_channels = in_channels
        self.channels = channels
        self.fusion = fusion

        # Global Information vector
        self.reduce_Conv = nn.Conv2d(self.in_channels, self.channels, 1)
        self.filter = nn.AdaptiveAvgPool2d(self.filter_size)

        self.filter_gen_conv = nn.Conv2d(self.in_channels, self.channels, 1, 1,
                                         0)

        self.residual_conv = nn.Conv2d(self.channels, self.channels, 1)
        self.global_info = nn.Conv2d(self.channels, self.channels, 1)
        self.gla = nn.Conv2d(self.channels, self.filter_size ** 2, 1, 1, 0)

        self.activate = nn.Sequential(nn.BatchNorm2d(self.channels),
                                      nn.ReLU()
                                      )
        if self.fusion:
            self.fusion_conv = nn.Conv2d(self.channels, self.channels, 1)

    def forward(self, x):
        b, c, h, w = x.shape
        generted_filter = self.filter_gen_conv(self.filter(x)).view(b, self.channels, self.filter_size,
                                                                    self.filter_size)
        x = self.reduce_Conv(x)

        c = self.channels
        # [1, b * c, h, w], c = self.channels
        x = x.view(1, b * c, h, w)
        # [b * c, 1, filter_size, filter_size]
        generted_filter = generted_filter.view(b * c, 1, self.filter_size,
                                               self.filter_size)

        pad = (self.filter_size - 1) // 2

        if (self.filter_size - 1) % 2 == 0:
            p2d = (pad, pad, pad, pad)
        else:
            p2d = (pad + 1, pad, pad + 1, pad)

        x = F.pad(input=x, pad=p2d, mode='constant', value=0)

        # [1, b * c, h, w]
        output = nn.functional.conv2d(input=x, weight=generted_filter, groups=b * c)
        # [b, c, h, w]
        output = output.view(b, c, h, w)

        output = self.activate(output)

        if self.fusion:
            output = self.fusion_conv(output)

        return output


class DCMModuleList(nn.ModuleList):
    def __init__(self, filter_sizes=[1, 2, 3, 6], in_channels=2048, channels=512):
        super(DCMModuleList, self).__init__()
        self.filter_sizes = filter_sizes
        self.in_channels = in_channels
        self.channels = channels

        for filter_size in self.filter_sizes:
            self.append(
                DCMModle(in_channels, channels, filter_size)
            )

    def forward(self, x):
        out = []
        for DCM in self:
            DCM_out = DCM(x)
            out.append(DCM_out)
        return out


class DMNet(nn.Module):
    def __init__(self, num_categories):
        super(DMNet, self).__init__()
        self.num_categories = num_categories
        self.backbone = ResNet.resnet50(replace_stride_with_dilation=[1, 2, 4])
        self.in_channels = 2048
        self.channels = 512
        self.DMNet_pyramid = DCMModuleList(filter_sizes=[1, 2, 3, 6], in_channels=self.in_channels,
                                           channels=self.channels)
        self.conv1 = nn.Sequential(
            nn.Conv2d(4 * self.channels + self.in_channels, self.channels, 3, padding=1),
            nn.BatchNorm2d(self.channels),
            nn.ReLU()
        )
        self.cls_conv = nn.Conv2d(self.channels, self.num_categories, 3, padding=1)

    def forward(self, x):
        x = self.backbone(x)
        DM_out = self.DMNet_pyramid(x)
        DM_out.append(x)
        x = torch.cat(DM_out, dim=1)
        x = self.conv1(x)
        x = Resize((8 * x.shape[-2], 8 * x.shape[-1]))(x)
        x = self.cls_conv(x)
        return x



