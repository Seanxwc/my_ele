

import torch.nn as nn
import torch.nn.functional as F

from lib.models.backbones.backbone_selector import BackboneSelector
from lib.models.tools.module_helper import ModuleHelper

class BackboneSelector(object):

    def __init__(self, configer):
        self.configer = configer

    def get_backbone(self, **params):
        backbone = self.configer.get('network', 'backbone')

        model = None
        if ('resnet' in backbone or 'resnext' in backbone or 'resnest' in backbone) and 'senet' not in backbone:
            model = ResNetBackbone(self.configer)(**params)

        elif 'hrne' in backbone:
            model = HRNetBackbone(self.configer)(**params)

        else:
            Log.error('Backbone {} is invalid.'.format(backbone))
            exit(1)

        return model



class BaseOCNet(nn.Module):
    """
    OCNet: Object Context Network for Scene Parsing
    """
    def __init__(self, configer):
        self.inplanes = 128
        super(BaseOCNet, self).__init__()
        self.configer = configer
        self.num_classes = self.configer.get('data', 'num_classes')
        self.backbone = BackboneSelector(configer).get_backbone()

        # extra added layers
        if "wide_resnet38" in self.configer.get('network', 'backbone'):
            in_channels = [2048, 4096]
        else:
            in_channels = [1024, 2048]
        self.oc_module_pre = nn.Sequential(
            nn.Conv2d(in_channels[1], 512, kernel_size=3, stride=1, padding=1),
            ModuleHelper.BNReLU(512, bn_type=self.configer.get('network', 'bn_type')),
        )
        from lib.models.modules.base_oc_block import BaseOC_Module
        self.oc_module = BaseOC_Module(in_channels=512,
                                       out_channels=512,
                                       key_channels=256,
                                       value_channels=256,
                                       dropout=0.05,
                                       sizes=([1]),
                                       bn_type=self.configer.get('network', 'bn_type'))
        self.cls = nn.Conv2d(512, self.num_classes, kernel_size=1, stride=1, padding=0, bias=True)
        self.dsn = nn.Sequential(
            nn.Conv2d(in_channels[0], 512, kernel_size=3, stride=1, padding=1),
            ModuleHelper.BNReLU(512, bn_type=self.configer.get('network', 'bn_type')),
            nn.Conv2d(512, self.num_classes, kernel_size=1, stride=1, padding=0, bias=True)
        )

    def forward(self, x_):
        x = self.backbone(x_)
        x_dsn = self.dsn(x[-2])
        x = self.oc_module_pre(x[-1])
        x = self.oc_module(x)
        x = self.cls(x)
        x_dsn = F.interpolate(x_dsn, size=(x_.size(2), x_.size(3)), mode="bilinear", align_corners=True)
        x = F.interpolate(x, size=(x_.size(2), x_.size(3)), mode="bilinear", align_corners=True)
        return x_dsn, x


class AspOCNet(nn.Module):
    """
    OCNet: Object Context Network for Scene Parsing
    """
    def __init__(self, configer):
        self.inplanes = 128
        super(AspOCNet, self).__init__()
        self.configer = configer
        self.num_classes = self.configer.get('data', 'num_classes')
        self.backbone = BackboneSelector(configer).get_backbone()

        # extra added layers
        if "wide_resnet38" in self.configer.get('network', 'backbone'):
            in_channels = [2048, 4096]
        else:
            in_channels = [1024, 2048]
        from lib.models.modules.asp_oc_block import ASP_OC_Module
        self.context = nn.Sequential(
            nn.Conv2d(in_channels[1], 512, kernel_size=3, stride=1, padding=1),
            ModuleHelper.BNReLU(512, bn_type=self.configer.get('network', 'bn_type')),
            ASP_OC_Module(512, 256, bn_type=self.configer.get('network', 'bn_type')),
        )
        self.cls = nn.Conv2d(512, self.num_classes, kernel_size=1, stride=1, padding=0, bias=True)
        self.dsn = nn.Sequential(
            nn.Conv2d(in_channels[0], 512, kernel_size=3, stride=1, padding=1),
            ModuleHelper.BNReLU(512, bn_type=self.configer.get('network', 'bn_type')),
            nn.Conv2d(512, self.num_classes, kernel_size=1, stride=1, padding=0, bias=True)
        )

    def forward(self, x_):
        x = self.backbone(x_)
        aux_x = self.dsn(x[-2])
        x = self.context(x[-1])
        x = self.cls(x)
        aux_x = F.interpolate(aux_x, size=(x_.size(2), x_.size(3)), mode="bilinear", align_corners=True)
        x = F.interpolate(x, size=(x_.size(2), x_.size(3)), mode="bilinear", align_corners=True)
        return aux_x, x