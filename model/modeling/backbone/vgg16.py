import torch
import torch.nn as nn

# A B D E分别对应VGG11/13/16/19
cfg = {
    'A' : [64,     'M', 128,      'M', 256, 256,           'M', 512, 512,           'M', 512, 512,           'M'],
    'B' : [64, 64, 'M', 128, 128, 'M', 256, 256,           'M', 512, 512,           'M', 512, 512,           'M'],
    'D' : [64, 64, 'M', 128, 128, 'M', 256, 256, 256,      'M', 512, 512, 512,      'M', 512, 512, 512,      'M'],
    'E' : [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M']
}

class VGG16(nn.Module):
    # num_classes=3表明是三分类
    def __init__(self, features, num_classes=3, init_weights=True):
        super(VGG16, self).__init__()
        self.features = features
        # 构造序列器
        self.classifier = nn.Sequential(
            # nn.Linear用于构造全连接层
            # 第一个参数512*7*7由网络结构决定，不可变
            # 第二个参数表示输出张量的大小，也表示神经元的个数，可以微调
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(True),
            nn.Dropout(),

            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(),

            nn.Linear(4096, num_classes)
        )

        if init_weights:
            self._initialize_weights()

    # --------------------------------------------------------#
    # 关于nn.Linear torch.nn.Linear(in_features, out_features, bias=True)
    # nn.Linear用于设置全连接层，其输入输出一般都设置为二维张量，形状为[batch_size,size]
    #   in_features - size of each input sample，输入的二维张量大小
    #   out_features - size of each output sample，输出的二维张量大小
    #   bias - if set to False, the layer will not learn an additive bias
    # 从shape的角度来理解，就是相当于输入[batch_size, in_features],输出[batch_size, out_features]
    # --------------------------------------------------------#

    def forward(self, x): # x为输入的张量
        # 输入的张量x经过所有的卷积层，得到的特征层
        x = self.features(x)
        # 调整x的维度
        x = x.view(x.size(0), -1)
        # 使用构造的序列器，最终输出num_classes用于最后的类别判断
        x = self.classifier(x)
        return x

    # 初始化权重
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

# 构造卷积层的实现函数
def make_layers(cfg, batch_normal=False):
    layers = []
    # 初始通道数为3，假设输入的图片为RGB图
    in_channels = 3
    for v in cfg:
        # 如果当前是池化层
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        # 如果是卷积层+（BN）+ReLU
        else:
            # in_channels：输入通道数
            # v：输出通道数
            # kernel_size：卷积核大小
            # padding：边缘补数
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_normal: # 如果要进行批处理规范化，即添加BN层
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(True)]
            else:
                layers += [conv2d, nn.ReLU(True)]
            # 更新通道数
            in_channels = v
    # 返回卷积部分的序列器
    return nn.Sequential(*layers)

def vgg16(**kwargs):
	model = VGG16(make_layers(cfg['D'], batch_normal=False), **kwargs)
	return model
