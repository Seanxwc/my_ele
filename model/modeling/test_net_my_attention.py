
import torch
import torch.nn as nn
from torchsummary import summary
import torch.nn.functional as F
#from torchstat import stat


class CPAMEnc(nn.Module):
    """
    CPAM encoding module
    """

    def __init__(self, in_channels):
        super(CPAMEnc, self).__init__()
        self.pool1 = nn.AdaptiveAvgPool2d(1)
        self.pool2 = nn.AdaptiveAvgPool2d(2)
        self.pool3 = nn.AdaptiveAvgPool2d(3)
        self.pool4 = nn.AdaptiveAvgPool2d(6)



        self.conv1 = nn.Sequential(nn.Conv2d(in_channels, in_channels, 1, bias=False),
                                nn.BatchNorm2d(in_channels),
                                nn.ReLU(True))
        self.conv2 = nn.Sequential(nn.Conv2d(in_channels, in_channels, 1, bias=False),
                                nn.BatchNorm2d(in_channels),
                                nn.ReLU(True))
        self.conv3 = nn.Sequential(nn.Conv2d(in_channels, in_channels, 1, bias=False),
                                nn.BatchNorm2d(in_channels),
                                nn.ReLU(True))
        self.conv4 = nn.Sequential(nn.Conv2d(in_channels, in_channels, 1, bias=False),
                                nn.BatchNorm2d(in_channels),
                                nn.ReLU(True))

    def forward(self, x):
        b, c, h, w = x.size()




        feat1 = self.conv1(self.pool1(x)).view(b, c, -1)
        feat2 = self.conv2(self.pool2(x)).view(b, c, -1)
        feat3 = self.conv3(self.pool3(x)).view(b, c, -1)
        feat4 = self.conv4(self.pool4(x)).view(b, c, -1)

        return torch.cat((feat1, feat2, feat3, feat4), 2)




class CCAMDec(nn.Module):
    def __init__(self, xin_channels, yin_channels, mid_channels, scale=False):
        super(CCAMDec, self).__init__()
        self.mid_channels = mid_channels

        self.f_self = nn.Sequential(
            nn.Conv2d(xin_channels, 16, 1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(True),
        )

        self.f_y = nn.Sequential(
            nn.Conv2d(xin_channels, 16, 1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(True),
        )
        self.f_up = nn.Sequential(
            nn.Conv2d(in_channels=mid_channels, out_channels=xin_channels,
                      kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(xin_channels),
        )
        self.scale = scale
        nn.init.constant_(self.f_up[1].weight, 0)
        nn.init.constant_(self.f_up[1].bias, 0)

    def forward(self, x, y):
        C = x.size(1)

        fself = self.f_self(x)  #卷积改通道 BXCXHXW  ---->BXKXHXW
        B, K, H, W = fself.size()
        fself = fself.view(B, K, -1)  #   BXKXHXW ----> BXKXN

        fx = x.view(B, C, -1)#BXCXHXW  ---->BXCXN



        #print(fx.size())
        fy = self.f_y(y)#卷积改通道 BXCXHXW  ---->BXKXHXW

        fy = fy.view(B, K, -1).permute(0, 2, 1)  # BXKXHXW ----> BXKXN ------>BXNXK


        #print(fy.size())
        sim_map = torch.matmul(fx, fy)
        if self.scale:
            sim_map = (self.mid_channels ** -.5) * sim_map
        sim_map_div_C = F.softmax(sim_map, dim=-1)

        fout = torch.matmul(sim_map_div_C, fself)
        fout = fout.permute(0, 2, 1).contiguous()
        fout = fout.view(B, C, *x.size()[2:])
        out = self.f_up(fout)
        return x + out












class CPAMDec(nn.Module):
    def __init__(self, xin_channels, yin_channels, mid_channels, scale=False):
        super(CPAMDec, self).__init__()
        self.mid_channels = mid_channels

        self.f_self = nn.Sequential(
            CPAMEnc(xin_channels),
            nn.Linear(50, 50),
        )
        self.f_x = nn.Sequential(
            nn.Conv2d(in_channels=xin_channels, out_channels=mid_channels,
                      kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.Conv2d(in_channels=mid_channels, out_channels=mid_channels,
                      kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(mid_channels),
        )
        self.f_y = nn.Sequential(
            CPAMEnc(yin_channels),
            nn.Linear(50, 50),
        )
        self.f_up = nn.Sequential(
            nn.Conv2d(in_channels=mid_channels, out_channels=xin_channels,
                      kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(xin_channels),
        )
        self.scale = scale
        nn.init.constant_(self.f_up[1].weight, 0)
        nn.init.constant_(self.f_up[1].bias, 0)

    def forward(self, x, y):
        batch_size = x.size(0)

        fself = self.f_self(x)  #BXCXHXW  ---->BXCXM
        fself = fself.permute(0, 2, 1)  #   BXCXM ----> BXMXC
        fx = self.f_x(x).view(batch_size, self.mid_channels, -1)#BXCXHXW  ---->BXCXN
        fx = fx.permute(0, 2, 1)#   BXCXN ----> BXNXC
        #print(fx.size())
        fy = self.f_y(y)#BXCXHXW  ---->BXCXM
        #print(fy.size())
        sim_map = torch.matmul(fx, fy)
        if self.scale:
            sim_map = (self.mid_channels ** -.5) * sim_map
        sim_map_div_C = F.softmax(sim_map, dim=-1) #BXCX    ((N)XM) ===  BXCX   ((HXW)XM)

        fout = torch.matmul(sim_map_div_C, fself)
        fout = fout.permute(0, 2, 1).contiguous()
        fout = fout.view(batch_size, self.mid_channels, *x.size()[2:])
        out = self.f_up(fout)
        return x + out



class Fusion(nn.Module):
    def __init__(self):
        super(Fusion, self).__init__()

        self.conv_cat = nn.Sequential(nn.Conv2d(32 * 2, 32, 3, padding=1, bias=False),
                                      nn.BatchNorm2d(32),
                                      nn.ReLU(True),)  # conv_f

        self.att1 = CPAMDec(32, 32, 32)
        self.att2 = CCAMDec(32, 32, 32)

    def forward(self, x, y):
        x_PAM_out = self.att1(x,y)
        y_CAM_out = self.att2(x,y)
        feat_sum = self.conv_cat(torch.cat([x_PAM_out,y_CAM_out],1))
        #feat_sum = x_PAM_out + y_CAM_out
        feat_sum = x + feat_sum
        return feat_sum




class double_conv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3):
        super(double_conv, self).__init__()
        padding = kernel_size // 2
        self._layers = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size, padding=padding),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self._layers(x)
        return x


class down(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(down, self).__init__()
        self.down = nn.Sequential(
            nn.MaxPool2d(2),
            double_conv(in_channels, out_channels, 3)
        )

    def forward(self, x):
        x = self.down(x)
        return x


class up(nn.Module):
    def __init__(self, in_channels, out_channels):
        # self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        super(up, self).__init__()
        self.up = nn.Upsample(scale_factor=2, mode='bilinear')
        self.dconv = double_conv(in_channels, out_channels, 3)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        x = torch.cat([x1, x2], dim=1)
        x = self.dconv(x)

        return x


class UNet(nn.Module):
    def __init__(self, num_filters=32, num_categories=3, num_in_channels=1):
        super(UNet, self).__init__()

        kernel = 3
        padding = 1
        self.inconv = double_conv(num_in_channels, num_filters)
        self.downconv1 = down(num_filters, num_filters * 2)
        self.downconv2 = down(num_filters * 2, num_filters * 4)
        self.downconv3 = down(num_filters * 4, num_filters * 8)
        self.downconv4 = down(num_filters * 8, num_filters * 16)

        self.upconv1 = up(num_filters * (16 + 8), num_filters * 8)
        self.upconv2 = up(num_filters * (8 + 4), num_filters * 4)
        self.upconv3 = up(num_filters * (4 + 2), num_filters* 2)
        self.upconv4 = up(num_filters * (2 + 1), num_filters)

        self.finalconv = nn.Conv2d(num_filters, num_categories, 1)

        self.edge_conv4 = self.generate_edge_conv(32)
        self.edge_out4 = nn.Sequential(nn.Conv2d(512, 3, 1))
        self.edge_conv5 = self.generate_edge_conv(64)
        self.edge_out5 = nn.Sequential(nn.Conv2d(256, 3, 1))
        self.edge_conv6 = self.generate_edge_conv(128)
        self.edge_out6 = nn.Sequential(nn.Conv2d(128, 3, 1))
        self.edge_conv7 = self.generate_edge_conv(256)
        self.edge_out7 = nn.Sequential(nn.Conv2d(64, 3, 1))

        self.out_loss = nn.Sequential(nn.Conv2d(32, 3, 1))


        self.fusion =Fusion()


        self.up_edge = nn.Sequential(
            nn.Conv2d(128, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )

    def generate_edge_conv(self, in_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, 32, 3,padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )




    def forward(self, x):
        out0 = self.inconv(x)  # *32
        out1 = self.downconv1(out0)  # 64
        out2 = self.downconv2(out1)  # 128
        out3 = self.downconv3(out2)  # 256
        out4 = self.downconv4(out3)  # 512

        out5 = self.upconv1(out4, out3)  # 256
        out6 = self.upconv2(out5, out2)  # 128
        out7 = self.upconv3(out6, out1)  # 64
        out8 = self.upconv4(out7, out0)  # 32
        #out_final = self.finalconv(out8)

        # print(out4.size())
        # print(out5.size())
        # print(out6.size())
        # print(out7.size())
        # print(out_final.size())

        e4 = self.edge_conv4(out0)
        e4 = nn.functional.interpolate(e4, out8.size()[2:], mode='bilinear', align_corners=True)

        # out4_loss = nn.functional.interpolate(out4, out8.size()[2:], mode='bilinear', align_corners=True)
        # e4_out = self.edge_out4(out4_loss)

        e5 = self.edge_conv5(out1)
        e5 = nn.functional.interpolate(e5, out8.size()[2:], mode='bilinear', align_corners=True)

        # out5_loss = nn.functional.interpolate(out5, out8.size()[2:], mode='bilinear', align_corners=True)
        # e5_out = self.edge_out5(out5_loss)

        e6 = self.edge_conv6(out2)
        e6 = nn.functional.interpolate(e6, out8.size()[2:], mode='bilinear', align_corners=True)

        # out6_loss = nn.functional.interpolate(out6, out8.size()[2:], mode='bilinear', align_corners=True)
        # e6_out = self.edge_out6(out6_loss)

        e7 = self.edge_conv7(out3)
        e7 = nn.functional.interpolate(e7, out8.size()[2:], mode='bilinear', align_corners=True)

        # out7_loss = nn.functional.interpolate(out7, out8.size()[2:], mode='bilinear', align_corners=True)
        # e7_out = self.edge_out7(out7_loss)

        # print(e4.size())
        # print(e5.size())
        # print(e6.size())
        # print(e7.size())
        # print(out8.size())
        # print(e4_out.size())
        # print(e5_out.size())
        # print(e6_out.size())
        # print(e7_out.size())
        e = torch.cat((e4, e5, e6, e7), dim=1)  # 32 + 32 + 32 + 32 = 128
        e = self.up_edge(e)  # 128==》32
        #print(e.size())

        out_feature = self.fusion(out8, e)
        self.feature2 = e
        self.feature1 = out8

        out_final = self.finalconv(out_feature)
        self.feature3 = out_feature

        out8_loss = nn.functional.interpolate(out8, out8.size()[2:], mode='bilinear', align_corners=True)
        out8_loss = self.out_loss(out8_loss)
        e_loss = nn.functional.interpolate(e, out8.size()[2:], mode='bilinear', align_corners=True)
        e_loss = self.out_loss(e_loss)
        #print(e_loss)



        return out8_loss, e_loss, out_final