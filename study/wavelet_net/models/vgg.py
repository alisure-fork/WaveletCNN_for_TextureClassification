'''VGG11/13/16/19 in Pytorch.'''
import torch
import torch.nn as nn


cfg = {
    'VGG11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'VGG19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M']
}


class VGG(nn.Module):
    def __init__(self, vgg_name="VGG16"):
        super(VGG, self).__init__()
        self.features = self._make_layers(cfg[vgg_name])
        self.classifier = nn.Linear(512, 10)
        pass

    def forward(self, x):
        out = self.features(x)
        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        return out

    @staticmethod
    def _make_layers(cfg_vgg, in_channels=3):
        layers = []
        for x in cfg_vgg:
            if x == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                layers += [nn.Conv2d(in_channels, x, kernel_size=3, padding=1),
                           nn.BatchNorm2d(x), nn.ReLU(inplace=True)]
                in_channels = x
            pass
        layers += [nn.AvgPool2d(kernel_size=1, stride=1)]
        return nn.Sequential(*layers)

    pass


class Conv2dBNReLU(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1):
        super(Conv2dBNReLU, self).__init__()
        self.seq = nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding),
                                 nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True))
        pass

    def forward(self, x):
        return self.seq(x)

    def __call__(self, *args, **kwargs):
        return super(Conv2dBNReLU, self).__call__(*args, **kwargs)

    pass


class WaveletConv2d(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1):
        super(WaveletConv2d, self).__init__()

        self.ll_conv1_1 = Conv2dBNReLU(in_channels, out_channels, kernel_size, padding)
        self.ll_conv1_2 = Conv2dBNReLU(out_channels, out_channels, kernel_size, padding)
        self.lh_conv1_1 = Conv2dBNReLU(in_channels, out_channels, kernel_size, padding)
        self.lh_conv1_2 = Conv2dBNReLU(out_channels, out_channels, kernel_size, padding)
        self.hl_conv1_1 = Conv2dBNReLU(in_channels, out_channels, kernel_size, padding)
        self.hl_conv1_2 = Conv2dBNReLU(out_channels, out_channels, kernel_size, padding)
        self.hh_conv1_1 = Conv2dBNReLU(in_channels, out_channels, kernel_size, padding)
        self.hh_conv1_2 = Conv2dBNReLU(out_channels, out_channels, kernel_size, padding)
        pass

    def forward(self, ll, lh, hl, hh):
        ll = self.ll_conv1_2(self.ll_conv1_1(ll))
        lh = self.lh_conv1_2(self.lh_conv1_1(lh))
        hl = self.hl_conv1_2(self.hl_conv1_1(hl))
        hh = self.hh_conv1_2(self.hh_conv1_1(hh))
        return torch.cat([ll, lh, hl, hh], 1)

    def __call__(self, *args, **kwargs):
        return super(WaveletConv2d, self).__call__(*args, **kwargs)

    pass


class WaveletVGG(nn.Module):

    def __init__(self, level=0):
        super(WaveletVGG, self).__init__()

        self.level = level
        # self.kernel = [128, 256, 512, 1024, 1024]
        self.kernel = [64, 128, 256, 512, 512]
        # self.kernel = [32, 64, 128, 256, 256]
        # self.kernel = [16, 32, 64, 128, 128]
        # self.kernel = [8, 16, 32, 64, 64]
        # self.kernel = [2, 4, 8, 16, 16]

        # VGG16
        self.conv1_1 = Conv2dBNReLU(3, self.kernel[0])
        self.conv1_2 = Conv2dBNReLU(self.kernel[0], self.kernel[0])
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)  # 64
        self.conv2_1 = Conv2dBNReLU(self.kernel[0], self.kernel[1])
        self.conv2_2 = Conv2dBNReLU(self.kernel[1], self.kernel[1])
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)  # 128
        self.conv3_1 = Conv2dBNReLU(self.kernel[1], self.kernel[2])
        self.conv3_2 = Conv2dBNReLU(self.kernel[2], self.kernel[2])
        self.conv3_3 = Conv2dBNReLU(self.kernel[2], self.kernel[2])
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv4_1 = Conv2dBNReLU(self.kernel[2], self.kernel[3])
        self.conv4_2 = Conv2dBNReLU(self.kernel[3], self.kernel[3])
        self.conv4_3 = Conv2dBNReLU(self.kernel[3], self.kernel[3])
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv5_1 = Conv2dBNReLU(self.kernel[3], self.kernel[4])
        self.conv5_2 = Conv2dBNReLU(self.kernel[4], self.kernel[4])
        self.conv5_3 = Conv2dBNReLU(self.kernel[4], self.kernel[4])
        self.pool5 = nn.MaxPool2d(kernel_size=2, stride=2)

        # wavelet
        if self.level >= 1:
            self.wavelet1 = WaveletConv2d(3, self.kernel[1] // 4)
            self.reduce1 = Conv2dBNReLU(self.kernel[1] * 2, self.kernel[1])
        if self.level >= 2:
            self.wavelet2 = WaveletConv2d(3, self.kernel[2] // 4)
            self.reduce2 = Conv2dBNReLU(self.kernel[2] * 2, self.kernel[2])
        if self.level >= 3:
            self.wavelet3 = WaveletConv2d(3, self.kernel[3] // 4)
            self.reduce3 = Conv2dBNReLU(self.kernel[3] * 2, self.kernel[3])
        if self.level >= 4:
            self.wavelet4 = WaveletConv2d(3, self.kernel[4] // 4)
            self.reduce4 = Conv2dBNReLU(self.kernel[4] * 2, self.kernel[4])

        # 分类
        self.pool_avg = nn.AvgPool2d(kernel_size=1, stride=1)
        self.classifier = nn.Linear(self.kernel[-1], 10)
        pass

    def forward(self, x):
        out = self.pool1(self.conv1_2(self.conv1_1(x["Input"])))  # 64x16x16

        out = self.conv2_2(self.conv2_1(out))
        if self.level >= 1:
            out_wavelet = self.wavelet1(x["HH_1"], x["HL_1"], x["LH_1"], x["LL_1"])
            out = self.reduce1(torch.cat([out, out_wavelet], 1))
        out = self.pool2(out)  # 128x8x8

        out = self.conv3_3(self.conv3_2(self.conv3_1(out)))
        if self.level >= 2:
            out_wavelet = self.wavelet2(x["HH_2"], x["HL_2"], x["LH_2"], x["LL_2"])
            out = self.reduce2(torch.cat([out, out_wavelet], 1))
        out = self.pool3(out)  # 256x4x4

        out = self.conv4_3(self.conv4_2(self.conv4_1(out)))
        if self.level >= 3:
            out_wavelet = self.wavelet3(x["HH_3"], x["HL_3"], x["LH_3"], x["LL_3"])
            out = self.reduce3(torch.cat([out, out_wavelet], 1))
        out = self.pool4(out)  # 512x2x2

        out = self.conv5_3(self.conv5_2(self.conv5_1(out)))
        if self.level >= 4:
            out_wavelet = self.wavelet4(x["HH_4"], x["HL_4"], x["LH_4"], x["LL_4"])
            out = self.reduce4(torch.cat([out, out_wavelet], 1))
        out = self.pool5(out)  # 512x1x1

        out = self.pool_avg(out)  # 512x1x1
        out = out.view(out.size(0), -1)  # 512
        out = self.classifier(out)  # 10
        return out

    pass


class VGGBlock(nn.Module):

    def __init__(self, in_channels, out_channels, wavelet_fn, conv_more=False, kernel_size=3, padding=1):
        super(VGGBlock, self).__init__()
        self.conv_more = conv_more
        self.conv1 = Conv2dBNReLU(in_channels, out_channels, kernel_size, padding)
        self.conv2 = Conv2dBNReLU(out_channels, out_channels, kernel_size, padding)
        if self.conv_more:
            self.conv3 = Conv2dBNReLU(out_channels, out_channels, kernel_size, padding)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        pass

    def forward(self, inputs):
        out = self.conv1(inputs)
        out = self.conv2(out)
        if self.conv_more:
            out = self.conv3(out)
        out = self.pool(out)
        return out

    def __call__(self, *args, **kwargs):
        return super(VGGBlock, self).__call__(*args, **kwargs)

    pass


class VGGWaveletBlock(nn.Module):

    def __init__(self, in_channels, out_channels, wavelet_fn, conv_more=False, kernel_size=3, padding=1):
        super(VGGWaveletBlock, self).__init__()
        self.wavelet_fn = wavelet_fn
        self.conv_more = conv_more
        self.conv1 = Conv2dBNReLU(in_channels, out_channels, kernel_size, padding)
        self.conv2 = Conv2dBNReLU(out_channels, out_channels, kernel_size, padding)
        if self.conv_more:
            self.conv3 = Conv2dBNReLU(out_channels, out_channels, kernel_size, padding)
        pass

    def forward(self, inputs):
        out = self.conv1(inputs)
        out = self.conv2(out)
        if self.conv_more:
            out = self.conv3(out)
        out = self.wavelet_fn(out, detach=True)
        ll, lh, hl, hh = out["LL_1"], out["LH_1"], out["HL_1"], out["HH_1"]
        return ll, lh, hl, hh

    def __call__(self, *args, **kwargs):
        return super(VGGWaveletBlock, self).__call__(*args, **kwargs)

    pass


class WaveletBlock(nn.Module):

    def __init__(self, in_channels, out_channels, wavelet_fn, conv_more=False, kernel_size=3, padding=1):
        super(WaveletBlock, self).__init__()
        self.wavelet_fn = wavelet_fn
        self.conv_more = conv_more
        self.conv1_ll = Conv2dBNReLU(in_channels, out_channels//4, kernel_size, padding)
        self.conv1_lh = Conv2dBNReLU(in_channels, out_channels//4, kernel_size, padding)
        self.conv1_hl = Conv2dBNReLU(in_channels, out_channels//4, kernel_size, padding)
        self.conv1_hh = Conv2dBNReLU(in_channels, out_channels//4, kernel_size, padding)
        self.conv2 = Conv2dBNReLU(out_channels, out_channels, kernel_size, padding)
        if self.conv_more:
            self.conv3 = Conv2dBNReLU(out_channels, out_channels, kernel_size, padding)
        pass

    def forward(self, inputs):
        ll, lh, hl, hh = inputs
        ll = self.conv1_ll(ll)
        lh = self.conv1_lh(lh)
        hl = self.conv1_hl(hl)
        hh = self.conv1_hh(hh)
        out = torch.cat([ll, lh, hl, hh], 1)
        out = self.conv2(out)
        if self.conv_more:
            out = self.conv3(out)

        out = self.wavelet_fn(out, detach=True)
        ll, lh, hl, hh = out["LL_1"], out["LH_1"], out["HL_1"], out["HH_1"]
        return ll, lh, hl, hh

    def __call__(self, *args, **kwargs):
        return super(WaveletBlock, self).__call__(*args, **kwargs)

    pass


class WaveletVGGBlock(nn.Module):

    def __init__(self, in_channels, out_channels, wavelet_fn, conv_more=False, kernel_size=3, padding=1):
        super(WaveletVGGBlock, self).__init__()
        self.wavelet_fn = wavelet_fn
        self.conv_more = conv_more
        self.conv1_ll = Conv2dBNReLU(in_channels, out_channels//4, kernel_size, padding)
        self.conv1_lh = Conv2dBNReLU(in_channels, out_channels//4, kernel_size, padding)
        self.conv1_hl = Conv2dBNReLU(in_channels, out_channels//4, kernel_size, padding)
        self.conv1_hh = Conv2dBNReLU(in_channels, out_channels//4, kernel_size, padding)
        self.conv2 = Conv2dBNReLU(out_channels, out_channels, kernel_size, padding)
        if self.conv_more:
            self.conv3 = Conv2dBNReLU(out_channels, out_channels, kernel_size, padding)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        pass

    def forward(self, inputs):
        ll, lh, hl, hh = inputs
        ll = self.conv1_ll(ll)
        lh = self.conv1_lh(lh)
        hl = self.conv1_hl(hl)
        hh = self.conv1_hh(hh)
        out = torch.cat([ll, lh, hl, hh], 1)
        out = self.conv2(out)
        if self.conv_more:
            out = self.conv3(out)
        out = self.pool(out)
        return out

    def __call__(self, *args, **kwargs):
        return super(WaveletVGGBlock, self).__call__(*args, **kwargs)

    pass


class WaveletVGG2(nn.Module):

    def __init__(self, level=0, wavelet_fn=None):
        super(WaveletVGG2, self).__init__()

        self.level = level
        self.wavelet_fn = wavelet_fn
        self.kernel = [64, 128, 256, 512, 512]

        config = [[0, 0, 0, 0, 0],  # VGG16
                  [1, 3, 0, 0, 0],  # Level 1
                  [1, 2, 3, 0, 0],  # Level 2
                  [1, 2, 2, 3, 0],  # Level 3
                  [1, 2, 2, 2, 3]]  # Level 4
        block = [VGGBlock, VGGWaveletBlock, WaveletBlock, WaveletVGGBlock]

        self.block1 = block[config[self.level][0]](3, self.kernel[0], self.wavelet_fn, conv_more=False)
        self.block2 = block[config[self.level][1]](self.kernel[0], self.kernel[1], self.wavelet_fn, conv_more=False)
        self.block3 = block[config[self.level][2]](self.kernel[1], self.kernel[2], self.wavelet_fn, conv_more=True)
        self.block4 = block[config[self.level][3]](self.kernel[2], self.kernel[3], self.wavelet_fn, conv_more=True)
        self.block5 = block[config[self.level][4]](self.kernel[3], self.kernel[4], self.wavelet_fn, conv_more=True)

        # 分类
        self.pool_avg = nn.AvgPool2d(kernel_size=1, stride=1)
        self.classifier = nn.Linear(self.kernel[-1], 10)
        pass

    def forward(self, x):
        out = self.block1(x)
        out = self.block2(out)
        out = self.block3(out)
        out = self.block4(out)
        out = self.block5(out)

        out = self.pool_avg(out)  # 512x1x1
        out = out.view(out.size(0), -1)  # 512
        out = self.classifier(out)  # 10
        return out

    pass
