import os
import pickle

import torch
import torch.nn as nn
import torch.nn.functional as F


def conv3x3(in_planes, out_planes, strd=1, padding=1, bias=False):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=strd, padding=padding, bias=bias)


class ConvBlock(nn.Module):
    def __init__(self, in_planes, out_planes):
        super(ConvBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = conv3x3(in_planes, out_planes // 2)
        self.bn2 = nn.BatchNorm2d(out_planes // 2)
        self.conv2 = conv3x3(out_planes // 2, out_planes // 4)
        self.bn3 = nn.BatchNorm2d(out_planes // 4)
        self.conv3 = conv3x3(out_planes // 4, out_planes // 4)

        if in_planes != out_planes:
            self.downsample = nn.Sequential(nn.BatchNorm2d(in_planes), nn.ReLU(True),
                                            nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=1, bias=False))
        else:
            self.downsample = None

    def forward(self, x, name=None, debug=False):
        residual = x

        out1 = self.bn1(x)  # this line is giving different results
        if debug:
            with open(os.path.join("batch_norm_test_resources", torch.__version__ + "_log.out"), "a") as file:
                file.write(f"Convblock: {name}.bn1 \n x: {x[0,0,0]} \n out1: {out1[0,0,0]} \n\n")
        out1 = F.relu(out1, True)
        out1 = self.conv1(out1)
        if debug:
            with open(os.path.join("batch_norm_test_resources", torch.__version__ + "_log.out"), "a") as file:
                file.write(f"Convblock: {name}.conv1 \n x: {x[0,0,0]} \n out1: {out1[0,0,0]} \n\n")

        out2 = self.bn2(out1)
        if debug:
            with open(os.path.join("batch_norm_test_resources", torch.__version__ + "_log.out"), "a") as file:
                file.write(f"Convblock: {name}.bn2 \n x: {x[0,0,0]} \n out2: {out2[0,0,0]} \n\n")
        out2 = F.relu(out2, True)
        out2 = self.conv2(out2)
        if debug:
            with open(os.path.join("batch_norm_test_resources", torch.__version__ + "_log.out"), "a") as file:
                file.write(f"Convblock: {name}.conv2 \n x: {x[0,0,0]} \n out2: {out2[0,0,0]} \n\n")

        out3 = self.bn3(out2)
        if debug:
            with open(os.path.join("batch_norm_test_resources", torch.__version__ + "_log.out"), "a") as file:
                file.write(f"Convblock: {name}.bn3 \n x: {x[0,0,0]} \n out3: {out3[0,0,0]} \n\n")
        out3 = F.relu(out3, True)
        out3 = self.conv3(out3)
        if debug:
            with open(os.path.join("batch_norm_test_resources", torch.__version__ + "_log.out"), "a") as file:
                file.write(f"Convblock: {name}.conv3 \n x: {x[0,0,0]} \n out3: {out3[0,0,0]} \n\n")

        out3 = torch.cat((out1, out2, out3), 1)

        if self.downsample is not None:
            residual = self.downsample(residual)
        if debug:
            with open(os.path.join("batch_norm_test_resources", torch.__version__ + "_log.out"), "a") as file:
                file.write(f"Convblock: {name}.downsample \n x: {x[0,0,0]} \n residual: {residual[0,0,0]} \n\n")

        out3 += residual

        return out3


class HourGlass(nn.Module):
    def __init__(self, config, name):
        super(HourGlass, self).__init__()
        self.config = config
        self.name = name

        self._generate_network(self.config.hg_depth)

    def _generate_network(self, level):
        self.add_module('b1_' + str(level), ConvBlock(self.config.hg_num_features, self.config.hg_num_features))

        self.add_module('b2_' + str(level), ConvBlock(self.config.hg_num_features, self.config.hg_num_features))

        if level > 1:
            self._generate_network(level - 1)
        else:
            self.add_module('b2_plus_' + str(level),ConvBlock(self.config.hg_num_features,
                                                              self.config.hg_num_features))

        self.add_module('b3_' + str(level), ConvBlock(self.config.hg_num_features, self.config.hg_num_features))

    def _forward(self, level, inp):
        up1 = inp

        up1 = self._modules['b1_' + str(level)](up1)

        if self.config.use_avg_pool:
            low1 = F.avg_pool2d(inp, 2, stride=2)
        else:
            low1 = F.max_pool2d(inp, 2, stride=2)
        low1 = self._modules['b2_' + str(level)](low1)
        # same up till here

        if level > 1:
            low2 = self._forward(level - 1, low1)  # correct at level 2 but wrong at level 3
        else:
            low2 = low1
            low2 = self._modules['b2_plus_' + str(level)](low2)  # still the same

        low3 = low2
        debug = level == 1
        low3 = self._modules['b3_' + str(level)](low3, name='b3_' + str(level), debug=debug)  # first difference I see is here, at level 2

        if debug:
            tensor_snapshot = os.path.join("batch_norm_test_resources", f"{self.name}_{level}_before_interpolate.pickle")
            if os.path.exists(tensor_snapshot):
                with open(tensor_snapshot, "rb") as file:
                    saved_tensor = pickle.load(file)
                    print(f"\n\n>>>>>> Is equal to snapshot? {tensor_snapshot}", torch.equal(low3, saved_tensor))
            with open(tensor_snapshot, "wb") as file:
                pickle.dump(low3, file)
        up2 = F.interpolate(low3, scale_factor=2, mode='nearest')
        if debug:
            tensor_snapshot = os.path.join("batch_norm_test_resources", f"{self.name}_{level}_after_interpolate.pickle")
            if os.path.exists(tensor_snapshot):
                with open(tensor_snapshot, "rb") as file:
                    saved_tensor = pickle.load(file)
                    print(f"\n\n>>>>>> Is equal to snapshot? {tensor_snapshot}", torch.equal(up2, saved_tensor))
            with open(tensor_snapshot, "wb") as file:
                pickle.dump(up2, file)
        with open(os.path.join("batch_norm_test_resources", torch.__version__ + "_log.out"), "a") as file:
            file.write(f"interpolate: \n low3: {low3[0,0,0]} \n up2: {up2[0,0,0]} \n\n")

        return up1 + up2

    def forward(self, x):
        return self._forward(self.config.hg_depth, x)


class FAN(nn.Module):
    def __init__(self, config):
        super(FAN, self).__init__()
        self.config = config

        # Stem
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = ConvBlock(64, 128)
        self.conv3 = ConvBlock(128, 128)
        self.conv4 = ConvBlock(128, self.config.hg_num_features)

        # Hourglasses
        for hg_module in range(self.config.num_modules):
            self.add_module('m' + str(hg_module), HourGlass(self.config, name='m' + str(hg_module)))
            self.add_module('top_m_' + str(hg_module), ConvBlock(self.config.hg_num_features,
                                                                 self.config.hg_num_features))
            self.add_module('conv_last' + str(hg_module), nn.Conv2d(self.config.hg_num_features,
                                                                    self.config.hg_num_features,
                                                                    kernel_size=1, stride=1, padding=0))
            self.add_module('bn_end' + str(hg_module), nn.BatchNorm2d(self.config.hg_num_features))
            self.add_module('l' + str(hg_module), nn.Conv2d(self.config.hg_num_features, 68,
                                                            kernel_size=1, stride=1, padding=0))

            if hg_module < self.config.num_modules - 1:
                self.add_module('bl' + str(hg_module), nn.Conv2d(self.config.hg_num_features,
                                                                 self.config.hg_num_features,
                                                                 kernel_size=1, stride=1, padding=0))
                self.add_module('al' + str(hg_module), nn.Conv2d(68, self.config.hg_num_features,
                                                                 kernel_size=1, stride=1, padding=0))

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)), True)
        if self.config.use_avg_pool:
            x = F.avg_pool2d(self.conv2(x), 2, stride=2)
        else:
            x = F.max_pool2d(self.conv2(x), 2, stride=2)
        x = self.conv3(x)
        x = self.conv4(x)

        previous = x
        hg_feats = []
        tmp_out = None
        for i in range(self.config.num_modules):
            hg = self._modules['m' + str(i)](previous)

            ll = hg
            ll = self._modules['top_m_' + str(i)](ll)

            ll = F.relu(self._modules['bn_end' + str(i)](self._modules['conv_last' + str(i)](ll)), True)

            # Predict heatmaps
            tmp_out = self._modules['l' + str(i)](ll)

            if i < self.config.num_modules - 1:
                ll = self._modules['bl' + str(i)](ll)
                tmp_out_ = self._modules['al' + str(i)](tmp_out)
                previous = previous + ll + tmp_out_

            hg_feats.append(ll)

        return tmp_out, x, tuple(hg_feats)
