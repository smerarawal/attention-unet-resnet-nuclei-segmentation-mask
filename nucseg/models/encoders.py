"""Both encoders return 5 feature maps and expose `out_channels` and `first_stride`."""
import warnings

import torch
import torch.nn as nn

from .blocks import ResidualBlock, conv_bn_relu


class ScratchEncoder(nn.Module):
    """Your original residual encoder (trained from scratch): strides 1,2,4,8,16."""
    first_stride = 1

    def __init__(self, in_ch=3, filters=(32, 64, 128, 256, 512)):
        super().__init__()
        f = filters
        self.out_channels = list(f)
        self.stem = nn.Sequential(conv_bn_relu(in_ch, f[0]), ResidualBlock(f[0], f[0]))
        self.stages = nn.ModuleList([
            nn.Sequential(ResidualBlock(f[i], f[i + 1], 2), ResidualBlock(f[i + 1], f[i + 1])) for i in range(4)
        ])

    def forward(self, x):
        feats = [self.stem(x)]
        for s in self.stages:
            feats.append(s(feats[-1]))
        return feats


class ResNet34Encoder(nn.Module):
    """torchvision ResNet-34, strides 2,4,8,16,32. ImageNet init if available (offline: weights_path)."""
    first_stride = 2

    def __init__(self, pretrained=True, weights_path=""):
        super().__init__()
        import torchvision
        net = torchvision.models.resnet34(weights=None)
        if weights_path:
            net.load_state_dict(torch.load(weights_path, map_location="cpu"))
            print(f"[model] loaded encoder weights from {weights_path}")
        elif pretrained:
            try:
                from torchvision.models import ResNet34_Weights
                net = torchvision.models.resnet34(weights=ResNet34_Weights.IMAGENET1K_V1)
                print("[model] ImageNet-pretrained ResNet-34 loaded")
            except Exception as e:  # Kaggle internet off -> don't crash, but be loud
                warnings.warn(f"Pretrained weights unavailable ({type(e).__name__}); using RANDOM init. "
                              "Turn Internet on, or upload resnet34 weights and set encoder_weights_path.")
        self.l0 = nn.Sequential(net.conv1, net.bn1, net.relu)
        self.l1 = nn.Sequential(net.maxpool, net.layer1)
        self.l2, self.l3, self.l4 = net.layer2, net.layer3, net.layer4
        self.out_channels = [64, 64, 128, 256, 512]

    def forward(self, x):
        f0 = self.l0(x)
        f1 = self.l1(f0)
        f2 = self.l2(f1)
        f3 = self.l3(f2)
        f4 = self.l4(f3)
        return [f0, f1, f2, f3, f4]


def build_encoder(name, pretrained=True, weights_path="", base_ch=32):
    if name == "scratch":
        return ScratchEncoder(filters=(base_ch, base_ch * 2, base_ch * 4, base_ch * 8, base_ch * 16))
    if name == "resnet34":
        return ResNet34Encoder(pretrained, weights_path)
    raise ValueError(f"unknown encoder {name!r}")
