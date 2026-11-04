"""
U-Net with a pretrained ResNet-18 encoder (torchvision). Needs torch + torchvision (Colab has both).

Encoder: ImageNet-pretrained torchvision resnet18, fine-tuned end to end together with the decoder.
Decoder: four up-sampling stages with skip connections from conv1/layer1/layer2/layer3, then a final
x2 up-sampling and a 1x1 classifier. Input side must be a multiple of 32 (the project uses 512).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision


def _block(cin: int, cout: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(cin, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
        nn.Conv2d(cout, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True))


class UNetResNet18(nn.Module):
    def __init__(self, n_classes: int = 5, pretrained: bool = True):
        super().__init__()
        weights = torchvision.models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        r = torchvision.models.resnet18(weights=weights)
        self.stem = nn.Sequential(r.conv1, r.bn1, r.relu)          # 1/2,  64 ch
        self.pool = r.maxpool                                       # 1/4
        self.layer1, self.layer2, self.layer3, self.layer4 = r.layer1, r.layer2, r.layer3, r.layer4  # 64,128,256,512
        self.dec4 = _block(512 + 256, 256)                          # 1/16
        self.dec3 = _block(256 + 128, 128)                          # 1/8
        self.dec2 = _block(128 + 64, 64)                            # 1/4
        self.dec1 = _block(64 + 64, 64)                             # 1/2
        self.head = nn.Sequential(nn.Conv2d(64, 32, 3, padding=1), nn.ReLU(inplace=True), nn.Conv2d(32, n_classes, 1))
        self.n_classes = n_classes

    @staticmethod
    def _up(x: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
        return F.interpolate(x, size=ref.shape[-2:], mode="bilinear", align_corners=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        size = x.shape[-2:]
        x0 = self.stem(x)
        l1 = self.layer1(self.pool(x0))
        l2 = self.layer2(l1)
        l3 = self.layer3(l2)
        l4 = self.layer4(l3)
        d = self.dec4(torch.cat([self._up(l4, l3), l3], 1))
        d = self.dec3(torch.cat([self._up(d, l2), l2], 1))
        d = self.dec2(torch.cat([self._up(d, l1), l1], 1))
        d = self.dec1(torch.cat([self._up(d, x0), x0], 1))
        return F.interpolate(self.head(d), size=size, mode="bilinear", align_corners=False)


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def normalise(batch_u8_nchw: torch.Tensor) -> torch.Tensor:
    """uint8 RGB (N,3,H,W) -> float normalised with the ImageNet statistics the encoder was pretrained with."""
    x = batch_u8_nchw.float() / 255.0
    mean = torch.tensor(IMAGENET_MEAN, device=x.device).view(1, 3, 1, 1)
    std = torch.tensor(IMAGENET_STD, device=x.device).view(1, 3, 1, 1)
    return (x - mean) / std
