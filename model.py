import torch
import torch.nn as nn

class BasicBlock(nn.Module):
    def __init__(self, in_planes, planes):
        super(BasicBlock, self).__init__()
        # 논문 조건: stride를 제거하여 항상 1로 고정
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        # 논문 조건: ReLU 대신 Sigmoid 사용
        self.act = nn.Sigmoid() 
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if in_planes != planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes, kernel_size=1, stride=1, bias=False),
                nn.BatchNorm2d(planes)
            )

    def forward(self, x):
        out = self.act(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = self.act(out)
        return out

class ModifiedResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=100):
        super(ModifiedResNet, self).__init__()
        self.in_planes = 16

        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.act = nn.Sigmoid()
        
        self.layer1 = self._make_layer(block, 16, num_blocks[0])
        self.layer2 = self._make_layer(block, 32, num_blocks[1])
        self.layer3 = self._make_layer(block, 64, num_blocks[2])
        
        # Stride가 없으므로 이미지 크기(32x32)가 끝까지 유지됨
        self.linear = nn.Linear(64 * 32 * 32, num_classes)

    def _make_layer(self, block, planes, num_blocks):
        layers = []
        for _ in range(num_blocks):
            layers.append(block(self.in_planes, planes))
            self.in_planes = planes
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.act(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out

def get_resnet56_modified(num_classes=100):
    """논문에서 사용한 원본 ResNet-56 구조"""
    return ModifiedResNet(BasicBlock, [9, 9, 9], num_classes)

def get_resnet8_modified(num_classes=100):
    """로컬 테스트를 위한 가벼운 ResNet 구조"""
    return ModifiedResNet(BasicBlock, [1, 1, 1], num_classes)