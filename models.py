import torch.nn as nn
from torchvision import models

def get_mobilenet_v2(num_classes=10):
    # 가벼운 실험을 위해 사전학습 가중치 제외
    model = models.mobilenet_v2(weights=None)
    model.classifier[1] = nn.Linear(model.last_channel, num_classes)
    return model