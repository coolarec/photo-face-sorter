# network.py
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights

class FaceClassifier(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.backbone = resnet18(weights=ResNet18_Weights.DEFAULT)
        for name, param in self.backbone.named_parameters():
            if "layer4" not in name:  # 只训练最后几层
                param.requires_grad = False
        self.backbone.fc = nn.Sequential(  # 替换最后一层用于分类
            nn.Linear(self.backbone.fc.in_features, 256),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        return self.backbone(x)
