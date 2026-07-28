import torch.nn as nn
from torchvision.models import resnet50, ResNet50_Weights


# ResNet-50 backbone fine-tuned for binary real/AI classification
class CNNDetector(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = resnet50(weights=ResNet50_Weights.DEFAULT)
        in_features = self.backbone.fc.in_features
        # Replace the classifier head with dropout + single binary output
        self.backbone.fc = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(in_features, 1)
        )

    def forward(self, x):
        return self.backbone(x)
