import torch.nn as nn
from torchvision.models.video import r3d_18, R3D_18_Weights

def initialize_model(num_classes=2, use_pretrained=True):
    weights = R3D_18_Weights.DEFAULT if use_pretrained else None
    model = r3d_18(weights=weights)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    model.stem[0] = nn.Conv3d(1, 64, kernel_size=(3, 7, 7), stride=(1, 2, 2), padding=(1, 3, 3), bias=False)
    return model