
import torch
import torch.nn as nn

from torchvision.models import (
    resnet18,
    vgg16,
    ResNet18_Weights,
    VGG16_Weights
)

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


def load_resnet18(checkpoint_path):
    model = resnet18(weights=None)

    model.fc = nn.Linear(
        model.fc.in_features,
        4
    )

    checkpoint = torch.load(
        checkpoint_path,
        map_location=DEVICE,
        weights_only=True
    )

    model.load_state_dict(checkpoint)
    model = model.to(DEVICE)
    model.eval()

    return model


def load_vgg16(checkpoint_path):
    model = vgg16(weights=None)

    model.classifier[6] = nn.Linear(
        model.classifier[6].in_features,
        4
    )

    checkpoint = torch.load(
        checkpoint_path,
        map_location=DEVICE,
        weights_only=True
    )

    model.load_state_dict(checkpoint)
    model = model.to(DEVICE)
    model.eval()

    return model
