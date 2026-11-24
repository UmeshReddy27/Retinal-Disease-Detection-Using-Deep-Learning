import torch.nn.functional as F
import numpy as np
from pathlib import Path

import torch
import torch.nn as nn

from torchvision.models import (
    resnet18,
    vgg16,
    mobilenet_v2,
    densenet121
)

from src.preprocessing import CLASS_NAMES


DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

MODEL_DIR = Path(__file__).resolve().parent.parent / "modules"



class CustomCNN(nn.Module):
    def __init__(self, num_classes=4):
        super().__init__()

        self.conv_block1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        self.conv_block2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        self.conv_block3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        self.pool = nn.AdaptiveAvgPool2d(1)

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.conv_block1(x)
        x = self.conv_block2(x)
        x = self.conv_block3(x)
        x = self.pool(x)
        x = self.classifier(x)
        return x


MODEL_CONFIGS = {
    "Custom CNN": {
        "checkpoint": "custom_cnn1_best.pth",
        "architecture": "custom"
    },

    "ResNet18": {
        "checkpoint": "resnet18_finetuned_best.pth",
        "architecture": "resnet"
    },

    "VGG16": {
        "checkpoint": "vgg16_finetuned_best.pth",
        "architecture": "vgg"
    },

    "MobileNetV2": {
        "checkpoint": "mobilenetv2_finetuned_best.pth",
        "architecture": "mobile"
    },

    "DenseNet121": {
        "checkpoint": "densenet121_finetuned_best.pth",
        "architecture": "dense"
    }
}


def build_model(architecture):
    if architecture == "custom":
        model = CustomCNN(num_classes=4)

    elif architecture == "resnet":
        model = resnet18(weights=None)
        model.fc = nn.Linear(
            model.fc.in_features,
            4
        )

    elif architecture == "vgg":
        model = vgg16(weights=None)
        model.classifier[6] = nn.Linear(
            model.classifier[6].in_features,
            4
        )

    elif architecture == "mobile":
        model = mobilenet_v2(weights=None)
        model.classifier[1] = nn.Linear(
            model.classifier[1].in_features,
            4
        )

    elif architecture == "dense":
        model = densenet121(weights=None)
        model.classifier = nn.Linear(
            model.classifier.in_features,
            4
        )

    else:
        raise ValueError(
            f"Unknown architecture: {architecture}"
        )

    return model


def load_model(model_name):
    if model_name not in MODEL_CONFIGS:
        raise ValueError(
            f"Unknown model name: {model_name}"
        )

    config = MODEL_CONFIGS[model_name]

    checkpoint_path = MODEL_DIR / config["checkpoint"]

    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {checkpoint_path}"
        )

    model = build_model(config["architecture"])

    checkpoint = torch.load(
        checkpoint_path,
        map_location=DEVICE,
        weights_only=True
    )

    if "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    else:
        state_dict = checkpoint

    model.load_state_dict(state_dict)

    model = model.to(DEVICE)
    model.eval()

    return model


def predict_image(model, image_tensor):
    image_tensor = image_tensor.to(DEVICE)

    with torch.inference_mode():
        outputs = model(image_tensor)

        probabilities = torch.softmax(
            outputs,
            dim=1
        )

        confidence, predicted_index = torch.max(
            probabilities,
            dim=1
        )

    predicted_class = CLASS_NAMES[
        predicted_index.item()
    ]

    confidence = confidence.item()

    probabilities = probabilities.squeeze(0).cpu().tolist()

    probability_dict = {
        CLASS_NAMES[i]: probabilities[i]
        for i in range(len(CLASS_NAMES))
    }

    return predicted_class, confidence, probability_dict



def generate_gradcam(model, image_tensor, model_name, class_index=None):

    if model_name == "Custom CNN":
        target_layer = model.conv_block3

    elif model_name == "ResNet18":
        target_layer = model.layer4[-1]

    elif model_name == "VGG16":
        target_layer = model.features[28]

    elif model_name == "MobileNetV2":
        target_layer = model.features[-1]

    elif model_name == "DenseNet121":
        target_layer = model.features.denseblock4

    else:
        raise ValueError(f"Unsupported model: {model_name}")

    activations = []
    gradients = []

    def forward_hook(module, inputs, output):
        activations.append(output)

        output.register_hook(
            lambda grad: gradients.append(grad)
        )

    hook = target_layer.register_forward_hook(forward_hook)

    try:
        model.zero_grad(set_to_none=True)

        with torch.enable_grad():

            input_tensor = image_tensor.to(DEVICE).detach()
            input_tensor.requires_grad_(True)

            outputs = model(input_tensor)

            if class_index is None:
                class_index = outputs.argmax(dim=1).item()

            score = outputs[0, class_index]

            score.backward()

            activation = activations[0]
            gradient = gradients[0]

            weights = gradient.mean(
                dim=(2, 3),
                keepdim=True
            )

            cam = (weights * activation).sum(dim=1)

            cam = torch.relu(cam)

            cam = F.interpolate(
                cam.unsqueeze(1),
                size=(224, 224),
                mode="bilinear",
                align_corners=False
            )

            cam = cam.squeeze()

            cam = cam - cam.min()
            cam = cam / (cam.max() + 1e-8)

            return cam.detach().cpu().numpy()

    finally:
        hook.remove()
        model.zero_grad(set_to_none=True)
