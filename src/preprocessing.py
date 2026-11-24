
from PIL import Image
from torchvision import transforms


CLASS_NAMES = [
    "CNV",
    "DME",
    "DRUSEN",
    "NORMAL"
]


def preprocess_image(image, model_name):

    if image.mode != "L":
        image = image.convert("L")

    if model_name == "Custom CNN":
        normalize = transforms.Normalize(
            mean=[0.2022, 0.2022, 0.2022],
            std=[0.2133, 0.2133, 0.2133]
        )
    else:
        normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
        normalize
    ])

    image_tensor = transform(image)

    return image_tensor.unsqueeze(0)
