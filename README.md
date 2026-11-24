# RetinaAI — Retinal Disease Detection Using Deep Learning on OCT Images

**RetinaAI** is a deep learning-based retinal disease classification system that analyzes Optical Coherence Tomography (OCT) images to identify four retinal conditions: **CNV, DME, DRUSEN, and NORMAL**.

The project implements and compares five deep learning architectures and integrates Grad-CAM (Gradient-weighted Class Activation Mapping) to provide visual explanations of model predictions.

---

## Project Overview

Retinal diseases can affect vision and require accurate medical evaluation. Optical Coherence Tomography (OCT) provides cross-sectional images of the retina that can help identify structural abnormalities.

RetinaAI explores how deep learning can assist in classifying retinal OCT images through automated image analysis, model comparison, and visual explainability.

**The application allows users to:**
- Upload retinal OCT images.
- Select between five trained deep learning models.
- Predict the retinal image category.
- View prediction confidence and class probabilities.
- Generate Grad-CAM heatmaps to visualize important image regions.
- Compare predictions from all five models.
- Explore model performance through an interactive dashboard.

## Disease Categories

The system classifies OCT images into four categories:

| Class | Description |
|---|---|
| CNV | Choroidal Neovascularization |
| DME | Diabetic Macular Edema |
| DRUSEN | Drusen |
| NORMAL | Normal retinal OCT image |

## Deep Learning Architectures

Five architectures were implemented and evaluated.

| Model | Architecture Type | Parameters |
|---|---|---:|
| Custom CNN | Convolutional Neural Network | 110,724 |
| ResNet18 | Residual Network | 11.18M |
| VGG16 | Deep Convolutional Network | 138.36M |
| MobileNetV2 | Lightweight CNN | 2.23M |
| DenseNet121 | Densely Connected CNN | 6.96M |

## Model Performance

All models were evaluated using the same held-out test dataset containing 1,000 OCT images.

| Model | Accuracy | Macro Precision | Macro Recall | Macro F1-Score |
|---|---:|---:|---:|---:|
| Custom CNN | 87.30% | 89.00% | 87.00% | 87.00% |
| ResNet18 | 95.60% | 95.75% | 95.60% | 95.50% |
| VGG16 | 95.80% | 96.00% | 95.80% | 95.80% |
| MobileNetV2 | 93.80% | 94.00% | 93.80% | 93.80% |
| DenseNet121 | 94.40% | 94.75% | 94.40% | 94.25% |

*Metrics are rounded to two decimal places.*

**Key observations:**
- VGG16 achieved the highest test accuracy at 95.80%.
- ResNet18 delivered comparable performance with substantially fewer parameters.
- MobileNetV2 offers a relatively lightweight architecture.
- Custom CNN has the smallest parameter count.
- DRUSEN classification presented comparatively lower recall across the evaluated models.

## Explainable AI — Grad-CAM

RetinaAI integrates Grad-CAM to provide visual explanations of predictions.

Grad-CAM generates class-specific heatmaps using gradients flowing into the final convolutional feature maps.

This helps visualize which image regions contributed to the model's prediction.

The application displays:
- Original OCT image.
- Grad-CAM heatmap.
- Heatmap overlay on the original image.

This provides additional interpretability beyond prediction confidence alone.

## Application Features

### Image Classification
Upload an OCT image and obtain a predicted category with its confidence score.

### Model Selection
Choose between Custom CNN, ResNet18, VGG16, MobileNetV2, and DenseNet121.

### Model Comparison
Run all five models on the same uploaded image and compare their predictions and confidence scores.

### Model Performance Dashboard
Explore:
- Overall accuracy comparison.
- Precision, recall, and F1-score.
- Model parameter counts.
- Confusion matrices.
- Class-wise performance.
- Cross-model class-wise comparisons.

### Project Information
A dedicated page explains the dataset, architectures, methodology, workflow, and technical implementation.

## Dataset

**Dataset:** Kermany OCT Dataset (Kermany2017-OCT)

The project uses:
- 108,309 training images.
- 1,000 held-out test images.
- Four classification categories.

The training dataset was divided into training and validation subsets using a stratified 90/10 split.

Images were resized to 224 × 224 pixels and converted to three-channel inputs.

Dataset preprocessing and normalization were configured according to the model architecture.

## Technology Stack

| Category | Technologies |
|---|---|
| Programming Language | Python |
| Deep Learning | PyTorch, Torchvision |
| Image Processing | Pillow, OpenCV |
| Data Processing | NumPy, Pandas |
| Model Evaluation | Scikit-learn |
| Visualization | Matplotlib |
| Web Application | Streamlit |
| Explainable AI | Grad-CAM |
| Development Environment | Jupyter Notebook, VS Code |

## Project Structure

```text
RetinaAI/
│
├── app.py
│
├── pages/
│   ├── 1_Model_Performance.py
│   └── 2_About_Project.py
│
├── src/
│   ├── __init__.py
│   ├── model.py
│   ├── preprocessing.py
│   └── prediction.py
│
├── modules/
│   └── Model checkpoints (download separately)
│
├── evaluation_results/
│   ├── custom_cnn.json
│   ├── resnet18.json
│   ├── vgg16.json
│   ├── mobilenetv2.json
│   └── densenet121.json
│
├── sample_images/
│   ├── CNV/
│   ├── DME/
│   ├── DRUSEN/
│   └── NORMAL/
│
├── notebooks/
│
├── requirements.txt
├── .gitignore
└── README.md
```

## Installation and Setup

### 1. Clone the repository

```bash
git clone YOUR_GITHUB_REPOSITORY_URL
cd RetinaAI
```

### 2. Create a virtual environment

```bash
python -m venv .venv
```

Activate it on Windows:

```bash
.venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Download trained model checkpoints

The trained model checkpoints are hosted separately on Google Drive because of their file sizes.

**[[Download RetinaAI Model Checkpoints — GOOGLE DRIVE LINK TO BE ADDED]](https://drive.google.com/drive/folders/1UAOd_yxYmWgnoUr1iC9a8Mht9HNPUA2C?usp=sharing)**

Download all five `.pth` files and place them inside the `modules/` directory.

Ensure the filenames match the model configuration in the source code.

### 5. Run the application

```bash
streamlit run app.py
```

The application will open in your browser.

## Disclaimer

RetinaAI is an educational and experimental deep learning project developed for research and learning purposes.

It is not a clinically validated medical diagnostic system and must not be used as a substitute for professional medical examination, diagnosis, or treatment.

Predictions may be inaccurate. Always consult qualified healthcare professionals for medical decisions.

---

**Developed as a Deep Learning and Explainable AI Project.**

**RetinaAI | Retinal Disease Detection Using Deep Learning on OCT Images**
