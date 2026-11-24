
import streamlit as st

st.set_page_config(
    page_title="About Project | RetinaAI",
    page_icon="👁️",
    layout="wide"
)

st.markdown("""
<style>
    .block-container {
        padding-top: 2.5rem;
        padding-bottom: 3rem;
        max-width: 1400px;
    }

    .page-title {
        font-size: 38px;
        font-weight: 750;
        margin-bottom: 5px;
    }

    .subtitle {
        color: #9299a5;
        font-size: 16px;
        margin-bottom: 25px;
    }

    .info-card {
        padding: 20px;
        border-radius: 14px;
        background: rgba(128,128,128,0.10);
        border: 1px solid rgba(128,128,128,0.18);
        min-height: 120px;
    }

    .info-label {
        color: #9299a5;
        font-size: 14px;
    }

    .info-value {
        font-size: 25px;
        font-weight: 750;
        margin-top: 8px;
    }
</style>
""", unsafe_allow_html=True)


st.markdown(
    '<p class="page-title">About RetinaAI</p>',
    unsafe_allow_html=True
)

st.markdown(
    '<p class="subtitle">'
    'Project overview, methodology, dataset and technical architecture.'
    '</p>',
    unsafe_allow_html=True
)


st.divider()

st.header("Project Overview")

st.write("""
RetinaAI is a deep learning-based retinal disease classification
project developed using Optical Coherence Tomography (OCT) images.

The system uses five different convolutional neural network
architectures to classify retinal OCT scans into four categories.

It also incorporates Grad-CAM (Gradient-weighted Class Activation
Mapping) to visualize image regions that contribute to a model's
prediction.
""")

col1, col2 = st.columns(2)

with col1:

    st.subheader("Problem Statement")

    st.write("""
    Retinal OCT images contain structural information about the
    retina. Analyzing these images manually can be time-consuming.

    This project explores how deep learning models can assist
    with automated image classification and provide visual
    explanations of their predictions.
    """)

with col2:

    st.subheader("Project Objectives")

    st.write("""
    - Develop an automated OCT image classification pipeline.
    - Train and evaluate multiple CNN architectures.
    - Compare model performance using evaluation metrics.
    - Apply transfer learning to pretrained architectures.
    - Generate Grad-CAM visual explanations.
    - Develop an interactive Streamlit application.
    """)


st.divider()

st.header("Dataset Information")

st.write("""
The project uses the Kermany retinal OCT dataset,
containing images from four classification categories.
""")

metric1, metric2, metric3 = st.columns(3)

with metric1:
    st.metric("Training Images", "108,309")

with metric2:
    st.metric("Test Images", "1,000")

with metric3:
    st.metric("Image Input Size", "224 × 224")


st.caption(
    "The training dataset was divided into training and validation "
    "subsets using a stratified split."
)


st.subheader("Classification Categories")

disease_col1, disease_col2 = st.columns(2)

with disease_col1:

    st.markdown("### CNV")

    st.write(
        "Choroidal Neovascularization — a retinal condition "
        "involving abnormal blood vessel growth beneath the retina."
    )

    st.markdown("### DME")

    st.write(
        "Diabetic Macular Edema — fluid accumulation in the "
        "macula associated with diabetic retinopathy."
    )

with disease_col2:

    st.markdown("### DRUSEN")

    st.write(
        "Deposits that accumulate beneath the retina and are "
        "associated with age-related macular degeneration."
    )

    st.markdown("### NORMAL")

    st.write(
        "OCT images classified as normal, without the target "
        "abnormalities represented by the other categories."
    )


st.divider()

st.header("Deep Learning Architectures")

models = [
    ("Custom CNN", "A custom convolutional neural network developed for OCT image classification."),
    ("ResNet18", "A residual learning architecture using skip connections."),
    ("VGG16", "A deep convolutional network using stacked convolution layers."),
    ("MobileNetV2", "A lightweight architecture using inverted residual blocks and depthwise convolutions."),
    ("DenseNet121", "A densely connected architecture that reuses feature information across layers.")
]

for name, description in models:

    with st.expander(name):

        st.write(description)


st.divider()

st.header("Project Workflow")

workflow = [
    "Dataset acquisition and inspection",
    "Image preprocessing and normalization",
    "Training and validation data preparation",
    "Custom CNN model development",
    "Transfer learning using four pretrained architectures",
    "Model evaluation on the held-out test set",
    "Grad-CAM explainability implementation",
    "Streamlit application development"
]

for index, step in enumerate(workflow, start=1):

    st.write(f"**{index}.** {step}")


st.divider()

st.header("Technology Stack")

tech_col1, tech_col2, tech_col3 = st.columns(3)

with tech_col1:

    st.subheader("Programming")

    st.write("""
    - Python
    - NumPy
    - Pandas
    - OpenCV
    """)

with tech_col2:

    st.subheader("Deep Learning")

    st.write("""
    - PyTorch
    - Torchvision
    - CNN
    - Transfer Learning
    - Grad-CAM
    """)

with tech_col3:

    st.subheader("Application")

    st.write("""
    - Streamlit
    - Matplotlib
    - Scikit-learn
    - Git and GitHub
    """)


st.divider()

st.header("Project Highlights")

st.success(
    "Five trained architectures successfully integrated into "
    "a unified inference application."
)

st.success(
    "Grad-CAM implemented to provide visual explanations "
    "for model predictions."
)

st.success(
    "Interactive model performance dashboard with "
    "test-set metric comparisons."
)

st.info(
    "RetinaAI is an educational and experimental project. "
    "It is not a clinically validated diagnostic system and "
    "must not be used for medical decisions."
)

st.caption(
    "RetinaAI | Retinal Disease Detection Using Deep Learning"
)
