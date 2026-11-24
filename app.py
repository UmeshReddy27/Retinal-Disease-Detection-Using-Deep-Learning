
import streamlit as st
import pandas as pd
import torch
import numpy as np
import cv2
import hashlib

from PIL import Image
from io import BytesIO

from src.prediction import (
    load_model,
    predict_image,
    generate_gradcam,
    MODEL_CONFIGS
)

from src.preprocessing import (
    preprocess_image,
    CLASS_NAMES
)


st.set_page_config(
    page_title="RetinaAI | OCT Analysis",
    page_icon="👁️",
    layout="wide",
    initial_sidebar_state="expanded"
)


st.markdown("""
<style>
    .block-container {
        padding-top: 2.5rem;
        padding-bottom: 3rem;
        max-width: 1400px;
    }

    .main-title {
        font-size: 38px;
        font-weight: 750;
        line-height: 1.4;
        margin-bottom: 0;
    }

    .subtitle {
        color: #9299a5;
        font-size: 16px;
        margin-top: 5px;
        margin-bottom: 25px;
    }

    .result-card {
        padding: 22px;
        border-radius: 14px;
        background: rgba(128,128,128,0.10);
        border: 1px solid rgba(128,128,128,0.18);
    }

    .result-label {
        font-size: 14px;
        color: #9299a5;
    }

    .result-value {
        font-size: 32px;
        font-weight: 750;
        margin-top: 5px;
    }

    .section-title {
        font-size: 23px;
        font-weight: 700;
        margin-bottom: 15px;
    }

    div.stButton > button {
        width: 100%;
        border-radius: 10px;
        font-weight: 600;
        padding: 10px;
    }

    [data-testid="stFileUploader"] {
        border-radius: 12px;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource(show_spinner=False)
def get_cached_model(model_name):
    return load_model(model_name)


def create_gradcam_overlay(image, heatmap):

    original = image.convert("RGB").resize((224, 224))

    original_array = np.array(original)

    heatmap_array = np.uint8(
        255 * np.clip(heatmap, 0, 1)
    )

    colored_heatmap = cv2.applyColorMap(
        heatmap_array,
        cv2.COLORMAP_JET
    )

    colored_heatmap = cv2.cvtColor(
        colored_heatmap,
        cv2.COLOR_BGR2RGB
    )

    overlay = cv2.addWeighted(
        original_array,
        0.55,
        colored_heatmap,
        0.45,
        0
    )

    return original_array, colored_heatmap, overlay


def run_model_comparison(image):

    comparison_results = []

    progress_bar = st.progress(0)
    status_text = st.empty()

    for index, model_name in enumerate(MODEL_CONFIGS):

        status_text.write(
            f"Running {model_name}..."
        )

        model = get_cached_model(model_name)

        image_tensor = preprocess_image(
            image,
            model_name
        )

        predicted_class, confidence, probabilities = predict_image(
            model,
            image_tensor
        )

        comparison_results.append({
            "Model": model_name,
            "Prediction": predicted_class,
            "Confidence": confidence * 100
        })

        progress_bar.progress(
            (index + 1) / len(MODEL_CONFIGS)
        )

    status_text.success(
        "All five models completed their predictions!"
    )

    progress_bar.empty()

    return comparison_results


with st.sidebar:

    st.title("👁️ RetinaAI")

    st.caption("RETINAL OCT ANALYSIS PLATFORM")

    st.divider()

    st.subheader("Model Configuration")

    selected_model = st.selectbox(
        "Choose Deep Learning Model",
        list(MODEL_CONFIGS.keys())
    )

    st.info(
        "Select one of the five trained models "
        "to analyze your OCT scan."
    )

    st.divider()

    st.subheader("Available Models")

    for model_name in MODEL_CONFIGS:
        st.write(f"✓ {model_name}")

    st.divider()

    st.caption("Powered by PyTorch and Streamlit")


st.markdown(
    '<p class="main-title">Retinal Disease Detection</p>',
    unsafe_allow_html=True
)

st.markdown(
    '<p class="subtitle">Deep Learning Based OCT Image Classification and Explainable AI</p>',
    unsafe_allow_html=True
)


metric1, metric2, metric3 = st.columns(3)

with metric1:
    st.metric("Available Models", "5")

with metric2:
    st.metric("Classification Classes", "4")

with metric3:
    device_name = "CUDA" if torch.cuda.is_available() else "CPU"
    st.metric("Inference Device", device_name)


st.divider()


st.markdown(
    '<p class="section-title">OCT Image Analysis</p>',
    unsafe_allow_html=True
)


uploaded_file = st.file_uploader(
    "Upload your retinal OCT scan",
    type=["jpg", "jpeg", "png"],
    help="Upload a JPG or PNG OCT image."
)


if uploaded_file is None:

    st.markdown("""
    <div class="result-card">
        <h3>Welcome to RetinaAI</h3>
        <p>Upload an OCT retinal scan to begin the analysis.</p>
        <p>Supported models: Custom CNN, ResNet18, VGG16,
        MobileNetV2 and DenseNet121.</p>
    </div>
    """, unsafe_allow_html=True)

    st.session_state.pop("prediction_result", None)
    st.session_state.pop("comparison_results", None)


else:

    image_bytes = uploaded_file.getvalue()

    image_hash = hashlib.md5(image_bytes).hexdigest()

    image = Image.open(
        BytesIO(image_bytes)
    ).convert("RGB")

    image_col, analysis_col = st.columns(
        [1.1, 1],
        gap="large"
    )

    with image_col:

        st.subheader("Uploaded OCT Scan")

        st.image(
            image,
            use_container_width=True
        )

        st.caption(
            f"Image dimensions: {image.width} × {image.height}"
        )

    with analysis_col:

        st.subheader("Analysis Configuration")

        st.write(f"**Selected model:** {selected_model}")

        predict_button = st.button(
            "🔍 Analyze OCT Image",
            type="primary",
            use_container_width=True
        )

        compare_button = st.button(
            "⚖️ Compare All 5 Models",
            use_container_width=True
        )

    if predict_button:

        st.session_state.pop("prediction_result", None)

        try:

            with st.spinner(
                f"Analyzing image using {selected_model}..."
            ):

                model = get_cached_model(selected_model)

                image_tensor = preprocess_image(
                    image,
                    selected_model
                )

                predicted_class, confidence, probabilities = predict_image(
                    model,
                    image_tensor
                )

                class_index = CLASS_NAMES.index(
                    predicted_class
                )

                heatmap = generate_gradcam(
                    model,
                    image_tensor,
                    selected_model,
                    class_index=class_index
                )

                original_array, colored_heatmap, overlay = (
                    create_gradcam_overlay(image, heatmap)
                )

                st.session_state["prediction_result"] = {
                    "class": predicted_class,
                    "confidence": confidence,
                    "probabilities": probabilities,
                    "model": selected_model,
                    "image_hash": image_hash,
                    "original": original_array,
                    "heatmap": colored_heatmap,
                    "overlay": overlay
                }

        except Exception as error:

            st.error(
                f"Analysis failed: {error}"
            )

    if compare_button:

        st.session_state.pop("comparison_results", None)

        try:

            comparison_results = run_model_comparison(image)

            st.session_state["comparison_results"] = {
                "image_hash": image_hash,
                "results": comparison_results
            }

        except Exception as error:

            st.error(
                f"Model comparison failed: {error}"
            )


    result = st.session_state.get("prediction_result")

    result_is_current = (
        result is not None
        and result["model"] == selected_model
        and result["image_hash"] == image_hash
    )


    if result_is_current:

        st.divider()

        st.markdown(
            '<p class="section-title">Prediction Results</p>',
            unsafe_allow_html=True
        )

        result_col1, result_col2, result_col3 = st.columns(3)

        with result_col1:

            st.markdown(
                f"""
                <div class="result-card">
                    <div class="result-label">Predicted Disease Class</div>
                    <div class="result-value">{result["class"]}</div>
                </div>
                """,
                unsafe_allow_html=True
            )

        with result_col2:

            st.metric(
                "Prediction Confidence",
                f'{result["confidence"] * 100:.2f}%'
            )

        with result_col3:

            st.metric(
                "Selected Model",
                result["model"]
            )


        st.divider()

        st.markdown(
            '<p class="section-title">Grad-CAM Explainability</p>',
            unsafe_allow_html=True
        )

        st.write(
            "Grad-CAM highlights image regions that contributed "
            "to the selected model's predicted class."
        )

        gradcam_col1, gradcam_col2 = st.columns(
            2,
            gap="large"
        )

        with gradcam_col1:

            st.subheader("Original OCT Scan")

            st.image(
                result["original"],
                use_container_width=True
            )

        with gradcam_col2:

            st.subheader("Grad-CAM Overlay")

            st.image(
                result["overlay"],
                use_container_width=True
            )

        with st.expander("View Grad-CAM Heatmap"):

            st.image(
                result["heatmap"],
                caption="Grad-CAM activation heatmap",
                use_container_width=True
            )


        st.divider()

        st.markdown(
            '<p class="section-title">Class Probability Analysis</p>',
            unsafe_allow_html=True
        )

        probability_data = pd.DataFrame({
            "Disease Class": list(result["probabilities"].keys()),
            "Probability": [
                value * 100
                for value in result["probabilities"].values()
            ]
        })

        probability_data = probability_data.sort_values(
            "Probability",
            ascending=False
        )

        chart_col, table_col = st.columns(
            [1.3, 1],
            gap="large"
        )

        with chart_col:

            st.subheader("Probability Distribution")

            st.bar_chart(
                probability_data.set_index("Disease Class"),
                horizontal=True
            )

        with table_col:

            st.subheader("Detailed Results")

            for _, row in probability_data.iterrows():

                st.write(
                    f'**{row["Disease Class"]}** — '
                    f'{row["Probability"]:.2f}%'
                )

                st.progress(
                    min(
                        max(
                            float(row["Probability"]) / 100,
                            0.0
                        ),
                        1.0
                    )
                )

        st.caption(
            "Probabilities are model outputs and should not be "
            "interpreted as calibrated clinical probabilities."
        )


    comparison = st.session_state.get("comparison_results")

    comparison_is_current = (
        comparison is not None
        and comparison["image_hash"] == image_hash
    )

    if comparison_is_current:

        st.divider()

        st.markdown(
            '<p class="section-title">Five-Model Comparison</p>',
            unsafe_allow_html=True
        )

        comparison_df = pd.DataFrame(
            comparison["results"]
        )

        st.subheader("Prediction Results")

        display_df = comparison_df.copy()

        display_df["Confidence"] = display_df[
            "Confidence"
        ].map(
            lambda value: f"{value:.2f}%"
        )

        st.dataframe(
            display_df,
            use_container_width=True,
            hide_index=True
        )

        st.subheader("Confidence Comparison")

        chart_df = comparison_df.set_index(
            "Model"
        )[["Confidence"]]

        st.bar_chart(
            chart_df,
            horizontal=True
        )

        st.subheader("Model Agreement")

        prediction_counts = comparison_df[
            "Prediction"
        ].value_counts()

        majority_class = prediction_counts.index[0]

        agreement = (
            prediction_counts.iloc[0]
            / len(comparison_df)
        ) * 100

        agreement_col1, agreement_col2 = st.columns(2)

        with agreement_col1:

            st.metric(
                "Most Common Prediction",
                majority_class
            )

        with agreement_col2:

            st.metric(
                "Model Agreement",
                f"{agreement:.0f}%"
            )

        if agreement == 100:

            st.success(
                "All five models predicted the same class."
            )

        else:

            st.warning(
                "The models disagree. Review their individual "
                "predictions rather than relying on a single result."
            )

        st.caption(
            "Model agreement measures prediction consistency, "
            "not diagnostic accuracy."
        )


st.divider()

st.caption(
    "Disclaimer: RetinaAI is an educational and experimental "
    "deep learning project. It is not a clinically validated "
    "diagnostic system and must not be used for medical decisions."
)
