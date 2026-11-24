
import json
from pathlib import Path

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


st.set_page_config(
    page_title="Model Performance | RetinaAI",
    page_icon="📊",
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
        font-size: 36px;
        font-weight: 750;
        margin-bottom: 5px;
    }

    .page-subtitle {
        color: #9299a5;
        font-size: 16px;
        margin-bottom: 25px;
    }

    .highlight-card {
        padding: 20px;
        border-radius: 14px;
        background: rgba(128,128,128,0.10);
        border: 1px solid rgba(128,128,128,0.18);
        min-height: 125px;
    }

    .highlight-label {
        color: #9299a5;
        font-size: 14px;
    }

    .highlight-value {
        font-size: 27px;
        font-weight: 750;
        margin-top: 8px;
    }
</style>
""", unsafe_allow_html=True)


st.markdown(
    '<p class="page-title">Model Performance Dashboard</p>',
    unsafe_allow_html=True
)

st.markdown(
    '<p class="page-subtitle">'
    'Evaluation and comparison of five deep learning architectures '
    'for retinal OCT classification.'
    '</p>',
    unsafe_allow_html=True
)


RESULT_DIR = Path("evaluation_results")

MODEL_FILES = {
    "Custom CNN": "custom_cnn.json",
    "ResNet18": "resnet18.json",
    "VGG16": "vgg16.json",
    "MobileNetV2": "mobilenetv2.json",
    "DenseNet121": "densenet121.json"
}


@st.cache_data
def load_evaluation_results():

    results = {}

    for model_name, filename in MODEL_FILES.items():

        file_path = RESULT_DIR / filename

        if not file_path.exists():
            continue

        with open(file_path, "r") as file:
            results[model_name] = json.load(file)

    return results


results = load_evaluation_results()

if not results:
    st.error(
        "No evaluation results found. "
        "Run evaluate_models.py first."
    )
    st.stop()


parameter_counts = {
    "Custom CNN": 110724,
    "ResNet18": 11178564,
    "VGG16": 138357544,
    "MobileNetV2": 2228996,
    "DenseNet121": 6957956
}


def plot_grouped_horizontal_bars(
    data,
    category_column,
    metric_columns,
    title,
    xlabel="Score (%)"
):

    categories = data[category_column].tolist()
    y = np.arange(len(categories))
    bar_height = 0.8 / len(metric_columns)

    fig, ax = plt.subplots(
        figsize=(11, max(4, len(categories) * 0.8))
    )

    for i, metric in enumerate(metric_columns):

        offset = (
            i - (len(metric_columns) - 1) / 2
        ) * bar_height

        values = data[metric].to_numpy()

        bars = ax.barh(
            y + offset,
            values,
            height=bar_height * 0.85,
            label=metric
        )

        for bar, value in zip(bars, values):

            ax.text(
                value + 0.5,
                bar.get_y() + bar.get_height() / 2,
                f"{value:.1f}%",
                va="center",
                fontsize=8
            )

    ax.set_yticks(y)
    ax.set_yticklabels(categories)
    ax.set_xlabel(xlabel)
    ax.set_title(title, fontsize=14, pad=15)
    ax.legend(loc="lower right")
    ax.grid(axis="x", alpha=0.25)
    ax.set_axisbelow(True)
    ax.set_xlim(0, 115)
    ax.invert_yaxis()

    fig.tight_layout()

    return fig


st.divider()

st.subheader("Overall Performance Highlights")


summary = []

for model_name, result in results.items():

    report = result["classification_report"]

    summary.append({
        "Model": model_name,
        "Accuracy": result["accuracy"] * 100,
        "Precision": report["macro avg"]["precision"] * 100,
        "Recall": report["macro avg"]["recall"] * 100,
        "F1-Score": report["macro avg"]["f1-score"] * 100,
        "Parameters": parameter_counts[model_name]
    })


df = pd.DataFrame(summary)

best_accuracy = df.loc[df["Accuracy"].idxmax()]
best_f1 = df.loc[df["F1-Score"].idxmax()]
smallest_model = df.loc[df["Parameters"].idxmin()]


col1, col2, col3 = st.columns(3)

with col1:

    st.markdown(
        f"""
        <div class="highlight-card">
            <div class="highlight-label">Highest Test Accuracy</div>
            <div class="highlight-value">{best_accuracy["Accuracy"]:.2f}%</div>
            <div>{best_accuracy["Model"]}</div>
        </div>
        """,
        unsafe_allow_html=True
    )

with col2:

    st.markdown(
        f"""
        <div class="highlight-card">
            <div class="highlight-label">Highest Macro F1-Score</div>
            <div class="highlight-value">{best_f1["F1-Score"]:.2f}%</div>
            <div>{best_f1["Model"]}</div>
        </div>
        """,
        unsafe_allow_html=True
    )

with col3:

    st.markdown(
        f"""
        <div class="highlight-card">
            <div class="highlight-label">Smallest Architecture</div>
            <div class="highlight-value">{smallest_model["Parameters"]:,}</div>
            <div>{smallest_model["Model"]}</div>
        </div>
        """,
        unsafe_allow_html=True
    )


st.divider()

st.subheader("Test Performance Comparison")

st.caption(
    "All metrics are calculated using the same held-out test dataset."
)

display_df = df[
    ["Model", "Accuracy", "Precision", "Recall", "F1-Score"]
].copy()

for column in ["Accuracy", "Precision", "Recall", "F1-Score"]:
    display_df[column] = display_df[column].map(
        lambda value: f"{value:.2f}%"
    )

st.dataframe(
    display_df,
    use_container_width=True,
    hide_index=True
)


st.divider()

st.subheader("Accuracy Comparison")

st.bar_chart(
    df.set_index("Model")[["Accuracy"]],
    horizontal=True
)


st.subheader("Precision, Recall and F1-Score")

st.bar_chart(
    df.set_index("Model")[
        ["Precision", "Recall", "F1-Score"]
    ],
    horizontal=True
)


st.divider()

st.subheader("Model Complexity")

complexity_df = df.copy()

complexity_df["Parameters (Millions)"] = (
    complexity_df["Parameters"] / 1_000_000
)

st.bar_chart(
    complexity_df.set_index("Model")[
        ["Parameters (Millions)"]
    ],
    horizontal=True
)


st.divider()

st.header("Confusion Matrix Analysis")

selected_matrix_model = st.selectbox(
    "Select a model",
    list(results.keys()),
    key="matrix_model"
)

selected_result = results[selected_matrix_model]

matrix = np.array(
    selected_result["confusion_matrix"]
)

class_names = selected_result["class_names"]

fig, ax = plt.subplots(figsize=(8, 6))

image_plot = ax.imshow(
    matrix,
    interpolation="nearest",
    cmap="Blues"
)

fig.colorbar(image_plot, ax=ax)

ax.set(
    xticks=np.arange(len(class_names)),
    yticks=np.arange(len(class_names)),
    xticklabels=class_names,
    yticklabels=class_names,
    xlabel="Predicted Class",
    ylabel="Actual Class",
    title=f"{selected_matrix_model} Confusion Matrix"
)

plt.setp(
    ax.get_xticklabels(),
    rotation=35,
    ha="right",
    rotation_mode="anchor"
)

threshold = matrix.max() / 2

for i in range(matrix.shape[0]):
    for j in range(matrix.shape[1]):

        ax.text(
            j,
            i,
            format(matrix[i, j], "d"),
            ha="center",
            va="center",
            color="white" if matrix[i, j] > threshold else "black"
        )

fig.tight_layout()

st.pyplot(fig)
plt.close(fig)


st.caption(
    "Rows represent actual classes. Columns represent predicted classes. "
    "Diagonal values are correct predictions."
)


st.divider()

st.header("Class-wise Performance Analysis")

selected_class_model = st.selectbox(
    "Select model for class-wise metrics",
    list(results.keys()),
    key="class_model"
)

class_report = results[
    selected_class_model
]["classification_report"]

class_names = results[
    selected_class_model
]["class_names"]

class_metrics = []

for class_name in class_names:

    class_data = class_report[class_name]

    class_metrics.append({
        "Class": class_name,
        "Precision": class_data["precision"] * 100,
        "Recall": class_data["recall"] * 100,
        "F1-Score": class_data["f1-score"] * 100,
        "Support": int(class_data["support"])
    })


class_df = pd.DataFrame(class_metrics)

st.dataframe(
    class_df.style.format({
        "Precision": "{:.2f}%",
        "Recall": "{:.2f}%",
        "F1-Score": "{:.2f}%",
        "Support": "{:.0f}"
    }),
    use_container_width=True,
    hide_index=True
)


st.subheader("Class-wise Metric Visualization")

class_fig = plot_grouped_horizontal_bars(
    class_df,
    "Class",
    ["Precision", "Recall", "F1-Score"],
    f"{selected_class_model} - Class-wise Metrics"
)

st.pyplot(class_fig)
plt.close(class_fig)


st.divider()

st.header("Cross-Model Class-wise Comparison")

comparison_metric = st.selectbox(
    "Choose evaluation metric",
    ["precision", "recall", "f1-score"]
)

cross_model_data = []

for model_name, result in results.items():

    report = result["classification_report"]

    row = {"Model": model_name}

    for class_name in result["class_names"]:

        row[class_name] = (
            report[class_name][comparison_metric] * 100
        )

    cross_model_data.append(row)


cross_model_df = pd.DataFrame(cross_model_data)

cross_fig = plot_grouped_horizontal_bars(
    cross_model_df,
    "Model",
    class_names,
    f"Cross-Model Class-wise {comparison_metric.title()} Comparison"
)

st.pyplot(cross_fig)
plt.close(cross_fig)


st.divider()

st.header("Key Observations")

st.markdown("""
- VGG16 achieved the highest test accuracy among the five models.
- ResNet18 delivered comparable performance with substantially fewer parameters.
- Custom CNN has the smallest architecture.
- DRUSEN has comparatively lower recall across all five architectures.
- Class-wise metrics and confusion matrices reveal errors that overall accuracy alone cannot show.
""")

st.info(
    "All evaluation results are generated from the same 1,000-image "
    "held-out test dataset. Model agreement and evaluation metrics "
    "do not establish clinical validity."
)

st.caption(
    "RetinaAI | Retinal Disease Detection Using Deep Learning"
)
