
import json
from pathlib import Path
from io import BytesIO

import pandas as pd
import torch

from PIL import Image
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

from src.prediction import (
    load_model,
    MODEL_CONFIGS,
    DEVICE
)

from src.preprocessing import (
    preprocess_image,
    CLASS_NAMES
)


TEST_FILE = Path("Data/test-00000-of-00001.parquet")
OUTPUT_DIR = Path("evaluation_results")

BATCH_SIZE = 32


class OCTTestDataset(Dataset):

    def __init__(self, dataframe, model_name):
        self.dataframe = dataframe.reset_index(drop=True)
        self.model_name = model_name

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, index):

        row = self.dataframe.iloc[index]

        image_bytes = row["image"]["bytes"]

        image = Image.open(
            BytesIO(image_bytes)
        ).convert("L")

        image_tensor = preprocess_image(
            image,
            self.model_name
        ).squeeze(0)

        label = int(row["label"])

        return image_tensor, label


def evaluate_model(model_name, dataframe):

    print(f"\nEvaluating {model_name}...")

    model = load_model(model_name)

    dataset = OCTTestDataset(
        dataframe,
        model_name
    )

    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0
    )

    all_predictions = []
    all_labels = []

    with torch.inference_mode():

        for images, labels in loader:

            images = images.to(DEVICE)

            outputs = model(images)

            predictions = outputs.argmax(dim=1)

            all_predictions.extend(
                predictions.cpu().tolist()
            )

            all_labels.extend(
                labels.tolist()
            )

    accuracy = accuracy_score(
        all_labels,
        all_predictions
    )

    report = classification_report(
        all_labels,
        all_predictions,
        labels=list(range(len(CLASS_NAMES))),
        target_names=CLASS_NAMES,
        output_dict=True,
        zero_division=0
    )

    matrix = confusion_matrix(
        all_labels,
        all_predictions,
        labels=list(range(len(CLASS_NAMES)))
    )

    results = {
        "accuracy": accuracy,
        "classification_report": report,
        "confusion_matrix": matrix.tolist(),
        "class_names": CLASS_NAMES,
        "test_samples": len(all_labels)
    }

    print(f"Accuracy: {accuracy * 100:.2f}%")

    print(
        classification_report(
            all_labels,
            all_predictions,
            labels=list(range(len(CLASS_NAMES))),
            target_names=CLASS_NAMES,
            zero_division=0
        )
    )

    del model

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return results


def main():

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    print("Loading test dataset...")

    dataframe = pd.read_parquet(TEST_FILE)

    print("Test samples:", len(dataframe))

    all_results = {}

    for model_name in MODEL_CONFIGS:

        try:

            results = evaluate_model(
                model_name,
                dataframe
            )

            all_results[model_name] = results

            output_file = (
                OUTPUT_DIR /
                f"{model_name.lower().replace(' ', '_')}.json"
            )

            with open(output_file, "w") as file:

                json.dump(
                    results,
                    file,
                    indent=4
                )

            print(f"Saved results: {output_file}")

        except Exception as error:

            print(f"\nFAILED: {model_name}")
            print("Error:", error)

    print("\nEvaluation completed!")

    print("\nSuccessful models:", len(all_results), "/", len(MODEL_CONFIGS))


if __name__ == "__main__":
    main()
