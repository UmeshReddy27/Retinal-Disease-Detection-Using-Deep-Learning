
import pandas as pd
from PIL import Image
from io import BytesIO
from pathlib import Path

DATA_DIR = Path("Data")
OUTPUT_DIR = Path("sample_images")

CLASS_NAMES = {
    0: "CNV",
    1: "DME",
    2: "DRUSEN",
    3: "NORMAL"
}

images_per_class = 3
extracted_counts = {name: 0 for name in CLASS_NAMES.values()}

parquet_files = sorted(DATA_DIR.glob("train-*.parquet"))

for parquet_file in parquet_files:
    if all(count >= images_per_class for count in extracted_counts.values()):
        break

    df = pd.read_parquet(parquet_file)

    for _, row in df.iterrows():
        label = int(row["label"])
        class_name = CLASS_NAMES[label]

        if extracted_counts[class_name] >= images_per_class:
            continue

        image_bytes = row["image"]["bytes"]
        image = Image.open(BytesIO(image_bytes)).convert("RGB")

        class_dir = OUTPUT_DIR / class_name
        class_dir.mkdir(parents=True, exist_ok=True)

        image_number = extracted_counts[class_name] + 1
        image_path = class_dir / f"{class_name}_{image_number}.png"

        image.save(image_path)

        extracted_counts[class_name] += 1
        print(f"Saved: {image_path}")

print("\nExtraction completed!")
print(extracted_counts)
