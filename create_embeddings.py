
import csv
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image
from transformers import CLIPModel, AutoProcessor


BASE_DIR = Path(__file__).parent

IMAGE_DIR = BASE_DIR / "Images"
CATALOG_FILE = BASE_DIR / "catalog.csv"
OUTPUT_FILE = BASE_DIR / "image_embeddings.pt"

MODEL_NAME = "openai/clip-vit-base-patch32"


with open(CATALOG_FILE, "r", encoding="utf-8-sig", newline="") as file:
    reader = csv.DictReader(file)

    if not reader.fieldnames or "filename" not in reader.fieldnames:
        raise SystemExit("В catalog.csv отсутствует столбец filename")

    products = list(reader)

print(f"Товаров в каталоге: {len(products)}")


device = "cuda" if torch.cuda.is_available() else "cpu"

print(f"Используем устройство: {device}")


print("Загружаем модель CLIP...")

model = CLIPModel.from_pretrained(MODEL_NAME)
processor = AutoProcessor.from_pretrained(MODEL_NAME)

model = model.to(device)
model.eval()


embeddings = []
filenames = []


for i, product in enumerate(products, start=1):

    filename = product["filename"].strip()
    image_path = IMAGE_DIR / filename

    if not image_path.is_file():
        print(f"Файл не найден: {filename}")
        continue

    print(f"Обрабатываем {i}/{len(products)}: {filename}")

    try:

        with Image.open(image_path) as img:
            image = img.convert("RGB")


        inputs = processor(
            images=image,
            return_tensors="pt"
        )

        pixel_values = inputs["pixel_values"].to(device)


        with torch.inference_mode():
            features = model.get_image_features(
                pixel_values=pixel_values
            )

        if hasattr(features, "pooler_output"):
            features = features.pooler_output


        features = F.normalize(features, dim=-1)

        embeddings.append(features[0].cpu())
        filenames.append(filename)

    except Exception as error:
        print(f"Ошибка обработки {filename}: {error}")


if not embeddings:
    raise SystemExit("Не удалось обработать ни одной фотографии!")


embedding_matrix = torch.stack(embeddings)



torch.save(
    {
        "model_name": MODEL_NAME,
        "filenames": filenames,
        "embeddings": embedding_matrix
    },
    OUTPUT_FILE
)

print("\nГотово!")
print(f"Обработано изображений: {len(filenames)}")
print(f"Размер матрицы: {embedding_matrix.shape}")
print(f"Результат сохранён: {OUTPUT_FILE.name}")
