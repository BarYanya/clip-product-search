
from pathlib import Path
import csv

import torch
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration



project_folder = Path(__file__).parent
image_folder = project_folder / "Images"


catalog_file = project_folder / "catalog.csv"

if catalog_file.exists():
    catalog_file = project_folder / "catalog_auto.csv"


image_files = sorted([
    file for file in image_folder.iterdir()
    if file.suffix.lower() in {
        ".jpg", ".jpeg", ".png", ".jfif", ".webp"
    }
])

if not image_files:
    raise SystemExit("В папке Images нет фотографий!")

print(f"Найдено фотографий: {len(image_files)}")


model_name = "Salesforce/blip-image-captioning-base"

print("Загружаем модель BLIP...")

processor = BlipProcessor.from_pretrained(model_name)
model = BlipForConditionalGeneration.from_pretrained(model_name)


device = "cuda" if torch.cuda.is_available() else "cpu"

model = model.to(device)
model.eval()

print(f"Устройство: {device}")


catalog = []


for i, file in enumerate(image_files, start=1):

    print(f"Обрабатываем {i}/{len(image_files)}: {file.name}")

    try:
    
        with Image.open(file) as img:
            image = img.convert("RGB")

        inputs = processor(
            images=image,
            return_tensors="pt"
        ).to(device)

        with torch.inference_mode():
            output = model.generate(
                **inputs,
                max_new_tokens=40
            )

        description = processor.decode(
            output[0],
            skip_special_tokens=True
        ).strip()

        title = " ".join(description.split()[:6]).capitalize()

        catalog.append({
            "filename": file.name,
            "title": title,
            "description": description
        })

        print(f"Описание: {description}\n")

    except Exception as error:
        print(f"Ошибка с файлом {file.name}: {error}")


with open(catalog_file, "w", newline="", encoding="utf-8-sig") as file:

    writer = csv.DictWriter(
        file,
        fieldnames=["filename", "title", "description"]
    )

    writer.writeheader()
    writer.writerows(catalog)

print(f"\nГотово! Каталог сохранен: {catalog_file.name}")
print(f"Добавлено товаров: {len(catalog)}")
