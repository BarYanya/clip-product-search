
import csv
from pathlib import Path

import torch
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration


BASE_DIR = Path(__file__).parent
IMAGE_DIR = BASE_DIR / "Images"

INPUT_FILE = BASE_DIR / "catalog_updated.csv"
OUTPUT_FILE = BASE_DIR / "catalog_filled.csv"


with open(INPUT_FILE, "r", encoding="utf-8-sig", newline="") as file:
    products = list(csv.DictReader(file))


new_products = [
    product for product in products
    if not product["title"].strip()
    or not product["description"].strip()
]

print(f"Всего товаров: {len(products)}")
print(f"Нужно обработать: {len(new_products)}")


def save_catalog():
    with open(
        OUTPUT_FILE,
        "w",
        encoding="utf-8-sig",
        newline=""
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=["filename", "title", "description"],
            extrasaction="ignore"
        )

        writer.writeheader()
        writer.writerows(products)

if new_products:

    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Используем устройство: {device}")
    print("Загружаем BLIP...")

    model_name = "Salesforce/blip-image-captioning-base"

    processor = BlipProcessor.from_pretrained(model_name)
    model = BlipForConditionalGeneration.from_pretrained(
        model_name
    )

    model = model.to(device)
    model.eval()

    for i, product in enumerate(new_products, start=1):

        filename = product["filename"]
        image_path = IMAGE_DIR / filename

        print(f"\n{i}/{len(new_products)}: {filename}")

        try:
            with Image.open(image_path) as img:
                image = img.convert("RGB")

            inputs = processor(
                images=image,
                return_tensors="pt"
            ).to(device)

            with torch.inference_mode():
                
            output = model.generate(
                **inputs,
                max_new_tokens=30,
                num_beams=3,
                no_repeat_ngram_size=3,
                repetition_penalty=1.3
            )


            description = processor.decode(
                output[0],
                skip_special_tokens=True
            ).strip()

            if not description:
                raise ValueError("Модель вернула пустое описание")

            if not product["description"].strip():
                product["description"] = description

            if not product["title"].strip():
                product["title"] = description.capitalize()

            print(f"Описание: {description}")

            save_catalog()

        except Exception as error:
            print(f"Ошибка: {error}")


save_catalog()

remaining = sum(
    not p["title"].strip() or not p["description"].strip()
    for p in products
)

print("\nГотово!")
print(f"Каталог сохранён: {OUTPUT_FILE.name}")
print(f"Товаров без полного описания: {remaining}")
