
from pathlib import Path

import torch
from PIL import Image
from transformers import CLIPModel, AutoProcessor



MODEL_NAME = "openai/clip-vit-base-patch32"


image_folder = Path(__file__).parent / "Images"


image_paths = sorted(
    path for path in image_folder.iterdir()
    if path.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}
)

if not image_paths:
    raise SystemExit("В папке images нет фотографий!")


images = []

for path in image_paths:
    with Image.open(path) as img:
        images.append(img.convert("RGB"))

print("Загружаем модель CLIP...")

model = CLIPModel.from_pretrained(MODEL_NAME)
processor = AutoProcessor.from_pretrained(MODEL_NAME)
model.eval()


query = input("Введите запрос на английском: ")


inputs = processor(
    text=[query],
    images=images,
    return_tensors="pt",
    padding=True
)


with torch.inference_mode():
    outputs = model(**inputs)

scores = outputs.logits_per_text[0].tolist()


results = sorted(
    zip(image_paths, scores),
    key=lambda item: item[1],
    reverse=True
)


print("\nРезультаты поиска:")

for path, score in results[:5]:
    print(f"{path.name}: {score:.2f}")
