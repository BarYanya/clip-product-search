
from pathlib import Path
import csv
import re

import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from PIL import Image
from transformers import CLIPModel, AutoProcessor

BASE_DIR = Path(__file__).parent
IMAGE_DIR = BASE_DIR / "Images"
CATALOG_FILE = BASE_DIR / "catalog.csv"
EMBEDDINGS_FILE = BASE_DIR / "image_embeddings.pt"


data = torch.load(
    EMBEDDINGS_FILE,
    map_location="cpu",
    weights_only=True
)

filenames = data["filenames"]
image_embeddings = F.normalize(
    data["embeddings"].float(), dim=-1
)


with open(CATALOG_FILE, "r", encoding="utf-8-sig", newline="") as file:
    catalog = {
        product["filename"]: product
        for product in csv.DictReader(file)
    }

valid_indices = [
    i for i, filename in enumerate(filenames)
    if filename in catalog and (IMAGE_DIR / filename).is_file()
]

filenames = [filenames[i] for i in valid_indices]
image_embeddings = image_embeddings[valid_indices]

if not filenames:
    raise SystemExit("Нет товаров для сравнения!")

print(f"Товаров для сравнения: {len(filenames)}")


query = input("\nЧто будем искать? ").strip()

if not query:
    raise SystemExit("Пустой запрос!")


# ========== ПОИСК CLIP ==========

print("\nЗапускаем CLIP...")

device = "cuda" if torch.cuda.is_available() else "cpu"

model = CLIPModel.from_pretrained(data["model_name"])
processor = AutoProcessor.from_pretrained(data["model_name"])

model = model.to(device)
model.eval()

inputs = processor(
    text=[query],
    return_tensors="pt",
    padding=True
).to(device)

with torch.inference_mode():
    text_features = model.get_text_features(**inputs)

if hasattr(text_features, "pooler_output"):
    text_features = text_features.pooler_output

text_features = F.normalize(text_features, dim=-1)

# Сравниваем эмбеддинг текста с изображениями
clip_scores = (
    image_embeddings @ text_features[0].cpu()
).tolist()


# ========== ОБЫЧНЫЙ ПОИСК ==========

print("Запускаем поиск по словам...")

stop_words = {
    "a", "an", "the", "of", "on",
    "in", "with", "for", "and", "pair"
}

def get_words(text):
    words = re.findall(r"[a-z]+", text.lower())
    return set(words) - stop_words


query_words = get_words(query)
keyword_scores = []

for filename in filenames:
    product = catalog[filename]

    text = (
        Path(filename).stem + " "
        + product["title"] + " "
        + product["description"]
    )

    matches = query_words & get_words(text)
    keyword_scores.append(len(matches))


# ========== СОРТИРОВКА ==========

top_k = min(5, len(filenames))

clip_top = sorted(
    range(len(filenames)),
    key=lambda i: -clip_scores[i]
)[:top_k]

keyword_top = sorted(
    range(len(filenames)),
    key=lambda i: (-keyword_scores[i], filenames[i])
)

keyword_top = [
    i for i in keyword_top
    if keyword_scores[i] > 0
][:top_k]


# ========== ВЫВОД В ТЕРМИНАЛ ==========

print("\nCLIP:")

for rank, i in enumerate(clip_top, start=1):
    print(
        f"{rank}. {filenames[i]} "
        f"— сходство: {clip_scores[i]:.3f}"
    )

print("\nОбычный поиск:")

for rank, i in enumerate(keyword_top, start=1):
    print(
        f"{rank}. {filenames[i]} "
        f"— совпадений: {keyword_scores[i]}"
    )

if not keyword_top:
    print("Совпадений не найдено")


# ========== ПОКАЗЫВАЕМ ФОТОГРАФИИ ==========

fig, axes = plt.subplots(
    2, top_k,
    figsize=(top_k * 3, 8),
    squeeze=False
)

rankings = [clip_top, keyword_top]

for row, ranking in enumerate(rankings):

    for col in range(top_k):
        ax = axes[row][col]
        ax.axis("off")

        if col >= len(ranking):
            continue

        i = ranking[col]
        filename = filenames[i]

        with Image.open(IMAGE_DIR / filename) as img:
            ax.imshow(img.convert("RGB"))

        if row == 0:
            score = f"Similarity: {clip_scores[i]:.3f}"
        else:
            score = f"Matches: {keyword_scores[i]}"

        ax.set_title(
            f"{col + 1}. {filename}\n{score}",
            fontsize=9
        )

fig.suptitle(
    f"Search: {query}\n"
    "Top: CLIP | Bottom: Keyword Search",
    fontsize=14
)

plt.tight_layout()


output_file = BASE_DIR / "comparison.png"
plt.savefig(output_file, dpi=150, bbox_inches="tight")

print(f"\nСравнение сохранено: {output_file.name}")

plt.show()
