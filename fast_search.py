
from pathlib import Path
import csv

import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from PIL import Image
from transformers import CLIPModel, AutoProcessor



BASE_DIR = Path(__file__).parent
IMAGE_DIR = BASE_DIR / "Images"
EMBEDDINGS_FILE = BASE_DIR / "image_embeddings.pt"
CATALOG_FILE = BASE_DIR / "catalog.csv"


if not EMBEDDINGS_FILE.exists():
    raise SystemExit(
        "Сначала запусти create_embeddings.py!"
    )

data = torch.load(
    EMBEDDINGS_FILE,
    map_location="cpu",
    weights_only=True
)

image_embeddings = data["embeddings"]
filenames = data["filenames"]

print(f"Загружено товаров: {len(filenames)}")



with open(CATALOG_FILE, "r", encoding="utf-8-sig", newline="") as file:
    products = list(csv.DictReader(file))

titles = {
    product["filename"]: product["title"]
    for product in products
}



device = "cuda" if torch.cuda.is_available() else "cpu"

model_name = data["model_name"]

print("Загружаем CLIP...")

model = CLIPModel.from_pretrained(model_name)
processor = AutoProcessor.from_pretrained(model_name)

model = model.to(device)
model.eval()



query = input("\nЧто будем искать? ").strip()

if not query:
    raise SystemExit("Запрос не может быть пустым!")



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


text_vector = text_features[0].cpu()

scores = image_embeddings @ text_vector



top_k = min(5, len(filenames))

top_scores, top_indices = torch.topk(
    scores,
    k=top_k
)



print("\nРезультаты поиска:")

for rank, (index, score) in enumerate(
    zip(top_indices.tolist(), top_scores.tolist()),
    start=1
):
    filename = filenames[index]
    title = titles.get(filename, filename)

    print(f"{rank}. {title} — {score:.3f}")



fig, axes = plt.subplots(
    1,
    top_k,
    figsize=(top_k * 3, 4),
    squeeze=False
)

for i, (index, score) in enumerate(
    zip(top_indices.tolist(), top_scores.tolist())
):
    filename = filenames[index]
    image_path = IMAGE_DIR / filename
    ax = axes[0][i]

    ax.axis("off")

    if image_path.exists():
        with Image.open(image_path) as img:
            ax.imshow(img.convert("RGB"))

    title = titles.get(filename, filename)

    ax.set_title(
        f"{i + 1}. {title}\nScore: {score:.3f}",
        fontsize=9
    )

fig.suptitle(f"Search: {query}")
plt.tight_layout()
plt.show()
