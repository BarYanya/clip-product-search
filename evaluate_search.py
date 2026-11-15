
from pathlib import Path
import csv
import re

import torch
import torch.nn.functional as F
from transformers import CLIPModel, AutoProcessor


BASE_DIR = Path(__file__).parent

CATALOG_FILE = BASE_DIR / "catalog.csv"
TEST_FILE = BASE_DIR / "test_queries.csv"
EMBEDDINGS_FILE = BASE_DIR / "image_embeddings.pt"
RESULT_FILE = BASE_DIR / "evaluation_results.csv"


with open(CATALOG_FILE, encoding="utf-8-sig", newline="") as file:
    catalog = {
        row["filename"]: row
        for row in csv.DictReader(file)
    }


with open(TEST_FILE, encoding="utf-8-sig", newline="") as file:
    tests = list(csv.DictReader(file))

if not tests:
    raise SystemExit("Нет тестовых запросов!")


data = torch.load(
    EMBEDDINGS_FILE,
    map_location="cpu",
    weights_only=True
)

filenames = data["filenames"]

image_embeddings = F.normalize(
    data["embeddings"].float(),
    dim=-1
)

missing = [
    test["expected_filename"]
    for test in tests
    if test["expected_filename"] not in filenames
]

if missing:
    raise SystemExit(
        f"Нет фотографий в индексе: {missing}"
    )



print("Загружаем CLIP...")

device = "cuda" if torch.cuda.is_available() else "cpu"

model = CLIPModel.from_pretrained(data["model_name"])
processor = AutoProcessor.from_pretrained(data["model_name"])

model = model.to(device)
model.eval()


queries = [test["query"] for test in tests]

inputs = processor(
    text=queries,
    return_tensors="pt",
    padding=True,
    truncation=True
).to(device)

with torch.inference_mode():
    text_features = model.get_text_features(**inputs)

if hasattr(text_features, "pooler_output"):
    text_features = text_features.pooler_output

text_embeddings = F.normalize(
    text_features, dim=-1
).cpu()


score_matrix = text_embeddings @ image_embeddings.T


stop_words = {
    "a", "an", "the", "of", "on",
    "in", "with", "for", "and", "pair"
}

def get_words(text):
    words = re.findall(r"[a-z]+", text.lower())
    return set(words) - stop_words


def keyword_search(query):
    query_words = get_words(query)
    scores = []

    for filename in filenames:
        product = catalog[filename]

        
        
        text = (
            product["title"] + " "
            + product["description"]
        )


        matches = query_words & get_words(text)
        scores.append(len(matches))

    ranking = sorted(
        range(len(filenames)),
        key=lambda i: (-scores[i], filenames[i])
    )

    return [
        filenames[i]
        for i in ranking
        if scores[i] > 0
    ]


def hit_at(ranking, expected, k):
    return int(expected in ranking[:k])


results = []

print("\nНАЧИНАЕМ ОЦЕНКУ\n")

for i, test in enumerate(tests):

    query = test["query"]
    expected = test["expected_filename"]

    clip_scores = score_matrix[i].tolist()

    clip_indices = sorted(
        range(len(filenames)),
        key=lambda j: -clip_scores[j]
    )

    clip_ranking = [
        filenames[j] for j in clip_indices
    ]

    keyword_ranking = keyword_search(query)

    result = {
        "query": query,
        "expected": expected,
        "clip_hit1": hit_at(clip_ranking, expected, 1),
        "clip_hit5": hit_at(clip_ranking, expected, 5),
        "keyword_hit1": hit_at(keyword_ranking, expected, 1),
        "keyword_hit5": hit_at(keyword_ranking, expected, 5)
    }

    results.append(result)

    print(f"{i + 1}. {query}")
    print(f"   Нужен: {expected}")
    print(f"   CLIP: {clip_ranking[0]}")
    print(
        f"   Keywords: "
        f"{keyword_ranking[0] if keyword_ranking else 'Ничего'}"
    )
    print()


print("ИТОГОВЫЕ РЕЗУЛЬТАТЫ\n")

for method in ["clip", "keyword"]:
    for k in [1, 5]:
        column = f"{method}_hit{k}"

        accuracy = (
            sum(row[column] for row in results)
            / len(results) * 100
        )

        print(f"{method.upper()} Hit@{k}: {accuracy:.1f}%")


with open(RESULT_FILE, "w", encoding="utf-8-sig", newline="") as file:
    writer = csv.DictWriter(
        file,
        fieldnames=results[0].keys()
    )

    writer.writeheader()
    writer.writerows(results)

print(f"\nРезультаты сохранены: {RESULT_FILE.name}")
