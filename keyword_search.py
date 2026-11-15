
import csv
import re
from pathlib import Path


BASE_DIR = Path(__file__).parent
CATALOG_FILE = BASE_DIR / "catalog.csv"

with open(CATALOG_FILE, "r", encoding="utf-8-sig", newline="") as file:
    products = list(csv.DictReader(file))

print(f"Загружено товаров: {len(products)}")


stop_words = {
    "a", "an", "the", "of", "on",
    "in", "with", "for", "and", "pair"
}

def get_words(text):
    words = re.findall(r"[a-z]+", text.lower())
    return set(words) - stop_words


query = input("Что будем искать? ")

query_words = get_words(query)

if not query_words:
    raise SystemExit("Введите поисковый запрос!")


results = []

for product in products:

    filename = Path(product["filename"]).stem

    text = (
        filename + " "
        + product["title"] + " "
        + product["description"]
    )


    product_words = get_words(text)

    matches = query_words.intersection(product_words)

    score = len(matches)

    if score > 0:
        results.append((product, score))



results.sort(
    key=lambda item: item[1],
    reverse=True
)


print("\nРезультаты обычного поиска:")

for i, (product, score) in enumerate(results[:5], start=1):
    print(
        f"{i}. {product['title']} "
        f"— совпадений: {score}"
    )

if not results:
    print("Совпадений не найдено")
