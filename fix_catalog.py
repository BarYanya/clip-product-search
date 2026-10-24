
import csv
from pathlib import Path

# Находим каталог
folder = Path(__file__).parent

source = folder / "catalog.csv"

if (folder / "catalog_auto.csv").exists():
    source = folder / "catalog_auto.csv"

with open(source, "r", encoding="utf-8-sig", newline="") as file:
    reader = csv.DictReader(file)
    products = list(reader)

for product in products:
    filename = product["filename"]

    title = Path(filename).stem
    title = title.replace("_", " ")
    title = title.replace("-", " ")
    title = title.title()

    product["title"] = title

output = folder / "catalog_fixed.csv"

with open(output, "w", encoding="utf-8-sig", newline="") as file:
    writer = csv.DictWriter(
        file,
        fieldnames=["filename", "title", "description"]
    )
    writer.writeheader()
    writer.writerows(products)

print("Готово! Создан catalog_fixed.csv")
print(f"Обработано товаров: {len(products)}")

