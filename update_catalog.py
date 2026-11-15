
import csv
from pathlib import Path


BASE_DIR = Path(__file__).parent
IMAGE_DIR = BASE_DIR / "Images"

OLD_CATALOG = BASE_DIR / "catalog.csv"
NEW_CATALOG = BASE_DIR / "catalog_updated.csv"


with open(OLD_CATALOG, "r", encoding="utf-8-sig", newline="") as file:
    products = list(csv.DictReader(file))


existing_files = {
    product["filename"].casefold()
    for product in products
}


extensions = {".jpg", ".jpeg", ".png", ".jfif", ".webp"}

image_files = sorted(
    [
        file for file in IMAGE_DIR.iterdir()
        if file.is_file() and file.suffix.lower() in extensions
    ],
    key=lambda file: file.name.lower()
)


added = 0

for image in image_files:

    if image.name.casefold() not in existing_files:

        products.append({
            "filename": image.name,
            "title": "",
            "description": ""
        })

        existing_files.add(image.name.casefold())
        added += 1

with open(NEW_CATALOG, "w", encoding="utf-8-sig", newline="") as file:

    writer = csv.DictWriter(
        file,
        fieldnames=["filename", "title", "description"],
        extrasaction="ignore"
    )

    writer.writeheader()
    writer.writerows(products)

print("Готово!")
print(f"Найдено фотографий: {len(image_files)}")
print(f"Добавлено новых товаров: {added}")
print(f"Всего записей в каталоге: {len(products)}")
print("Создан файл catalog_updated.csv")
