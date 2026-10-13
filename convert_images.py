
from pathlib import Path
from PIL import Image


image_folder = Path(__file__).parent / "Images"


jfif_files = [
    file for file in image_folder.iterdir()
    if file.suffix.lower() == ".jfif"
]

for file in jfif_files:
    new_file = file.with_suffix(".jpg")


    if new_file.exists():
        print(f"Пропускаем: {new_file.name} уже существует")
        continue

    with Image.open(file) as img:
        img.convert("RGB").save(
            new_file,
            format="JPEG",
            quality=95
        )

    print(f"Готово: {file.name} → {new_file.name}")

print(f"\nОбработка завершена! Найдено JFIF: {len(jfif_files)}")
