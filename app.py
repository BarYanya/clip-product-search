
import csv
import re
from pathlib import Path

import streamlit as st
import torch
import torch.nn.functional as F
from transformers import CLIPModel, AutoProcessor


# Настройки
BASE_DIR = Path(__file__).parent
IMAGE_DIR = BASE_DIR / "Images"

st.set_page_config(
    page_title="CLIP Product Search",
    page_icon="🔎",
    layout="wide"
)

st.title("🔎 CLIP Product Search")
st.write("Поиск товаров по текстовому описанию")
st.caption("Сравнение CLIP ViT-B/32 и поиска по ключевым словам")


# Загружаем данные
@st.cache_resource
def load_data():
    data = torch.load(
        BASE_DIR / "image_embeddings.pt",
        map_location="cpu",
        weights_only=True
    )

    with open(
        BASE_DIR / "catalog.csv",
        encoding="utf-8-sig",
        newline=""
    ) as file:
        catalog = {
            row["filename"]: row
            for row in csv.DictReader(file)
        }

    valid_indices = [
        i for i, name in enumerate(data["filenames"])
        if name in catalog and (IMAGE_DIR / name).is_file()
    ]

    filenames = [
        data["filenames"][i] for i in valid_indices
    ]

    embeddings = F.normalize(
        data["embeddings"][valid_indices].float(),
        dim=-1
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = CLIPModel.from_pretrained(data["model_name"])
    processor = AutoProcessor.from_pretrained(data["model_name"])

    model = model.to(device)
    model.eval()

    return filenames, embeddings, catalog, model, processor, device


with st.spinner("Загружаем модель CLIP..."):
    filenames, embeddings, catalog, model, processor, device = load_data()

if not filenames:
    st.error("Не найдены товары для поиска!")
    st.stop()

st.success(f"Каталог загружен: {len(filenames)} товаров")


# Поле ввода
query = st.text_input(
    "Что будем искать?",
    value="a pair of black sneakers"
).strip()


# Поиск CLIP
def clip_search(query):
    inputs = processor(
        text=[query],
        return_tensors="pt",
        padding=True
    ).to(device)

    with torch.inference_mode():
        features = model.get_text_features(**inputs)

    if hasattr(features, "pooler_output"):
        features = features.pooler_output

    vector = F.normalize(features, dim=-1)[0].cpu()

    scores = embeddings @ vector

    top_k = min(5, len(filenames))
    values, indices = torch.topk(scores, k=top_k)

    return list(zip(indices.tolist(), values.tolist()))


# Поиск по словам
def keyword_search(query):
    stop_words = {
        "a", "an", "the", "of", "on",
        "in", "with", "for", "and", "pair"
    }

    def words(text):
        return set(re.findall(r"[a-z]+", text.lower())) - stop_words

    query_words = words(query)
    results = []

    for i, filename in enumerate(filenames):
        product = catalog[filename]

        text = (
            Path(filename).stem + " "
            + product["title"] + " "
            + product["description"]
        )

        score = len(query_words & words(text))

        if score > 0:
            results.append((i, score))

    results.sort(key=lambda x: (-x[1], filenames[x[0]]))

    return results[:5]


# Показываем результаты
if query:
    clip_results = clip_search(query)
    keyword_results = keyword_search(query)

    left, right = st.columns(2)

    with left:
        st.subheader("🧠 Поиск CLIP")

        for rank, (index, score) in enumerate(clip_results, 1):
            filename = filenames[index]
            product = catalog[filename]

            st.image(str(IMAGE_DIR / filename), width=220)
            st.write(f"**{rank}. {product['title']}**")
            st.caption(f"Сходство: {score:.3f}")
            st.divider()

    with right:
        st.subheader("🔤 Поиск по словам")

        if not keyword_results:
            st.warning("Совпадений не найдено")

        for rank, (index, score) in enumerate(keyword_results, 1):
            filename = filenames[index]
            product = catalog[filename]

            st.image(str(IMAGE_DIR / filename), width=220)
            st.write(f"**{rank}. {product['title']}**")
            st.caption(f"Совпадений: {score}")
            st.divider()
else:
    st.info("Введите запрос для поиска товаров")
