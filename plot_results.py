import csv
from pathlib import Path
import matplotlib.pyplot as plt


BASE_DIR = Path(__file__).parent


with open(
    BASE_DIR / "evaluation_results.csv",
    encoding="utf-8-sig",
    newline=""
) as file:
    results = list(csv.DictReader(file))

if not results:
    raise SystemExit("Файл с результатами пуст!")


def calculate_metric(column):
    correct = sum(int(row[column]) for row in results)
    return correct / len(results) * 100


clip_scores = [
    calculate_metric("clip_hit1"),
    calculate_metric("clip_hit5")
]

keyword_scores = [
    calculate_metric("keyword_hit1"),
    calculate_metric("keyword_hit5")
]


fig, ax = plt.subplots(figsize=(9, 5))

x = [0, 1]
width = 0.35

bars_clip = ax.bar(
    [i - width / 2 for i in x],
    clip_scores,
    width,
    label="CLIP"
)

bars_keyword = ax.bar(
    [i + width / 2 for i in x],
    keyword_scores,
    width,
    label="Keyword Search"
)

ax.bar_label(bars_clip, fmt="%.1f%%", padding=3)
ax.bar_label(bars_keyword, fmt="%.1f%%", padding=3)

ax.set_xticks(x, ["Hit@1", "Hit@5"])
ax.set_ylim(0, 115)

ax.set_ylabel("Accuracy (%)")
ax.set_title(
    f"CLIP vs Keyword Search\n"
    f"60 products, {len(results)} queries"
)

ax.legend()
ax.grid(axis="y", alpha=0.2)
ax.set_axisbelow(True)

plt.tight_layout()

output_dir = BASE_DIR / "docs" / "images"
output_dir.mkdir(parents=True, exist_ok=True)

output_file = output_dir / "comparison_chart.png"

plt.savefig(output_file, dpi=150)
plt.close()

print(f"График сохранён: {output_file}")
