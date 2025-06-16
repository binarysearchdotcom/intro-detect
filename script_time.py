import json


def hms_to_seconds(hms: str) -> int:
    h, m, s = map(int, hms.strip().split(":"))
    return h * 3600 + m * 60 + s


with open("data/raw/labels_json/train_labels.json", "r", encoding="utf-8") as f:
    labels = json.load(f)

raw_durations = []
fixed_durations = []

for vid, info in labels.items():
    start = hms_to_seconds(info["start"])
    end = hms_to_seconds(info["end"])

    # Без исправлений
    raw_duration = end - start
    if raw_duration > 0:
        raw_durations.append(raw_duration)

    # С исправлением, если start > end
    if start > end:
        start -= 60
    fixed_duration = end - start
    if fixed_duration > 0:
        fixed_durations.append(fixed_duration)

print("Статистика заставок:\n")

if raw_durations:
    avg_raw = sum(raw_durations) / len(raw_durations)
    print(f"Без фикса:  средняя = {avg_raw:.2f} сек | примеров: {len(raw_durations)}")
else:
    print("Без фикса:   нет подходящих примеров")

if fixed_durations:
    avg_fixed = sum(fixed_durations) / len(fixed_durations)
    print(
        f"С фиксом:    средняя = {avg_fixed:.2f} сек | примеров: {len(fixed_durations)}"
    )
else:
    print("С фиксом:    нет подходящих примеров")
