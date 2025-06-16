import os
import json
import numpy as np
from tqdm import tqdm
from clip_model import CLIPModel
from utils.video import extract_frames
from utils.parse import parse_show_and_season, hms_to_sec

LABELS_PATH = "data/raw/labels_json/train_labels.json"
VIDEOS_ROOT = "data/raw/data_train_short"
SAVE_DIR = "embeddings_by_show_v2"
FPS = 2
RESIZE = (224, 224)
BATCH_SIZE = 64
FIRST_PCT = 15

clip = CLIPModel(batch_size=BATCH_SIZE, show_progress=False)

season_cache = {}  # {(show_name, season_num): {series_num: {start, end}}}


def save_clip(video_id: str, meta: dict):
    video_name = meta["name"]
    start_time, end_time = meta["start"], meta["end"]
    video_path = os.path.join(VIDEOS_ROOT, video_id, f"{video_id}.mp4")
    if not os.path.isfile(video_path):
        return

    show_name, season, series = parse_show_and_season(video_name)
    save_path = os.path.join(SAVE_DIR, show_name, f"{season} сезон")
    os.makedirs(save_path, exist_ok=True)

    frames = extract_frames(video_path, FPS, RESIZE, FIRST_PCT)
    embeddings = clip.encode(frames)

    np.save(os.path.join(save_path, f"серия_{series}.npy"), embeddings)

    start_sec = hms_to_sec(start_time)
    end_sec = hms_to_sec(end_time)
    if end_sec < start_sec:
        start_sec = max(0, start_sec - 60)

    key = (show_name, season)
    if key not in season_cache:
        season_cache[key] = {}
    season_cache[key][series] = {"start": start_sec, "end": end_sec}


def save_labels_per_season():
    for (show, season), label_dict in season_cache.items():
        season_dir = os.path.join(SAVE_DIR, show, f"{season} сезон")
        os.makedirs(season_dir, exist_ok=True)
        out_path = os.path.join(season_dir, "labels.json")
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(label_dict, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    with open(LABELS_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    for vid, meta in tqdm(data.items(), desc="Обработка серий"):
        try:
            save_clip(vid, meta)
        except Exception as e:
            print(f"Ошибка с {vid}: {e}")

    save_labels_per_season()
