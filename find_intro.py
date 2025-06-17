import numpy as np
import faiss
from utils.parse import frames_to_hms


path_a = "embeddings_by_show_v2/Блеск/2 сезон/серия_2.npy"
path_b = "embeddings_by_show_v2/Блеск/2 сезон/серия_3.npy"
MIN_LEN, MAX_LEN = 2, 30  # окно, кадры
FPS = 2
SCORE_TH = 0.915
MERGE_GAP = 15  # объединение, кадры


A = np.load(path_a).astype("float32")  # [Ta, 512]
B = np.load(path_b).astype("float32")  # [Tb, 512]

# faiss.normalize_L2 теоретически работает быстрее, чем numpy нормализация
faiss.normalize_L2(A)
faiss.normalize_L2(B)

# одна матрица скалярных произведений
S = A @ B.T  # shape [Ta, Tb]

# предварительные кумулятивные суммы по осям
cum_rows = np.cumsum(S, axis=0, dtype=np.float32)
cum_rows = np.vstack([np.zeros((1, S.shape[1]), dtype=np.float32), cum_rows])

results = []

for W in range(MIN_LEN, MAX_LEN + 1):
    # среднее по W строк (окно в A)
    row_avg = (cum_rows[W:] - cum_rows[:-W]) / W  # [Ta-W+1, Tb]

    # среднее по W столбцов (окно в B)
    cum_cols = np.cumsum(row_avg, axis=1, dtype=np.float32)
    cum_cols = np.hstack([np.zeros((row_avg.shape[0], 1), dtype=np.float32), cum_cols])
    scores = (cum_cols[:, W:] - cum_cols[:, :-W]) / W  # [Ra, Rb]

    # кандидаты ≥ порога
    mask = scores >= SCORE_TH
    idx_rows, idx_cols = np.nonzero(mask)
    for r, c in zip(idx_rows, idx_cols):
        results.append(
            {
                "score": float(scores[r, c]),
                "start_A": int(r),
                "start_B": int(c),
                "end_B": int(c + W),
            }
        )

# объединение близких фрагментов
results.sort(key=lambda x: x["start_B"])
merged = []
for res in results:
    if not merged:
        merged.append(res)
    else:
        last = merged[-1]
        if res["start_B"] <= last["end_B"] + MERGE_GAP:
            last["end_B"] = max(last["end_B"], res["end_B"])
            last["score"] = max(last["score"], res["score"])
        else:
            merged.append(res)


if merged:
    print("Найденные сегменты в B:")
    for i, seg in enumerate(merged, 1):
        s, e = seg["start_B"], seg["end_B"]
        print(
            f"#{i}: кадры {s}–{e} "
            f"({frames_to_hms(s, FPS)} – {frames_to_hms(e, FPS)}), "
            f"длительность {(e - s) / FPS:.2f} сек, "
            f"max score {seg['score']:.4f}"
        )
else:
    print("Совпадений выше порога не найдено.")
