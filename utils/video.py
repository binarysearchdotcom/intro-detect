import cv2
from PIL import Image


def extract_frames(video_path: str, fps: int, resize: tuple[int, int], first_pct: int):
    cap = cv2.VideoCapture(video_path)
    video_fps = cap.get(cv2.CAP_PROP_FPS) or 25
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    max_frame = int(total * first_pct / 100)
    step = max(1, int(round(video_fps / fps)))

    frames, idx = [], 0
    while cap.isOpened() and idx < max_frame:
        ok, frame = cap.read()
        if not ok:
            break
        if idx % step == 0:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = cv2.resize(frame, resize)
            frames.append(Image.fromarray(frame))
        idx += 1
    cap.release()
    return frames
