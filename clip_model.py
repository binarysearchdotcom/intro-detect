import torch
import numpy as np
import open_clip
from PIL import Image
from tqdm import tqdm


class CLIPModel:
    def __init__(
        self,
        model_name="ViT-B-16-SigLIP-512",
        weights="webli",
        device=None,
        batch_size=128,
        show_progress=False,
    ):
        self.model_name = model_name
        self.weights = weights
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.batch_size = batch_size
        self.show_progress = show_progress
        self.model = None
        self.preprocess = None

    def _load_model(self):
        if self.model is None or self.preprocess is None:
            self.model, _, self.preprocess = open_clip.create_model_and_transforms(
                self.model_name, pretrained=self.weights
            )
            self.model = self.model.to(self.device).eval()

    def encode(self, images: list[Image.Image]) -> np.ndarray:
        self._load_model()
        embs = []
        iterator = range(0, len(images), self.batch_size)
        if self.show_progress:
            iterator = tqdm(iterator, desc="Encoding with CLIP", unit="batch")

        for i in iterator:
            batch = images[i : i + self.batch_size]
            tensor = torch.stack([self.preprocess(img) for img in batch]).to(
                self.device
            )
            with torch.no_grad():
                out = self.model.encode_image(tensor).cpu().numpy()
            embs.append(out)

        return np.vstack(embs) if embs else np.empty((0, 512), dtype=np.float32)
