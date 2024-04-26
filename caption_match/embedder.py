import clip
import torch
from PIL.Image import Image

from .config import settings


class Embedder:

    def __init__(self, model_name: str):
        self.model, self.preprocess = clip.load(
            model_name, download_root=settings.embedder.download_root
        )

    @staticmethod
    def listify(tensor: torch.Tensor) -> list[float]:
        return tensor.detach().numpy().tolist()

    def embed_images(self, images: list[Image]) -> list[float]:
        images_preprocessed = torch.stack([self.preprocess(image) for image in images])
        with torch.no_grad():
            embeddings = self.model.encode_image(images_preprocessed)
        embeddings = self.listify(embeddings)
        return embeddings

    def embed_captions(self, captions: list[str]) -> list[float]:
        tokens = clip.tokenize(captions)
        with torch.no_grad():
            embeddings = self.model.encode_text(tokens)
        embeddings = self.listify(embeddings)
        return embeddings
