from pathlib import Path
from PIL import Image
import numpy as np
from tqdm import tqdm
import torch
import clip


from superlinked.framework.common.schema.id_schema_object import IdField
from superlinked.framework.common.schema.schema import schema
from superlinked.framework.common.schema.schema_object import Array, Float
from superlinked.framework.dsl.index.index import Index
from superlinked.framework.dsl.space.custom_space import CustomSpace
from superlinked.framework.dsl.query.param import Param
from superlinked.framework.dsl.space.number_space import NumberSpace, Mode
from superlinked.framework.dsl.executor.in_memory.in_memory_executor import (
    InMemoryExecutor,
)
from superlinked.framework.dsl.source.in_memory_source import InMemorySource
from superlinked.framework.dsl.query.query import Query

from constants import IMAGE_SIZE, EMBEDDING_SIZE, MAX_BRIGHTNESS, QEURY_LIMIT


def calculate_brightness(image: Image) -> float:
    grayscale_image = image.convert("L")
    mean_brightness = sum(grayscale_image.getdata()) / (grayscale_image.width * grayscale_image.height)
    return mean_brightness


def load_dataset(folder: Path, preprocess) -> list[dict]:
    items = []
    for filename in tqdm(folder.glob("*.jpg")):
        image_raw = Image.open(filename)
        image_raw.thumbnail((IMAGE_SIZE, IMAGE_SIZE))
        brightness = calculate_brightness(image_raw)
        image = preprocess(image_raw)
        item = {
            "filename": filename,
            "name": filename.name,
            "image_raw": image_raw,
            "image_preprocessed": image,
            "brightness": brightness,
        }
        items.append(item)
    return items


def embed_caption(caption: str, model) -> np.ndarray:
    texts = clip.tokenize([caption])
    with torch.no_grad():
        embeding = model.encode_text(texts)
    embeding = embeding.squeeze().detach().numpy()
    return embeding


def embed_image(image: torch.Tensor, model) -> np.ndarray:
    with torch.no_grad():
        embeding = model.encode_image(image.unsqueeze(0))
    embeding = embeding.squeeze().detach().numpy()
    return embeding


def create_superlinked_objects():
    @schema
    class Photo:
        name: IdField
        brightness: Float
        features: Array

    photo = Photo()

    photo_features_space = CustomSpace(vector=photo.features, length=EMBEDDING_SIZE)
    photo_brightness_space = NumberSpace(
        number=photo.brightness,
        min_value=0,
        max_value=MAX_BRIGHTNESS,
        mode=Mode.SIMILAR,
    )
    photo_index = Index(spaces=[photo_features_space, photo_brightness_space])
    source = InMemorySource(photo)
    executor = InMemoryExecutor(sources=[source], indices=[photo_index])

    photo_query = (
        Query(
            photo_index,
            weights={
                photo_features_space: Param("features_weight"),
                photo_brightness_space: Param("brightness_weight"),
            },
        )
        .find(photo)
        .similar(photo_features_space.vector, Param("features"))
        .similar(photo_brightness_space.number, Param("brightness"))
        .limit(QEURY_LIMIT)
    )

    return source, executor, photo_query


def populate_source(source: InMemorySource, dataset: list[dict], model):
    items = []

    for x in tqdm(dataset):
        features = embed_image(x["image_preprocessed"], model)
        item = {"name": x["name"], "brightness": x["brightness"], "features": features}
        items.append(item)

    source.put(items)
