from pathlib import Path
from PIL import Image
from PIL.ExifTags import TAGS
from datetime import datetime, timedelta
import numpy as np
from tqdm import tqdm
import torch
import clip


from superlinked.framework.common.schema.id_schema_object import IdField
from superlinked.framework.common.schema.schema import schema
from superlinked.framework.common.schema.schema_object import Array, Float, Timestamp
from superlinked.framework.dsl.index.index import Index
from superlinked.framework.dsl.space.custom_space import CustomSpace
from superlinked.framework.dsl.query.param import Param
from superlinked.framework.dsl.space.number_space import NumberSpace, Mode
from superlinked.framework.dsl.executor.in_memory.in_memory_executor import (
    InMemoryExecutor,
)
from superlinked.framework.dsl.space.recency_space import RecencySpace
from superlinked.framework.common.dag.period_time import PeriodTime
from superlinked.framework.dsl.source.in_memory_source import InMemorySource
from superlinked.framework.dsl.query.query import Query

from constants import (
    IMAGE_SIZE,
    EMBEDDING_SIZE,
    MAX_BRIGHTNESS,
    QEURY_LIMIT,
    DAYS_PER_YEAR,
)


def get_image_creation_time(image: Image) -> datetime:
    exif_data = image._getexif()

    if exif_data is None:
        return

    for tag_id in exif_data:
        tag = TAGS.get(tag_id, tag_id)
        if tag == "DateTimeOriginal":
            date_str = exif_data[tag_id]
            date_obj = datetime.strptime(date_str, "%Y:%m:%d %H:%M:%S")
            return date_obj


def calculate_brightness(image: Image) -> float:
    grayscale_image = image.convert("L")
    mean_brightness = sum(grayscale_image.getdata()) / (
        grayscale_image.width * grayscale_image.height
    )
    return mean_brightness


def load_dataset(folder: Path, preprocess) -> list[dict]:
    items = []

    filenames = [
        filename for pattern in ("*.jpg", "*.png") for filename in folder.glob(pattern)
    ]

    for filename in tqdm(filenames):
        image_raw = Image.open(filename)
        image_raw.thumbnail((IMAGE_SIZE, IMAGE_SIZE))
        brightness = calculate_brightness(image_raw)
        creation_time = get_image_creation_time(image_raw)
        image = preprocess(image_raw)
        item = {
            "filename": filename,
            "name": filename.name,
            "image_raw": image_raw,
            "image_preprocessed": image,
            "brightness": brightness,
            "creation_time": creation_time,
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
        creation_time: Timestamp

    photo = Photo()

    photo_features_space = CustomSpace(vector=photo.features, length=EMBEDDING_SIZE)
    photo_brightness_space = NumberSpace(
        number=photo.brightness,
        min_value=0,
        max_value=MAX_BRIGHTNESS,
        mode=Mode.SIMILAR,
    )

    photo_recency_space = RecencySpace(
        timestamp=photo.creation_time,
        period_time_list=[
            PeriodTime(timedelta(days=2 * DAYS_PER_YEAR)),
            PeriodTime(timedelta(days=5 * DAYS_PER_YEAR)),
        ],
    )

    photo_index = Index(
        spaces=[photo_features_space, photo_brightness_space, photo_recency_space]
    )
    source = InMemorySource(photo)
    executor = InMemoryExecutor(sources=[source], indices=[photo_index])

    photo_query = (
        Query(
            photo_index,
            weights={
                photo_features_space: Param("features_weight"),
                photo_brightness_space: Param("brightness_weight"),
                photo_recency_space: Param("recency_weight"),
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
        item = {
            "name": x["name"],
            "brightness": x["brightness"],
            "creation_time": datetime.timestamp(x["creation_time"]),
            "features": features,
        }
        items.append(item)

    source.put(items)
