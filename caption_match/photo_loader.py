from datetime import datetime
from pathlib import Path
import base64
from io import BytesIO
import logging
from typing import Generator

# from multiprocessing import Pool

from PIL import Image
from PIL.ExifTags import TAGS
from pydantic import BaseModel

logger = logging.getLogger(__name__)


class ImageAttributes(BaseModel):
    filename: str
    brightness: float
    creation_datetime: datetime


class PhotoLoader:
    def __init__(
        self, path, image_resize: int = 512, extensions: list[str] | None = None
    ):
        self.path = path
        self.image_resize = image_resize
        self.extensions = (
            extensions if extensions is not None else ["jpg", "jpeg", "png"]
        )
        self.filenames = self._collect_filenames()

    def _collect_filenames(self) -> list[str]:
        filenames = []
        path = Path(self.path)
        for ext in self.extensions:
            filenames.extend(str(x) for x in path.glob(f"*.{ext}"))
        return filenames

    @staticmethod
    def get_image_creation_time(image: Image.Image) -> datetime:
        exif_data = image._getexif()  # type: ignore

        if exif_data is None:
            logger.warning(
                "No EXIF data found in the image, setting creation time to now."
            )
            date_obj = datetime.now()
            return date_obj

        for tag_id in exif_data:
            tag = TAGS.get(tag_id)
            if tag == "DateTimeOriginal":
                date_str = exif_data[tag_id]
                date_obj = datetime.strptime(date_str, "%Y:%m:%d %H:%M:%S")
                return date_obj

        logger.warning(
            "EXIF does not contain DateTimeOriginal, setting creation time to now."
        )
        date_obj = datetime.now()
        return date_obj

    @staticmethod
    def calculate_brightness(image: Image.Image) -> float:
        grayscale_image = image.convert("L")
        mean_brightness = sum(grayscale_image.getdata()) / (  # type: ignore
            grayscale_image.width * grayscale_image.height
        )
        return mean_brightness

    def create_image_attributes(
        self, filename: str, image: Image.Image
    ) -> ImageAttributes:
        brightness = PhotoLoader.calculate_brightness(image)
        creation_datetime = PhotoLoader.get_image_creation_time(image)
        attributes = ImageAttributes(
            filename=filename,
            brightness=brightness,
            creation_datetime=creation_datetime,
        )
        return attributes

    @staticmethod
    def convert_pil_to_base64(image: Image.Image) -> str:
        buffered = BytesIO()
        image.save(buffered, format="JPEG")
        image_base64 = base64.b64encode(buffered.getvalue()).decode("utf-8")
        return image_base64

    def batch(
        self, size: int = 16
    ) -> Generator[tuple[list[Image.Image], list[ImageAttributes]], None, None]:
        n = len(self.filenames)
        for i in range(0, n, size):
            filenames = self.filenames[i : i + size]
            images = self.load_images(filenames)
            attributes = []
            for image, filename in zip(images, filenames):
                attrs = self.create_image_attributes(filename, image)
                attributes.append(attrs)
            yield images, attributes

    def load_image(self, filename: str) -> Image.Image:
        image = Image.open(filename)
        image.thumbnail((self.image_resize, self.image_resize))
        return image

    def load_image_base64(self, filename: str) -> str:
        image = self.load_image(filename)
        image_base64 = self.convert_pil_to_base64(image)
        return image_base64

    def load_images(self, filenames: list[str]) -> list[Image.Image]:
        # multiprocessings doesn't work properly with streamlit
        # with Pool() as pool:
        #     images = pool.map(self.load_image, filenames)
        images = [self.load_image(filename) for filename in filenames]
        return images
