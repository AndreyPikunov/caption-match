from datetime import datetime
import os
import base64
from io import BytesIO

# from multiprocessing import Pool

from PIL import Image
from PIL.ExifTags import TAGS
from pydantic import BaseModel


class ImageAttributes(BaseModel):
    filename: str
    brightness: float
    creation_datetime: datetime


class PhotoLoader:
    def __init__(self, path, image_resize: int = 512, extensions=None):
        self.path = path
        self.image_resize = image_resize
        self.extensions = extensions
        if self.extensions is None:
            self.extensions = [".jpg", ".jpeg", ".png"]
        self.filenames = self._collect_filenames()

    def _collect_filenames(self):
        filenames = []
        for root, _, files in os.walk(self.path):
            for file in files:
                if file.lower().endswith(tuple(self.extensions)):
                    filenames.append(os.path.join(root, file))
        return filenames

    @staticmethod
    def get_image_creation_time(image: Image.Image) -> datetime:
        exif_data = image._getexif()

        if exif_data is None:
            return

        for tag_id in exif_data:
            tag = TAGS.get(tag_id, tag_id)
            if tag == "DateTimeOriginal":
                date_str = exif_data[tag_id]
                date_obj = datetime.strptime(date_str, "%Y:%m:%d %H:%M:%S")
                return date_obj

    @staticmethod
    def calculate_brightness(image: Image.Image) -> float:
        grayscale_image = image.convert("L")
        mean_brightness = sum(grayscale_image.getdata()) / (
            grayscale_image.width * grayscale_image.height
        )
        return mean_brightness

    @staticmethod
    def convert_pil_to_base64(image: Image.Image) -> str:
        buffered = BytesIO()
        image.save(buffered, format="JPEG")
        image_base64 = base64.b64encode(buffered.getvalue()).decode("utf-8")
        return image_base64

    def batch(self, size: int = 16):
        n = len(self.filenames)
        for i in range(0, n, size):
            filenames = self.filenames[i : i + size]
            images = self.load_images(filenames)
            attributes = []
            for image, filename in zip(images, filenames):
                attributes.append(
                    ImageAttributes(
                        filename=filename,
                        brightness=self.calculate_brightness(image),
                        creation_datetime=self.get_image_creation_time(image),
                    )
                )
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
