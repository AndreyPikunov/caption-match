from datetime import datetime
import os
from PIL import Image
from PIL.ExifTags import TAGS
from pydantic import BaseModel


class ImageAttributes(BaseModel):
    filename: str
    brightness: float
    creation_datetime: datetime


class PhotoLoader:
    def __init__(self, path, extensions=None):
        self.path = path
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

    def __len__(self):
        return len(self.filenames)

    def __iter__(self):
        self.__index = 0
        return self

    def __next__(self) -> tuple[Image.Image, ImageAttributes]:
        if self.__index >= len(self):
            raise StopIteration
        filename = self.filenames[self.__index]
        image = self.load_image(filename)
        attributes = ImageAttributes(
            filename=filename,
            brightness=self.calculate_brightness(image),
            creation_datetime=self.get_image_creation_time(image),
        )
        self.__index += 1
        return image, attributes

    def batch(self, size: int = 16):
        images, attributes = [], []
        for image, attribute in self:
            images.append(image)
            attributes.append(attribute)
            if len(images) == size:
                yield images, attributes
                images, attributes = [], []
        if images:
            yield images, attributes

    def load_image(self, filename: str) -> Image.Image:
        return Image.open(filename)
