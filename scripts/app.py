from datetime import datetime

import streamlit as st
from tqdm import tqdm

from caption_match.config import settings

from caption_match.superlinked_client import SLClient, PhotoQueryParams, Result
from caption_match.embedder import Embedder
from caption_match.photo_loader import PhotoLoader


def display_images(result: Result, photo_loader: PhotoLoader):

    cols = st.columns(len(result.entries))

    for entry, col in zip(result.entries, cols):
        stored_object = entry.stored_object
        filename = stored_object["filename"]
        image = photo_loader.load_image(filename)
        similarity = round(entry.entity.metadata.similarity * 100, 2)
        brightness = int(stored_object["brightness"])
        creation_timestamp = stored_object["creation_timestamp"]
        creation_datetime = datetime.fromtimestamp(creation_timestamp)
        creation_time_str = datetime.strftime(creation_datetime, "%Y/%m/%d")
        caption = f"{creation_time_str}; brightness: {int(brightness)}; similarity: {similarity} %"
        col.image(image, caption=caption, use_column_width=True)


def main():

    if "sl_client" not in st.session_state:
        sl_client = SLClient(embedding_size=settings.embedder.embedding_size)
        embedder = Embedder(model_name=settings.embedder.model_name)
        photo_loader = PhotoLoader(
            path=settings.photo_loader.path,
            image_resize=settings.photo_loader.image_resize,
            extensions=settings.photo_loader.extensions,
        )

        for images, attributes in photo_loader.batch(settings.superlinked.put_batch_size):
            embeddings = embedder.embed_images(images)
            data = []
            for embedding, attributes in zip(embeddings, attributes):
                data_item = {
                    "filename": attributes.filename,
                    "brightness": attributes.brightness,
                    "creation_timestamp": datetime.timestamp(
                        attributes.creation_datetime
                    ),
                    "features": embedding,
                }
                data.append(data_item)
            sl_client.source.put(data)

        st.session_state["sl_client"] = sl_client
        st.session_state["embedder"] = embedder
        st.session_state["photo_loader"] = photo_loader

    else:
        sl_client = st.session_state["sl_client"]
        embedder = st.session_state["embedder"]
        photo_loader = st.session_state["photo_loader"]

    caption = st.text_input("Caption", "Enter a caption")
    brightness = st.number_input(
        "Target brightness",
        min_value=0,
        max_value=settings.MAX_BRIGHTNESS,
        value=settings.MAX_BRIGHTNESS,
    )
    features_weight = st.slider(
        "Features Weight", min_value=-1.0, max_value=1.0, value=0.0
    )
    recency_weight = st.slider(
        "Recency Weight", min_value=-1.0, max_value=1.0, value=0.0
    )
    brightness_weight = st.slider(
        "Brightness Weight", min_value=-1.0, max_value=1.0, value=0.0
    )

    embedding = embedder.embed_captions([caption])[0]

    params = PhotoQueryParams(
        features=embedding,
        brightness=brightness,
        features_weight=features_weight,
        brightness_weight=brightness_weight,
        recency_weight=recency_weight,
    )

    result = sl_client.query_full(params)
    display_images(result, photo_loader)


if __name__ == "__main__":
    main()
