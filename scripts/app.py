from datetime import datetime

import streamlit as st
from pydantic import BaseModel


from caption_match.config import settings

from caption_match.superlinked_client import SLClient, PhotoQueryParams, Result
from caption_match.embedder import Embedder
from caption_match.photo_loader import PhotoLoader
from caption_match.rerank import Reranker


class DisplayEntry(BaseModel):
    filename: str
    brightness: float
    superlinked_score: float
    creation_timestamp: float
    reranking_score: float | None = None
    reranking_description: float | None = None


def prettify_score(score: float) -> float:
    return round(score * 100, 2)


def display_images(display_entries: list[DisplayEntry], photo_loader: PhotoLoader):

    n_cols = len(display_entries)

    image_cols = st.columns(n_cols)
    caption_cols = st.columns(n_cols)

    for display_entry, image_col, caption_col in zip(
        display_entries, image_cols, caption_cols
    ):
        filename = display_entry.filename
        image = photo_loader.load_image(filename)
        similarity = prettify_score(display_entry.superlinked_score)
        brightness = int(display_entry.brightness)
        creation_timestamp = display_entry.creation_timestamp
        creation_datetime = datetime.fromtimestamp(creation_timestamp)
        creation_time_str = datetime.strftime(creation_datetime, "%Y/%m/%d")
        caption = f"**Date:** {creation_time_str}\n\n**Brightness:** {brightness}\n\n**Similarity:** {similarity}"

        if display_entry.reranking_score is not None:
            reranking_score = prettify_score(display_entry.reranking_score)
            reranking_description = display_entry.reranking_description
            caption += f"\n\n**Reranking Score:** {reranking_score}\n\n**Reranking Description:** {reranking_description}"

        # Display the image in the first row
        image_col.image(image, use_column_width=True)

        # Display the caption in the second row
        caption_col.markdown(caption)


def main():

    if "sl_client" not in st.session_state:

        # status_text = st.empty()
        # status_text.write("Loading CLIP...")

        sl_client = SLClient(embedding_size=settings.embedder.embedding_size)
        embedder = Embedder(model_name=settings.embedder.model_name)
        photo_loader = PhotoLoader(
            path=settings.photo_loader.path,
            image_resize=settings.photo_loader.image_resize,
            extensions=settings.photo_loader.extensions,
        )

        # status_text.write("Loading Reranker...")
        reranker = Reranker()

        # status_text.write("Loading photos...")
        for images, attributes in photo_loader.batch(
            settings.superlinked.put_batch_size
        ):
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
        st.session_state["reranker"] = reranker

        # status_text.empty()

    else:
        sl_client = st.session_state["sl_client"]
        embedder = st.session_state["embedder"]
        photo_loader = st.session_state["photo_loader"]
        reranker = st.session_state["reranker"]

    caption = st.text_input("Caption", "winter, village, animals")
    brightness = st.number_input(
        "Target brightness",
        min_value=0,
        max_value=settings.MAX_BRIGHTNESS,
        value=settings.MAX_BRIGHTNESS,
    )
    features_weight = st.slider(
        "Features Weight", min_value=-1.0, max_value=1.0, value=1.0
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

    display_entries = [
        DisplayEntry(
            filename=entry.stored_object["filename"],
            brightness=entry.stored_object["brightness"],
            superlinked_score=entry.entity.metadata.similarity,
            creation_timestamp=entry.stored_object["creation_timestamp"],
        )
        for entry in result.entries
    ]

    display_entries.sort(key=lambda x: x.superlinked_score, reverse=True)

    if st.button("Rerank"):
        images_base64 = [
            photo_loader.load_image_base64(entry.stored_object["filename"])
            for entry in result.entries
        ]
        rows = reranker.rerank(caption, images_base64)
        # st.write(rows)

        for display_entry, row in zip(display_entries, rows):
            display_entry.reranking_score = row.score
            display_entry.reranking_description = row.description

        display_entries.sort(key=lambda x: x.reranking_score, reverse=True)

    display_images(display_entries, photo_loader)


if __name__ == "__main__":
    main()
