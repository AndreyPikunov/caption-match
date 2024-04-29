from datetime import datetime

import streamlit as st
from pydantic import BaseModel

from superlinked.framework.dsl.query.result import Result

from caption_match.config import settings

from caption_match.superlinked_client import SLClient, PhotoQueryParams
from caption_match.embedder import Embedder
from caption_match.photo_loader import PhotoLoader
from caption_match.rerank import RerankRow, Reranker


class DisplayEntry(BaseModel):
    filename: str
    brightness: float
    superlinked_score: float
    creation_timestamp: float
    reranking_score: float = 0.0
    reranking_description: str | None = None


def prettify_score(score: float) -> float:
    return round(score * 100, 2)


def craete_markdown_caption(
    display_entry: DisplayEntry, scores_only: bool = False
) -> str:
    similarity = prettify_score(display_entry.superlinked_score)
    brightness = int(display_entry.brightness)
    creation_timestamp = display_entry.creation_timestamp
    creation_datetime = datetime.fromtimestamp(creation_timestamp)
    creation_time_str = datetime.strftime(creation_datetime, "%Y/%m/%d")

    caption = ""
    if not scores_only:
        caption += f"📅: {creation_time_str}\n\n" f"☀️: {brightness}\n\n"

    caption += f"\n\n**Retrieval score:** {similarity}"

    if display_entry.reranking_description is not None:
        reranking_score = prettify_score(display_entry.reranking_score)
        reranking_description = display_entry.reranking_description
        caption += (
            f"\n\n**Reranking score:** {reranking_score}\n\n"
            f"**Reranking desc:** {reranking_description}"
        )
    else:
        caption += "\n\n**No reranking**"

    return caption


def display_images(
    display_entries: list[DisplayEntry],
    photo_loader: PhotoLoader,
    scores_only: bool = False,
):

    n_cols = len(display_entries)

    image_cols = st.columns(n_cols)
    caption_cols = st.columns(n_cols)

    for display_entry, image_col, caption_col in zip(
        display_entries, image_cols, caption_cols
    ):
        filename = display_entry.filename
        image = photo_loader.load_image(filename)
        caption = craete_markdown_caption(display_entry, scores_only=scores_only)
        image_col.image(image, use_column_width=True)
        caption_col.markdown(caption)


def init_session() -> dict[str, object]:

    embedding_size = settings.embedder.embedding_size
    year_deltas = settings.superlinked.year_deltas

    sl_client = SLClient(embedding_size=embedding_size, year_deltas=year_deltas)
    sl_client.run()

    embedder = Embedder(model_name=settings.embedder.model_name)

    photo_loader = PhotoLoader(
        path=settings.photo_loader.path,
        image_resize=settings.photo_loader.image_resize,
        extensions=settings.photo_loader.extensions,
    )

    reranker = Reranker()

    objects = {
        "sl_client": sl_client,
        "embedder": embedder,
        "photo_loader": photo_loader,
        "reranker": reranker,
    }

    return objects


def populate_source(sl_client: SLClient, photo_loader: PhotoLoader, embedder: Embedder):

    for images, attributes in photo_loader.batch(settings.superlinked.put_batch_size):
        embeddings = embedder.embed_images(images)
        data = []
        for embedding, attributes in zip(embeddings, attributes):
            data_item = {
                "filename": attributes.filename,
                "brightness": attributes.brightness,
                "creation_timestamp": datetime.timestamp(attributes.creation_datetime),
                "features": embedding,
            }
            data.append(data_item)
        sl_client.source.put(data)


def rerank_entries(
    *,
    caption: str,
    retrieval_result: Result,
    photo_loader: PhotoLoader,
    reranker: Reranker,
) -> list[RerankRow]:
    images_base64 = [
        photo_loader.load_image_base64(entry.stored_object["filename"])
        for entry in retrieval_result.entries
    ]
    rows = reranker.rerank(caption, images_base64)
    return rows


def main():

    if "sl_client" not in st.session_state:

        objects = init_session()

        populate_source(
            sl_client=objects["sl_client"],  # type: ignore
            photo_loader=objects["photo_loader"],  # type: ignore
            embedder=objects["embedder"],  # type: ignore
        )

        st.session_state.update(**objects)

    sl_client: SLClient = st.session_state["sl_client"]
    embedder: Embedder = st.session_state["embedder"]
    photo_loader: PhotoLoader = st.session_state["photo_loader"]
    reranker: Reranker = st.session_state["reranker"]

    caption = st.text_input("Caption", "Photo of a dog laying on its back on a sofa")

    show_sliders = st.toggle("More parameters?")
    if show_sliders:
        features_weight = st.slider(
            "🗯️ Caption weight", min_value=-1.0, max_value=1.0, value=1.0
        )
        recency_weight = st.slider(
            "📅 Recency weight", min_value=-1.0, max_value=1.0, value=0.0
        )
        brightness_weight = st.slider(
            "☀️ Brightness weight", min_value=-1.0, max_value=1.0, value=0.0
        )
        brightness = st.slider(
            "Target brightness",
            min_value=0,
            max_value=settings.MAX_BRIGHTNESS,
            value=settings.MAX_BRIGHTNESS,
        )
    else:
        features_weight = 1.0
        brightness = settings.MAX_BRIGHTNESS
        recency_weight = 0.0
        brightness_weight = 0.0

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

    if st.button("Rerank by caption"):

        rerank_rows = rerank_entries(
            caption=caption,
            retrieval_result=result,
            photo_loader=photo_loader,
            reranker=reranker,
        )

        for display_entry, row in zip(display_entries, rerank_rows):
            display_entry.reranking_score = row.score
            display_entry.reranking_description = row.description

        display_entries.sort(
            key=lambda x: x.reranking_score,
            reverse=True,
        )

    display_images(display_entries, photo_loader, scores_only=not show_sliders)


if __name__ == "__main__":
    main()
