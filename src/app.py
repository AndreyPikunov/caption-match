from pathlib import Path
import clip
import streamlit as st
from datetime import datetime

from utils import (
    load_dataset,
    embed_caption,
    create_superlinked_objects,
    populate_source,
)

from config import settings


def display_images(names, similarities, dataset, name_to_index):
    cols = st.columns(len(names))
    for col, name, similarity in zip(cols, names, similarities):
        index = name_to_index[name]
        item = dataset[index]
        image = item["image_raw"]
        brightness = item["brightness"]
        similarity = round(similarity * 100, 2)
        creation_time = datetime.strftime(item["creation_time"], "%Y/%m/%d")
        caption = (
            f"{creation_time}; brightness: {int(brightness)}; similarity: {similarity} %"
        )
        col.image(image, caption=caption, use_column_width=True)


@st.cache_resource
def load_model_and_data():
    model, preprocess = clip.load(settings.MODEL_NAME, download_root="./clip")
    dataset = load_dataset(Path(settings.PHOTO_FOLDER), preprocess)
    print("dataset len:", len(dataset))
    return model, dataset


def main():

    model, dataset = load_model_and_data()
    name_to_index = {x["name"]: i for i, x in enumerate(dataset)}

    if "source" not in st.session_state or "executor" not in st.session_state:
        source, executor, photo_query = create_superlinked_objects()
        sl_app = executor.run()
        populate_source(source, dataset=dataset, model=model)
        st.session_state["source"] = source
        st.session_state["executor"] = executor
        st.session_state["photo_query"] = photo_query
        st.session_state["sl_app"] = sl_app
    else:
        source = st.session_state["source"]
        executor = st.session_state["executor"]
        photo_query = st.session_state["photo_query"]
        sl_app = st.session_state["sl_app"]

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

    embedding = embed_caption(caption, model)
    result = sl_app.query(
        photo_query,
        features=embedding,
        brightness=brightness,
        features_weight=features_weight,
        brightness_weight=brightness_weight,
        recency_weight=recency_weight,
    )
    names = [entry.stored_object["name"] for entry in result.entries]
    similarities = [entry.entity.metadata.similarity for entry in result.entries]
    display_images(names, similarities, dataset=dataset, name_to_index=name_to_index)


if __name__ == "__main__":
    main()
