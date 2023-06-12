from dotenv import load_dotenv
import requests
import streamlit as st
import numpy as np
import vecs


import os

from PIL import Image

import face_recognition

from typing import Dict, Any

load_dotenv()

IMG_URL = "https://www.themoviedb.org/t/p/w600_and_h900_bestv2/"


def create_vector_database_client():
    vx = vecs.create_client(os.getenv("DB_CONNECTION"))
    return vx


def set_page_config():
    st.set_page_config(
        page_icon="🤩",
        layout="wide",
        initial_sidebar_state="expanded",
        page_title="Who is your celebrity look-alike?",
    )


def upload_image():
    return st.file_uploader("Upload Your Image (we aren't saving anything)", type=['png', 'jpg', 'jpeg'])


def resize_image(image, max_width):
    width, height = image.size
    if width > max_width:
        ratio = max_width / width
        height = int(height * ratio)
        image = image.resize((max_width, height))
    return image


def find_similar_faces(person_image: Image) -> Image:
    face_encodings = face_recognition.face_encodings(np.array(person_image))

    if len(face_encodings) == 0:  # If list is empty, no faces were detected
        st.error("No faces detected in the image. Please upload a different image.")
        return

    face_to_search = face_encodings[0]

    vx = create_vector_database_client()
    faces = vx.get_collection(name="faces")

    # Create an empty element for displaying the loading message
    loading_message = st.empty()
    loading_message.text("Searching for similar faces...")

    similar_faces = faces.query(
        face_to_search, limit=6, include_metadata=True)

    # Update the loading message with the search results
    loading_message.empty()

    return similar_faces


def main():
    set_page_config()

    sidebar = st.sidebar

    with sidebar:
        uploaded_image = upload_image()

    st.title("Who is your celebrity look-alike?")
    st.text("Upload an image of yourself and we'll find your celebrity look-alike!")

    if uploaded_image:
        image = Image.open(uploaded_image)
        image = resize_image(image, 150)

        sidebar.image(image, caption="Your image", use_column_width=True)

        similar_faces = find_similar_faces(image)

        columns = col1, col2, col3 = st.columns(3)

        column_count = 0

        for idx, (id, metadata) in enumerate(similar_faces):
            name = metadata["name"]
            profile_path = metadata["profile_path"]

            image_url = IMG_URL + profile_path

            image_from_url = Image.open(
                requests.get(image_url, stream=True).raw)
            image_from_url = resize_image(image_from_url, 150)

            # Cycle through columns by using modulo operation
            columns[idx % len(columns)].image(image_from_url, caption=name)


if __name__ == "__main__":
    main()

# Add credit
st.markdown("""
---
Made with 🤖 by [Austin Johnson](https://github.com/AustonianAI)""")
