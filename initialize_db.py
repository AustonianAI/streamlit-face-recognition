from datasets import load_dataset

from typing import List, Dict, Tuple
from PIL import Image
import face_recognition
from flupy import flu
import numpy as np
from tqdm import tqdm
import vecs

from streamlit_app import create_vector_database_client


people = load_dataset("ashraq/tmdb-people-image", split="train")

vx = create_vector_database_client()

try:
    faces = vx.create_collection(name="faces", dimension=128)
except vecs.errors.CollectionAlreadyExists:
    faces = vx.get_collection(name="faces")

# Records we'll insert into the database
records: List[Tuple[str, np.ndarray, Dict]] = []

# Iterate over the dataset in chunks
for ix, person in tqdm(enumerate(people)):

    # Limit the number of records we insert, it takes several hours to do all the records
    if ix > 1000:
        break

    # Extract the person's image
    person_image = person['image']

    # Some of the images are grayscale with a single image channel
    # We'll normalize the image set by converting those to 3 channel RBG format
    if person_image.mode == 'L':
        # Extract the available channel
        L_channel = np.array(person_image)

        # Repeat that channel 3 times for R G B
        person_image = Image.fromarray(
            np.moveaxis(np.stack([L_channel, L_channel, L_channel]), 0, -1)
        )

    # Create embeddings for current chunk
    embeddings = face_recognition.face_encodings(np.array(person_image))

    # In some cases the face is too obscured to be detectable and no embedding
    # is produced. We'll skip those cases
    if len(embeddings) == 1:
        embedding = embeddings[0]
        records.append((
            f"{ix}",
            embedding,
            {k: v for k, v in person.items() if k != 'image'}
        ))

# Insert the records into the database
faces.upsert(records)
faces.create_index()
