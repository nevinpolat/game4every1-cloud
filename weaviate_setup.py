# weaviate_setup.py

import weaviate
from weaviate.auth import AuthApiKey
import pandas as pd
import re
from sentence_transformers import SentenceTransformer
import numpy as np
import streamlit as st

# Module-level variables to keep track of initialization
schema_initialized = False
data_ingested = False

def get_weaviate_client():
    WEAVIATE_URL = st.secrets["WEAVIATE"]["URL"]
    WEAVIATE_API_KEY = st.secrets["WEAVIATE"]["API_KEY"]

    auth_config = AuthApiKey(api_key=WEAVIATE_API_KEY) if WEAVIATE_API_KEY else None

    client = weaviate.Client(
        url=WEAVIATE_URL,
        auth_client_secret=auth_config
    )
    return client

def create_schema():
    client = get_weaviate_client()

    # Check if 'Game' class exists
    schema = client.schema.get()
    classes = [c['class'] for c in schema.get('classes', [])]

    if 'Game' not in classes:
        # Define the schema for the Game class
        game_class_schema = {
            "class": "Game",
            "vectorizer": "none",
            "properties": [
                {"name": "gameId", "dataType": ["string"]},
                {"name": "gameName", "dataType": ["string"]},
                {"name": "alternateNames", "dataType": ["string"]},
                {"name": "subcategory", "dataType": ["string"]},
                {"name": "level", "dataType": ["string"]},
                {"name": "description", "dataType": ["text"]},
                {"name": "playersMax", "dataType": ["int"]},
                {"name": "ageRange", "dataType": ["string"]},
                {"name": "duration", "dataType": ["int"]},
                {"name": "equipmentNeeded", "dataType": ["string"]},
                {"name": "objective", "dataType": ["string"]},
                {"name": "skillsDeveloped", "dataType": ["string"]},
                {"name": "setupTime", "dataType": ["int"]},
                {"name": "place", "dataType": ["string"]},
                {"name": "physicalIntensityLevel", "dataType": ["string"]},
                {"name": "educationalBenefits", "dataType": ["string"]},
                {"name": "category", "dataType": ["string"]},
            ]
        }
        client.schema.create_class(game_class_schema)
        st.write("✅ Schema for 'Game' class created successfully.")
    # Do not print anything if the schema already exists

def ingest_data():
    client = get_weaviate_client()

    # Check if data is already ingested
    result = client.query.aggregate("Game").with_meta_count().do()
    count = result["data"]["Aggregate"]["Game"][0]["meta"]["count"]
    if count == 0:
        # Initialize Sentence Transformer model
        model = SentenceTransformer('all-MiniLM-L6-v2')

        # Read the CSV file
        data = pd.read_csv("data/game-dataset.csv")

        # Include your helper functions like extract_duration, parse_players_max, etc.

        # Iterate through each row and create a Weaviate object
        for _, row in data.iterrows():
            # Prepare the game data with error handling
            game_data = {
                "gameId": str(row['gameId']) if pd.notnull(row['gameId']) else "",
                "gameName": str(row['gameName']) if pd.notnull(row['gameName']) else "Unknown Game",
                # ... include all your fields
            }

            # Vectorize the game description
            description_vector = model.encode(game_data['description']).tolist()

            # Add the object to Weaviate with the vector
            try:
                client.data_object.create(
                    data_object=game_data,
                    class_name="Game",
                    vector=description_vector
                )
            except Exception as e:
                print(f"Failed to insert game: {game_data['gameName']}. Error: {e}")

        st.write("✅ Data ingestion completed.")
    # Do not print anything if data is already ingested

def initialize_weaviate():
    global schema_initialized
    global data_ingested

    if not schema_initialized:
        create_schema()
        schema_initialized = True

    if not data_ingested:
        ingest_data()
        data_ingested = True
