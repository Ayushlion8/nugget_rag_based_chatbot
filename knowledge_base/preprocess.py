# ZOMATO_RAG_CHATBOT/nugget_rag_based_chatbot/knowledge_base/preprocess.py

import json
import os
import re
import pickle

# --- Configuration ---
# Assuming script is run from the root 'ZOMATO_RAG_CHATBOT/nugget_rag_based_chatbot/' directory
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) # Gets nugget_rag_based_chatbot directory
RAW_DATA_PATH = os.path.join(BASE_DIR, "data", "raw_data", "lucknow_top50_with_menus.json")
CHUNK_OUTPUT_PATH = os.path.join(BASE_DIR, "data", "lucknow_with_menus_kb_chunks.pkl")
METADATA_OUTPUT_PATH = os.path.join(BASE_DIR, "data", "lucknow_with_menus_kb_metadata.pkl")
OUTPUT_DATA_DIR = os.path.join(BASE_DIR, "data")

# --- Helper Functions ---

def load_raw_data(filepath: str) -> list[dict]:
    """Loads restaurant data from a JSON file."""
    print(f"Loading raw data from: {filepath}")
    if not os.path.exists(filepath):
        print(f"Error: Raw data file not found at {filepath}")
        raise FileNotFoundError(f"File not found: {filepath}")
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        if not isinstance(data, list):
            raise ValueError("Expected a list of restaurants in the JSON file.")
        print(f"Successfully loaded {len(data)} restaurant entries.")
        return data
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from {filepath}")
        raise
    except ValueError as ve:
        print(f"Error: {ve}")
        raise
    except Exception as e:
        print(f"An unexpected error occurred during loading: {e}")
        raise

def preprocess_text(text: str | None) -> str:
    """Basic text cleaning."""
    if text is None:
        return ""
    text = str(text).lower() # Lowercase
    text = re.sub(r'\s+', ' ', text).strip() # Replace multiple whitespace with single space
    # Add more cleaning steps if needed (e.g., remove special characters, normalize currency)
    return text

def create_documents_and_metadata(restaurant_data: list[dict]) -> tuple[list[str], list[dict]]:
    """
    Transforms restaurant data into indexable text chunks and creates metadata.
    """
    chunks = []
    metadata_list = []

    print("Creating text chunks and metadata...")
    for i, restaurant in enumerate(restaurant_data):
        # --- Basic Validation ---
        if not isinstance(restaurant, dict):
             print(f"Warning: Skipping invalid restaurant entry (not a dict) at index {i}: {restaurant}")
             continue

        resto_name = preprocess_text(restaurant.get("restaurant_name", f"Unknown Restaurant {i}"))
        if not resto_name or resto_name == f"unknown restaurant {i}":
             print(f"Warning: Skipping restaurant entry at index {i} due to missing/invalid name.")
             continue # Skip if name is missing, crucial for context

        resto_loc = preprocess_text(restaurant.get("location", ""))
        resto_hours = preprocess_text(restaurant.get("operating_hours", ""))
        resto_contact = preprocess_text(restaurant.get("contact_info", ""))
        resto_features = ", ".join(filter(None, map(preprocess_text, restaurant.get("special_features", []))))
        resto_url = restaurant.get("website", restaurant.get("website_scraped", "")) # Get URL if available

        # --- Chunk 1: General Restaurant Info ---
        general_info_text = (
            f"Restaurant Name: {resto_name}. "
            f"Location: {resto_loc}. "
            f"Operating Hours: {resto_hours}. "
            f"Contact: {resto_contact}. "
            f"Features: {resto_features}."
        )
        chunks.append(general_info_text)
        metadata_list.append({
            "doc_type": "general_info",
            "restaurant_index": i,
            "restaurant_name": resto_name,
            "location": resto_loc,
            "hours": resto_hours,
            "contact": resto_contact,
            "features": restaurant.get("special_features", []), # Keep original list potentially
            "url": resto_url,
            "text_chunk": general_info_text # Store the text itself
        })

        # --- Chunks 2..N: Menu Items ---
        menu = restaurant.get("menu", [])
        if isinstance(menu, list):
            for j, item in enumerate(menu):
                if not isinstance(item, dict):
                    print(f"Warning: Skipping invalid menu item (not a dict) in '{resto_name}': {item}")
                    continue

                item_name = preprocess_text(item.get("item_name", ""))
                if not item_name: # Skip items without names
                     print(f"Warning: Skipping menu item in '{resto_name}' due to missing name: {item}")
                     continue

                item_desc = preprocess_text(item.get("description", ""))
                item_price = preprocess_text(item.get("price", "Price not listed")) # Keep raw price as text
                item_cat = preprocess_text(item.get("category", "Uncategorized"))
                item_flags = list(filter(None, map(preprocess_text, item.get("dietary_flags", []))))
                item_spice = preprocess_text(item.get("spice_level", ""))

                menu_item_text = (
                    f"Restaurant: {resto_name}. "
                    f"Menu Item: {item_name}. "
                    f"Category: {item_cat}. "
                    f"Description: {item_desc}. "
                    f"Price: {item_price}. "
                    f"Dietary Information: {', '.join(item_flags) if item_flags else 'None'}. "
                    f"{'Spice Level: ' + item_spice + '.' if item_spice else ''}"
                ).strip()

                chunks.append(menu_item_text)
                metadata_list.append({
                    "doc_type": "menu_item",
                    "restaurant_index": i,
                    "menu_item_index": j,
                    "restaurant_name": resto_name,
                    "item_name": item.get("item_name", ""), # Keep original case potentially
                    "category": item.get("category", "Uncategorized"),
                    "description": item.get("description", ""),
                    "price": item.get("price", "Price not listed"),
                    "dietary_flags": item.get("dietary_flags", []),
                    "spice_level": item.get("spice_level", ""),
                    "url": resto_url,
                    "text_chunk": menu_item_text
                })
        elif menu is not None: # Only warn if menu exists but isn't a list
             print(f"Warning: Menu data for '{resto_name}' is not a list, skipping menu items. Found type: {type(menu)}")

    print(f"Created {len(chunks)} text chunks and {len(metadata_list)} metadata entries.")
    return chunks, metadata_list

def save_processed_data(chunks: list[str], metadata: list[dict], chunk_path: str, metadata_path: str):
    """Saves the processed chunks and metadata using pickle."""
    print(f"Saving processed chunks to: {chunk_path}")
    try:
        with open(chunk_path, 'wb') as f:
            pickle.dump(chunks, f)
    except Exception as e:
        print(f"Error saving chunks file: {e}")
        raise

    print(f"Saving metadata to: {metadata_path}")
    try:
        with open(metadata_path, 'wb') as f:
            pickle.dump(metadata, f)
    except Exception as e:
        print(f"Error saving metadata file: {e}")
        raise

    print("Processed data saved successfully.")

# --- Main Execution ---
if __name__ == "__main__":
    print("--- Starting Knowledge Base Preprocessing ---")

    # Ensure output directory exists
    os.makedirs(OUTPUT_DATA_DIR, exist_ok=True)

    try:
        # 1. Load Data
        raw_restaurant_data = load_raw_data(RAW_DATA_PATH)

        if not raw_restaurant_data:
            print("Exiting: No data loaded.")
        else:
            # 2. Create Chunks and Metadata
            text_chunks, metadata = create_documents_and_metadata(raw_restaurant_data)

            if not text_chunks:
                print("Exiting: No text chunks created for indexing.")
            else:
                # 3. Save Processed Data
                save_processed_data(text_chunks, metadata, CHUNK_OUTPUT_PATH, METADATA_OUTPUT_PATH)

    except Exception as main_e:
        print(f"\nAn error occurred during preprocessing: {main_e}")

    print("--- Knowledge Base Preprocessing Finished ---")