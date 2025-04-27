# ZOMATO_RAG_CHATBOT/nugget_rag_based_chatbot/knowledge_base/indexer.py

import os
import pickle
import time
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

# --- Configuration ---
# Assuming script is run from the root 'ZOMATO_RAG_CHATBOT/nugget_rag_based_chatbot/' directory
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) # Gets nugget_rag_based_chatbot directory
CHUNK_INPUT_PATH = os.path.join(BASE_DIR, "data", "lucknow_with_menus_kb_chunks.pkl")
METADATA_INPUT_PATH = os.path.join(BASE_DIR, "data", "lucknow_with_menus_kb_metadata.pkl")
PROCESSED_DATA_DIR = os.path.join(BASE_DIR, "data", "processed_data")
INDEX_OUTPUT_PATH = os.path.join(PROCESSED_DATA_DIR, "lucknow_with_menus_restaurant_index.faiss")
MAPPING_OUTPUT_PATH = os.path.join(PROCESSED_DATA_DIR, "lucknow_with_menus_index_to_doc_mapping.pkl")

# Choose a sentence transformer model
EMBEDDING_MODEL_NAME = 'all-MiniLM-L6-v2' # Consistent with previous step

# --- Helper Functions ---

def load_processed_data(chunk_path: str, metadata_path: str) -> tuple[list[str], list[dict]]:
    """Loads the processed chunks and metadata from pickle files."""
    print(f"Loading processed chunks from: {chunk_path}")
    if not os.path.exists(chunk_path):
        print(f"Error: Chunks file not found at {chunk_path}")
        raise FileNotFoundError(f"File not found: {chunk_path}")
    try:
        with open(chunk_path, 'rb') as f:
            chunks = pickle.load(f)
    except Exception as e:
        print(f"Error loading chunks: {e}")
        raise

    print(f"Loading metadata from: {metadata_path}")
    if not os.path.exists(metadata_path):
        print(f"Error: Metadata file not found at {metadata_path}")
        raise FileNotFoundError(f"File not found: {metadata_path}")
    try:
        with open(metadata_path, 'rb') as f:
            metadata = pickle.load(f)
    except Exception as e:
        print(f"Error loading metadata: {e}")
        raise

    if not isinstance(chunks, list) or not isinstance(metadata, list):
        raise TypeError("Expected chunks and metadata to be lists.")

    if len(chunks) != len(metadata):
        print(f"Warning: Mismatch between number of chunks ({len(chunks)}) and metadata entries ({len(metadata)}).")
        # Decide how to handle: raise error or proceed with caution?
        # For now, let's raise an error as it indicates a problem in preprocessing.
        raise ValueError("Inconsistent number of chunks and metadata entries.")

    print(f"Loaded {len(chunks)} chunks and {len(metadata)} metadata entries.")
    return chunks, metadata


def generate_embeddings(documents: list[str], model_name: str) -> np.ndarray:
    """Generates embeddings for a list of text documents."""
    print(f"Loading embedding model: {model_name}")
    try:
        # Consider adding device='cuda' if GPU is available and faiss-gpu is installed
        model = SentenceTransformer(model_name)
        print("Generating embeddings (this might take a while)...")
        start_time = time.time()
        embeddings = model.encode(documents, show_progress_bar=True, convert_to_numpy=True)
        end_time = time.time()
        if not isinstance(embeddings, np.ndarray) or embeddings.ndim != 2:
             raise TypeError(f"Embeddings are not a valid numpy 2D array. Type: {type(embeddings)}")
        print(f"Generated {embeddings.shape[0]} embeddings of dimension {embeddings.shape[1]} in {end_time - start_time:.2f} seconds.")
        return embeddings
    except Exception as e:
        print(f"Error loading model or generating embeddings: {e}")
        raise

def build_faiss_index(embeddings: np.ndarray) -> faiss.Index:
    """Builds a FAISS index for the given embeddings."""
    if not isinstance(embeddings, np.ndarray) or embeddings.ndim != 2 or embeddings.shape[0] == 0:
        raise ValueError("Invalid embeddings provided for building the index.")

    dimension = embeddings.shape[1]
    print(f"Building FAISS index with dimension {dimension}...")

    # Using IndexFlatL2 - simple baseline for exact L2 distance search.
    index = faiss.IndexFlatL2(dimension)

    # Add the embeddings to the index
    try:
        index.add(embeddings.astype(np.float32)) # FAISS often expects float32
    except Exception as e:
        print(f"Error adding vectors to FAISS index: {e}")
        raise

    print(f"FAISS index built successfully. Total vectors indexed: {index.ntotal}")
    return index

def save_indexing_artifacts(index: faiss.Index, mapping_data: list[dict], index_path: str, mapping_path: str):
    """Saves the FAISS index and the final document mapping file."""
    print(f"Saving FAISS index to: {index_path}")
    try:
        # Ensure the output directory exists
        os.makedirs(os.path.dirname(index_path), exist_ok=True)
        faiss.write_index(index, index_path)
    except Exception as e:
        print(f"Error saving FAISS index: {e}")
        raise

    print(f"Saving index-to-document mapping to: {mapping_path}")
    try:
        # Ensure the output directory exists
        os.makedirs(os.path.dirname(mapping_path), exist_ok=True)
        with open(mapping_path, 'wb') as f:
            pickle.dump(mapping_data, f) # Save the metadata list as the mapping
    except Exception as e:
        print(f"Error saving mapping file: {e}")
        raise

    print("Indexing artifacts saved successfully.")

# --- Main Execution ---
if __name__ == "__main__":
    print("--- Starting Knowledge Base Indexing ---")

    # Ensure output directory exists
    os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)

    try:
        # 1. Load Processed Data
        text_chunks, metadata = load_processed_data(CHUNK_INPUT_PATH, METADATA_INPUT_PATH)

        if not text_chunks:
            print("Exiting: No text chunks loaded.")
        else:
            # 2. Generate Embeddings
            embeddings_array = generate_embeddings(text_chunks, EMBEDDING_MODEL_NAME)

            # 3. Build FAISS Index
            faiss_index = build_faiss_index(embeddings_array)

            # 4. Save Index and Mapping
            # We save the metadata list itself as the mapping. The RAG pipeline
            # will use the index returned by FAISS search to look up the corresponding
            # entry in this loaded metadata list.
            save_indexing_artifacts(faiss_index, metadata, INDEX_OUTPUT_PATH, MAPPING_OUTPUT_PATH)

    except FileNotFoundError as fnf_e:
        print(f"\nError: Required input file not found. Please ensure the required input files exist and 'preprocess.py' has been run successfully.")
        print(f"Details: {fnf_e}")
    except Exception as main_e:
        print(f"\nAn error occurred during indexing: {main_e}")

    print("--- Knowledge Base Indexing Finished ---")