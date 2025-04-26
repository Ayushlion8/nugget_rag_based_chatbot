# ZOMATO_RAG_CHATBOT/nugget_rag_based_chatbot/chatbot/rag_pipeline.py

import os
import pickle
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
import torch # Ensure PyTorch is installed

# --- Configuration ---
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) # nugget_rag_based_chatbot directory
PROCESSED_DATA_DIR = os.path.join(BASE_DIR, "data", "processed_data")
INDEX_PATH = os.path.join(PROCESSED_DATA_DIR, "restaurant_index.faiss")
MAPPING_PATH = os.path.join(PROCESSED_DATA_DIR, "index_to_doc_mapping.pkl")

# Models - Use the same embedding model used for indexing
EMBEDDING_MODEL_NAME = 'all-MiniLM-L6-v2'
# Choose a suitable generative model from Hugging Face Hub (free tier)
# Flan-T5 is a good choice for instruction following / Q&A
LLM_NAME = "google/flan-t5-base" # Options: "google/flan-t5-small", "google/flan-t5-base", "google/flan-t5-large" (larger needs more resources)

# Retrieval settings
NUM_DOCS_TO_RETRIEVE = 5 # How many relevant documents to fetch

# --- Global Variables (Load models once) ---
embedding_model = None
index = None
doc_mapping = None
llm_tokenizer = None
llm_model = None
device = "cuda" if torch.cuda.is_available() else "cpu" # Use GPU if available

# --- Initialization Function ---

def load_models_and_data():
    """Loads all necessary models and data artifacts."""
    global embedding_model, index, doc_mapping, llm_tokenizer, llm_model, device

    print("--- Loading RAG Pipeline Components ---")

    # 1. Load FAISS Index
    if not os.path.exists(INDEX_PATH):
        raise FileNotFoundError(f"FAISS index file not found at {INDEX_PATH}. Please run knowledge_base/indexer.py first.")
    print(f"Loading FAISS index from {INDEX_PATH}...")
    index = faiss.read_index(INDEX_PATH)
    print(f"FAISS index loaded. Contains {index.ntotal} vectors.")

    # 2. Load Index-to-Document Mapping
    if not os.path.exists(MAPPING_PATH):
        raise FileNotFoundError(f"Document mapping file not found at {MAPPING_PATH}. Please run knowledge_base/indexer.py first.")
    print(f"Loading document mapping from {MAPPING_PATH}...")
    with open(MAPPING_PATH, 'rb') as f:
        doc_mapping = pickle.load(f)
    if not isinstance(doc_mapping, list) or not doc_mapping:
        raise ValueError("Document mapping is empty or invalid.")
    print(f"Document mapping loaded. Contains {len(doc_mapping)} entries.")

    # Verify index and mapping length match
    if index.ntotal != len(doc_mapping):
         print(f"Warning: FAISS index size ({index.ntotal}) does not match mapping size ({len(doc_mapping)}). There might be an inconsistency.")
         # You might want to raise an error here depending on strictness

    # 3. Load Embedding Model (Sentence Transformer)
    print(f"Loading embedding model: {EMBEDDING_MODEL_NAME}...")
    # Specify device for SentenceTransformer
    embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME, device=device)
    print("Embedding model loaded.")

    # 4. Load Generative LLM and Tokenizer (from Hugging Face)
    print(f"Loading LLM and Tokenizer: {LLM_NAME}...")
    try:
        llm_tokenizer = AutoTokenizer.from_pretrained(LLM_NAME)
        # Load model onto the correct device (GPU or CPU)
        llm_model = AutoModelForSeq2SeqLM.from_pretrained(LLM_NAME).to(device)
        llm_model.eval() # Set model to evaluation mode
        print(f"LLM and Tokenizer loaded successfully onto {device}.")
    except Exception as e:
        print(f"Error loading LLM/Tokenizer: {e}")
        print("Please ensure you have a working internet connection and the model name is correct.")
        raise

    print("--- RAG Pipeline Components Loaded Successfully ---")

# --- Core RAG Functions ---

def retrieve_relevant_documents(query: str, k: int = NUM_DOCS_TO_RETRIEVE) -> list[dict]:
    """Retrieves the top-k most relevant document metadata based on the query."""
    if embedding_model is None or index is None or doc_mapping is None:
        raise RuntimeError("Models or data not loaded. Call load_models_and_data() first.")
    if not query:
        return []

    print(f"Retrieving documents for query: '{query}'")
    # 1. Embed the query
    query_embedding = embedding_model.encode([query], convert_to_numpy=True, device=device)

    # 2. Search the FAISS index
    # FAISS returns distances and indices. Indices correspond to the order in doc_mapping.
    try:
        distances, indices = index.search(query_embedding.astype(np.float32), k)
    except Exception as e:
        print(f"Error during FAISS search: {e}")
        return []

    # 3. Get the corresponding documents/metadata from the mapping
    retrieved_docs_metadata = []
    if indices.size > 0:
        for i in range(indices.shape[1]): # Iterate through the top k results
            doc_index = indices[0, i]
            # Ensure the index is valid
            if 0 <= doc_index < len(doc_mapping):
                retrieved_docs_metadata.append(doc_mapping[doc_index])
            else:
                print(f"Warning: Retrieved invalid index {doc_index} from FAISS.")

    print(f"Retrieved {len(retrieved_docs_metadata)} documents.")
    return retrieved_docs_metadata


def format_context(retrieved_docs: list[dict]) -> str:
    """Formats the retrieved document text chunks into a single context string."""
    if not retrieved_docs:
        return "No relevant information found in the knowledge base."

    context = ""
    for i, doc in enumerate(retrieved_docs):
        # Ensure 'text_chunk' key exists and is not empty
        chunk_text = doc.get("text_chunk", "").strip()
        if chunk_text:
            context += f"--- Document {i+1} ---\n"
            context += chunk_text + "\n\n" # Add separation between docs

    return context.strip() if context else "No relevant text found in retrieved documents."


def generate_response(query: str, context: str) -> str:
    """Generates a response using the LLM based on the query and retrieved context."""
    if llm_tokenizer is None or llm_model is None:
        raise RuntimeError("LLM components not loaded. Call load_models_and_data() first.")

    # Construct the prompt for the LLM
    # This prompt guides the LLM to answer based *only* on the provided context.
    prompt = f"""Answer the following question based *only* on the provided context. If the context does not contain the answer, say "I cannot answer this question based on the available information."

Context:
{context}

Question: {query}

Answer:"""

    print("Generating response...")
    try:
        # Tokenize the prompt and generate response
        inputs = llm_tokenizer(prompt, return_tensors="pt", max_length=1024, truncation=True).to(device)

        # Generate output tokens - adjust parameters as needed
        # max_length controls the maximum length of the generated response
        # num_beams increases quality but slows down generation
        with torch.no_grad(): # Disable gradient calculations for inference
             outputs = llm_model.generate(
                **inputs,
                max_length=256, # Adjust max response length
                num_beams=5,    # Use beam search for potentially better quality
                early_stopping=True,
                temperature=0.7, # Control randomness (lower = more deterministic)
                no_repeat_ngram_size=2 # Prevent repeating phrases
            )

        # Decode the generated tokens into text
        response_text = llm_tokenizer.decode(outputs[0], skip_special_tokens=True)

        print("Response generated.")
        return response_text.strip()

    except Exception as e:
        print(f"Error during LLM generation: {e}")
        return "Sorry, I encountered an error while generating the response."


# --- Main Pipeline Function ---

def get_rag_response(query: str) -> str:
    """
    The main RAG pipeline function.
    Takes a user query, retrieves relevant documents, and generates a response.
    """
    # Ensure models are loaded (call if first time)
    if embedding_model is None:
         try:
              load_models_and_data()
         except Exception as load_err:
              return f"Error loading RAG components: {load_err}"

    # 1. Retrieve relevant documents
    retrieved_docs = retrieve_relevant_documents(query)

    # Handle case where no documents are found
    if not retrieved_docs:
        return "I couldn't find any information related to your query in the restaurant database."

    # 2. Format the context
    context_str = format_context(retrieved_docs)

    # Handle case where retrieved docs had no usable text
    if "No relevant information found" in context_str or "No relevant text found" in context_str:
         return "I found some related entries, but couldn't extract specific details to answer your question."

    # 3. Generate the response using LLM
    response = generate_response(query, context_str)

    return response


# --- Example Usage (for testing) ---
if __name__ == "__main__":
    print("Testing RAG Pipeline...")

    # Load models and data when script is run directly
    try:
        load_models_and_data()
    except Exception as e:
        print(f"Failed to initialize RAG pipeline: {e}")
        exit()

    # Example Queries:
    test_queries = [
        "What is the price of Paneer Tikka Masala at Spice Villa?",
        "Does Green Leaf Café have any vegan options?",
        "Compare the menus of Spice Villa and Urban Diner for appetizers.", # More complex
        "What are the operating hours for Ocean's Catch?",
        "Are there any gluten-free desserts at Sweet Delight?", # Might require checking features/descriptions
        "Which restaurant serves seafood?",
        "Tell me about the burgers available.",
        "What's the cheapest item on the menu at Spice Villa?", # Requires reasoning
        "Does any restaurant have outdoor seating?", # Check features
        "Gibberish question that shouldn't match anything?" # Out-of-scope test
    ]

    for q in test_queries:
        print("\n" + "="*40)
        print(f"Query: {q}")
        answer = get_rag_response(q)
        print(f"Answer: {answer}")
        print("="*40)