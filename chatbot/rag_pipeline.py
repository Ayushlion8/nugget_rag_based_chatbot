# ZOMATO_RAG_CHATBOT/chatbot/rag_pipeline.py

import os
import pickle
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
import torch
import requests
import json
import random
import re
from dotenv import load_dotenv
from typing import List, Dict, Union, Tuple
import traceback # Import traceback
import time # Import time for potential delays

# --- Configuration ---
# Load .env file from the project root (assuming rag_pipeline.py is in chatbot/)
PROJECT_ROOT_ENV = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
dotenv_path = os.path.join(PROJECT_ROOT_ENV, '.env')
if os.path.exists(dotenv_path):
    print(f"Loading environment variables from: {dotenv_path}")
    load_dotenv(dotenv_path=dotenv_path)
else:
    print(f"Warning: .env file not found at {dotenv_path}. API calls may fail.")
    load_dotenv() # Attempt to load from default location if not found

# BASE_DIR calculation for data paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROCESSED_DATA_DIR = os.path.join(BASE_DIR, "data", "processed_data") # Check if this path is correct
INDEX_PATH = os.path.join(PROCESSED_DATA_DIR, "lucknow_with_menus_restaurant_index.faiss")
MAPPING_PATH = os.path.join(PROCESSED_DATA_DIR, "lucknow_with_menus_index_to_doc_mapping.pkl")

# Embedding Model
EMBEDDING_MODEL_NAME = 'all-MiniLM-L6-v2'

# --- Hugging Face API Configuration ---
# Load MULTIPLE tokens
HF_API_TOKENS_STR = os.getenv("HF_API_TOKENS")
API_TOKEN_LIST = []
if HF_API_TOKENS_STR:
    API_TOKEN_LIST = [token.strip() for token in HF_API_TOKENS_STR.split(',') if token.strip()]

current_token_index = 0 # Start with the first token in the list

# --- SELECT YOUR LANGUAGE MODEL HERE ---
# HF_MODEL_ID = "meta-llama/Meta-Llama-3.1-8B-Instruct"
HF_MODEL_ID = "mistralai/Mistral-7B-Instruct-v0.3"
# HF_MODEL_ID = "microsoft/Phi-3-mini-4k-instruct"
# HF_MODEL_ID = "google/gemma-2b-it"

# --- End Model Selection ---

HF_API_URL = f"https://api-inference.huggingface.co/models/{HF_MODEL_ID}"

# Retrieval settings
NUM_DOCS_TO_RETRIEVE = 5

# --- Global Variables ---
embedding_model = None
index = None
doc_mapping = None
device = "cuda" if torch.cuda.is_available() else "cpu"
api_tokens_available = bool(API_TOKEN_LIST) # Track if any tokens were loaded

# --- Simple Greetings Set ---
GREETINGS = {"hi", "hello", "hey", "yo", "greetings", "good morning", "good afternoon", "good evening"}

# --- Initialization Functions ---
def check_api_tokens():
    """Checks if the HF API tokens are available."""
    global api_tokens_available, HF_MODEL_ID
    if not API_TOKEN_LIST:
        print("\n" + "="*30 + " HF API TOKENS MISSING " + "="*30)
        print("Error: Environment variable 'HF_API_TOKENS' not found or empty in .env file.")
        print("       Expected format: HF_API_TOKENS=token1,token2,token3")
        print("       Please get tokens from https://huggingface.co/settings/tokens")
        print("       Make sure the .env file is in the project root.")
        print("="*88 + "\n")
        api_tokens_available = False
        return False
    else:
        print(f"{len(API_TOKEN_LIST)} Hugging Face API token(s) found. Configured to use model: {HF_MODEL_ID}")
        if any(term in HF_MODEL_ID for term in ["meta-llama", "gemma", "mistral"]):
             print(f"IMPORTANT: Ensure you have accepted the terms for '{HF_MODEL_ID}' on Hugging Face Hub if required.")
        api_tokens_available = True
        return True

def load_models_and_data():
    """Loads embedding model, FAISS index, mapping, and checks HF API tokens."""
    global embedding_model, index, doc_mapping, device, api_tokens_available
    # Avoid reloading if components are already loaded
    if embedding_model is not None and index is not None and doc_mapping is not None:
        if not api_tokens_available:
             print("Re-checking Hugging Face API tokens...")
             check_api_tokens()
        return True

    print("\n--- Loading RAG Pipeline Components ---")
    components_loaded = True
    check_api_tokens() # Initial check

    # Load FAISS Index
    try:
        if not os.path.exists(INDEX_PATH):
            raise FileNotFoundError(f"FAISS index not found at specified path: {INDEX_PATH}")
        print(f"Loading FAISS index from {INDEX_PATH}...")
        index = faiss.read_index(INDEX_PATH)
        print(f"FAISS index loaded: {index.ntotal} vectors.")
    except FileNotFoundError as fnf_err:
        print(f"CRITICAL ERROR: {fnf_err}")
        print(f"Looked in: {os.path.abspath(PROCESSED_DATA_DIR)}")
        print("Please ensure the index file exists and the path is correct. Did you run the data processing/indexing script?")
        components_loaded = False; index = None
    except Exception as e:
        print(f"Error loading FAISS index: {e}"); components_loaded = False; index = None
        traceback.print_exc()

    # Load Document Mapping
    try:
        if not os.path.exists(MAPPING_PATH):
            raise FileNotFoundError(f"Mapping file not found at specified path: {MAPPING_PATH}")
        print(f"Loading document mapping from {MAPPING_PATH}...")
        with open(MAPPING_PATH, 'rb') as f:
            doc_mapping = pickle.load(f)
        if not isinstance(doc_mapping, list) or not doc_mapping:
            raise ValueError("Document mapping file is invalid, empty, or not a list.")
        print(f"Document mapping loaded: {len(doc_mapping)} entries.")
    except FileNotFoundError as fnf_err:
        print(f"CRITICAL ERROR: {fnf_err}")
        print(f"Looked in: {os.path.abspath(PROCESSED_DATA_DIR)}")
        print("Please ensure the mapping file exists and the path is correct. Did you run the data processing/indexing script?")
        components_loaded = False; doc_mapping = None
    except Exception as e:
        print(f"Error loading document mapping: {e}"); components_loaded = False; doc_mapping = None
        traceback.print_exc()

    # Validate Index and Mapping match (if both attempted to load)
    if index is not None and doc_mapping is not None and index.ntotal != len(doc_mapping):
        print(f"CRITICAL ERROR: FAISS index size ({index.ntotal}) does not match mapping size ({len(doc_mapping)}).")
        print("Data inconsistency detected. Please regenerate the index and mapping files together.")
        components_loaded = False; index = doc_mapping = None

    # Load Embedding Model
    try:
        print(f"Loading embedding model: {EMBEDDING_MODEL_NAME}...")
        embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME, device=device)
        _ = embedding_model.encode(["test sentence"])
        print(f"Embedding model loaded successfully onto {device}.")
    except Exception as e:
        print(f"Error loading embedding model '{EMBEDDING_MODEL_NAME}': {e}")
        print("Ensure the model name is correct and dependencies are installed.")
        components_loaded = False; embedding_model = None
        traceback.print_exc()

    # Final status check
    if components_loaded and index is not None and doc_mapping is not None and embedding_model is not None:
        print("--- RAG Components Loaded Successfully ---")
        if not api_tokens_available:
             print("--- WARNING: HF API Tokens Missing/Invalid - LLM Generation will fail ---")
        return True
    else:
        print("\n--- CRITICAL ERROR: One or more RAG Pipeline Components Failed to Load ---")
        print("--- Chatbot functionality will be severely limited or non-functional ---")
        embedding_model = index = doc_mapping = None
        return False


# --- Core RAG Functions (retrieve_relevant_documents, format_context, format_history remain the same) ---
def retrieve_relevant_documents(query: str, k: int = NUM_DOCS_TO_RETRIEVE) -> list[dict]:
    """Retrieves the top-k most relevant document metadata based on the query."""
    if embedding_model is None or index is None or doc_mapping is None:
        print("Error: Retrieval components not loaded. Cannot retrieve.")
        return [] # Return empty list immediately

    if not query or not query.strip():
        print("Warning: Empty query received for retrieval.")
        return []
    # print(f"Retrieving documents for query: '{query}'") # Reduced verbosity
    try:
        query_embedding = embedding_model.encode([query.strip()], convert_to_numpy=True)
    except Exception as e:
        print(f"Error encoding query: {e}")
        return []

    try:
        distances, indices = index.search(query_embedding.astype(np.float32), k)
    except Exception as e:
        print(f"Error during FAISS search: {e}")
        return []

    retrieved_docs_metadata = []
    seen_doc_indices = set()
    if indices.size > 0:
        valid_indices = indices[0][(indices[0] >= 0) & (indices[0] < len(doc_mapping))]
        for doc_index in valid_indices:
            if doc_index in seen_doc_indices:
                continue
            seen_doc_indices.add(doc_index)
            try:
                metadata = doc_mapping[doc_index]
                if isinstance(metadata, dict) and "text_chunk" in metadata:
                    retrieved_docs_metadata.append(metadata)
                else:
                    print(f"Warning: Invalid metadata format at index {doc_index}. Skipping.")
            except IndexError:
                 print(f"Warning: Index {doc_index} out of bounds for mapping list (size {len(doc_mapping)}). Skipping.")

    print(f"Retrieved {len(retrieved_docs_metadata)} unique, valid documents.")
    return retrieved_docs_metadata

def format_context(retrieved_docs: list[dict]) -> str:
    """Formats the retrieved document text chunks into a simpler context string for the LLM."""
    if not retrieved_docs:
        return "No specific context was found for this query in the restaurant data."

    context_parts = []
    unique_chunks = set()
    for doc in retrieved_docs:
        chunk_text = doc.get("text_chunk", "").strip()
        if chunk_text and chunk_text not in unique_chunks:
            source_info = []
            resto_name = doc.get("restaurant_name")
            item_name = doc.get("item_name")
            doc_type = doc.get("doc_type")

            if resto_name: source_info.append(f"Restaurant: {resto_name}")
            if doc_type == "menu_item" and item_name: source_info.append(f"Item: {item_name}")
            elif doc_type == "general_info": source_info.append("Type: General Info")

            source_str = f"[Source: {', '.join(source_info)}]" if source_info else "[Source: Unknown]"
            context_parts.append(f"{source_str}\n{chunk_text}")
            unique_chunks.add(chunk_text)

    if not context_parts:
        return "Found related entries, but could not extract usable text."

    full_context = "\n\n---\n\n".join(context_parts)
    return f"=== Relevant Information Found ===\n{full_context}\n=================================="

# format_history_for_prompt remains the same
def format_history_for_prompt(history: List[Dict[str, str]], max_turns=3) -> str:
    """Formats the last few turns of conversation history for the prompt."""
    if not history: return "No previous conversation."
    formatted_history = ""
    start_index = max(0, len(history) - max_turns * 2)
    relevant_history = history[start_index:]
    for msg in relevant_history:
        role = msg.get("role", "unknown").capitalize()
        content = msg.get("content", "").strip()
        if role in ["User", "Assistant"] and content:
             formatted_history += f"{role}: {content}\n"
    return formatted_history.strip() if formatted_history else "No recent conversation available."


# --- LLM Generation Function (with Token Switching) ---
def generate_response_llm(query: str, context: str, history: List[Dict[str, str]]) -> str:
    """Generates a response using the configured LLM via Hugging Face Inference API, with token switching on rate limits."""
    global api_tokens_available, HF_MODEL_ID, HF_API_URL, API_TOKEN_LIST, current_token_index

    if not api_tokens_available:
        return "My apologies, the connection to the language model API is not configured. No API tokens found."

    # --- System Prompt (same as before) ---
    system_prompt = f"""
You are Zomato Genie 🧞, a factual AI assistant providing information about restaurants.
Your ONLY task is to answer the user's CURRENT question based strictly on the “Relevant Information Found” section.

🔹 CRITICAL Instructions:
  1. **Answer ONLY the CURRENT Question**
     • List exactly what's asked—no extras.
     • E.g., for “vegetarian main courses,” show only items tagged both “vegetarian” + “main course.”
  2. **Strict Context Adherence**
     • **If info found:** Quote names, prices, hours, features exactly as in context.
     • **If info NOT found:** Reply: “I don't have that information in the provided context.”
     • Treat `[Address]`, `N/A`, etc. as missing data.
     • **When comparing a specific category** (e.g. “curries at A vs. B”):
        Filter only those items whose `category` matches.
        If one restaurant has none, say:
         “🍴 [That Restaurant] has no [category] items in the provided context.”
  3. **Conciseness**
     • Short, direct sentences—no intros.
  4. **Bullet Formatting**
     • When listing multiple items, start each line with 🍴
       🍴 Item Name: detail
  5. **Capabilities**
     • Only supply info—no ordering, reservations, or suggestions.
  6. **Ambiguity Handling**
     • If question is vague, ask:
       “What specifically would you like—menu, hours, or features?”
  7. **Tone & Emoji**
     • Helpful, neutral.
     • Use at most **one** emoji (🍽️ or 🤔) in your entire response.

— Now answer the user's question. 🍽️
"""
    # --- End System Prompt ---

    # --- Construct Prompt Messages (same as before) ---
    messages = []
    messages.append({"role": "system", "content": system_prompt})
    history_turns_to_include = 2
    start_index = max(0, len(history) - history_turns_to_include * 2)
    for msg in history[start_index:]:
        role = msg.get("role")
        content = msg.get("content", "").strip()
        if role in ["user", "assistant"] and content:
            messages.append({"role": role, "content": content})
    messages.append({"role": "system", "content": f"Relevant Information Found:\n{context}"})
    messages.append({"role": "user", "content": query})

    prompt_string = "<|begin_of_text|>"
    for msg in messages:
        role = msg['role']
        content = msg['content']
        prompt_string += f"<|start_header_id|>{role}<|end_header_id|>\n\n{content}<|eot_id|>"
    prompt_string += "<|start_header_id|>assistant<|end_header_id|>\n\n"
    # --- End Prompt Construction ---

    # --- Parameters (with updated stop sequences) ---
    parameters = {
        "max_new_tokens": 400,
        "temperature": 0.3,
        "top_p": 0.9,
        "top_k": None,
        "repetition_penalty": 1.1,
        "do_sample": True if 0.3 > 0 else False,
        "return_full_text": False,
        "stop_sequences": [
            "<|eot_id|>",
            "<|end_of_text|>",
            "<|start_header_id|>system<|end_header_id|>",
            "<|start_header_id|>user<|end_header_id|>", # <<< ADDED THIS STOP SEQUENCE
            "\n\nUser:",
            "\n\nSystem:",
            "User:",
            "Assistant:"
        ]
    }
    options = {"wait_for_model": True, "use_cache": False}

    # --- **Retry Loop for Token Switching** ---
    num_tokens = len(API_TOKEN_LIST)
    initial_token_index = current_token_index # Remember where we started for this request
    attempt_count = 0

    while attempt_count < num_tokens:
        active_token = API_TOKEN_LIST[current_token_index]
        print(f"Attempting API call with token index {current_token_index}...")

        headers = {"Authorization": f"Bearer {active_token}", "Content-Type": "application/json"}
        payload = {"inputs": prompt_string, "parameters": parameters, "options": options}

        try:
            response = requests.post(HF_API_URL, headers=headers, json=payload, timeout=90)
            response.raise_for_status() # Raise HTTPError for bad responses (4xx or 5xx)
            response_data = response.json()

            generated_text = ""
            # Handle response variations
            if isinstance(response_data, list) and len(response_data) > 0 and "generated_text" in response_data[0]:
                 generated_text = response_data[0].get("generated_text", "").strip()
            elif isinstance(response_data, dict) and "generated_text" in response_data:
                 generated_text = response_data.get("generated_text", "").strip()
            else:
                 print(f"Warning: Unexpected response format from HF API: {response_data}")
                 # Move to next token on unexpected format, maybe transient?
                 print("Switching token due to unexpected response format.")
                 current_token_index = (current_token_index + 1) % num_tokens
                 attempt_count += 1
                 if current_token_index == initial_token_index: # Prevent infinite loop if all tokens give bad format
                     print("Tried all tokens, all gave unexpected format.")
                     return "Sorry, I received an unexpected response structure from the language model using all available tokens."
                 continue # Try next token

            print(f"Response received successfully using token index {current_token_index}.")

            # --- Post-processing (with added user tag removal) ---
            tags_to_remove = [
                # Note: Keeping user tag here for redundancy, but stop_sequence should handle most cases
                "<|start_header_id|>user<|end_header_id|>",
                "<|start_header_id|>assistant<|end_header_id|>",
                "<|start_header_id|>system<|end_header_id|>",
                "<|eot_id|>",
                "<|end_of_text|>"
            ]
            for tag in tags_to_remove:
                generated_text = generated_text.replace(tag, "")

            # Clean up roles and common lead-ins
            generated_text = re.sub(r'^\s*(User|Assistant|System):\s*', '', generated_text).strip()
            generated_text = re.sub(r'^(Okay|Alright|Sure|Certainly|Here is|Here\'s|Based on the (provided )?context|According to the information provided|I found the following).*?:\s*', '', generated_text, flags=re.IGNORECASE | re.DOTALL).strip()
            generated_text = re.sub(r'\s*Please note that this information.*$', '', generated_text, flags=re.IGNORECASE).strip()

            # <<< ADDED Specific cleanup for any remaining user tag just in case >>>
            generated_text = generated_text.replace("<|start_header_id|>user<|end_header_id|>", "").strip()
            # --- End Post-processing ---


            if not generated_text or len(generated_text) < 5:
                 print(f"Warning: Received potentially empty/invalid response after cleaning: '{generated_text[:100]}...'")
                 # Treat empty response as potentially transient, try next token? Or return error? Let's try next token once.
                 print("Switching token due to empty/short response.")
                 current_token_index = (current_token_index + 1) % num_tokens
                 attempt_count += 1
                 if current_token_index == initial_token_index:
                     print("Tried all tokens, all gave empty/short response.")
                     return "I seem to be having trouble formulating a specific response right now using all available tokens."
                 continue # Try next token
            else:
                # ** SUCCESS! Return the response **
                return generated_text.strip()

        # --- **Specific Error Handling within the Loop** ---
        except requests.exceptions.HTTPError as e:
            error_status_code = e.response.status_code
            error_text = e.response.text[:500]
            print(f"HTTP Error {error_status_code} with token index {current_token_index}: {error_text}...")

            if error_status_code == 429: # Rate Limit Hit
                print(f"Rate limit hit on token index {current_token_index}. Switching token...")
                current_token_index = (current_token_index + 1) % num_tokens
                attempt_count += 1
                # Check if we've tried all tokens
                if attempt_count >= num_tokens or current_token_index == initial_token_index:
                    print("Tried all available tokens, all are rate-limited.")
                    return "Looks like the API is very busy right now (all tokens rate-limited). Please try again in a little while. 🤔"
                # Optional: Add a small delay before retrying with the next token
                time.sleep(0.5)
                continue # Continue the loop to try the next token

            # --- Handle other HTTP errors (Auth, Access Denied, Server errors) ---
            elif error_status_code == 401:
                return f"Authentication Error (401) with token index {current_token_index}. Please check if the token is valid/expired."
            elif error_status_code == 403:
                 if "gated repo" in error_text.lower() or "access is required" in error_text.lower():
                      return f"Access Denied (403) for token {current_token_index}: Ensure access is granted for '{HF_MODEL_ID}' on Hugging Face Hub."
                 else:
                     return f"Access Denied (403) for token {current_token_index}: Issue with permissions for '{HF_MODEL_ID}' or the token."
            elif error_status_code == 422:
                 return f"Invalid Request (422) - possible prompt issue. Details: {error_text}"
            elif str(error_status_code) == '503':
                 if "model is currently loading" in error_text.lower():
                     return f"The language model '{HF_MODEL_ID}' is currently loading (503). Please try again in a minute!"
                 else:
                     return f"The API service for '{HF_MODEL_ID}' seems temporarily unavailable (503). Please try again later."
            elif error_status_code >= 500:
                 return f"API service issue ({error_status_code}). Might be worth trying again later."
            else:
                 return f"An unexpected API error occurred ({error_status_code}). Details: {error_text}"

        except requests.exceptions.Timeout:
            print(f"Error: Request timed out with token index {current_token_index}.")
            return f"Apologies! The request timed out. It might be busy. Maybe try again in a moment? 🤔"

        except requests.exceptions.RequestException as e:
            print(f"Network Error communicating with HF API using token index {current_token_index}: {e}")
            return "I'm having trouble connecting to the Hugging Face API. Please check your internet connection. 🌐"

        except json.JSONDecodeError as json_err:
            print(f"Error decoding JSON response from HF API with token index {current_token_index}: {json_err}")
            print("Switching token due to JSON decode error.")
            current_token_index = (current_token_index + 1) % num_tokens
            attempt_count += 1
            if current_token_index == initial_token_index:
                 print("Tried all tokens, all gave JSON decode errors.")
                 return "Sorry, I received an invalid response structure from the language model using all available tokens."
            continue # Try next token

        except Exception as e:
            print(f"An unexpected error occurred during LLM generation with token index {current_token_index}: {e}")
            traceback.print_exc()
            return "Oops! An unexpected glitch happened while I was thinking. Let's try that again, shall we? 😊"

    # --- End of Retry Loop ---
    print("Exhausted all API tokens after encountering issues.")
    return "I'm currently unable to connect to the language model API after trying all available options. Please try again later. 🤔"


# --- Main Pipeline Function (Modified for clarity) ---
def get_rag_response(query: str, history: List[Dict[str, str]]) -> str:
    """
    Main RAG pipeline: Handles greetings, loads components, retrieves, formats context, generates response.
    """
    normalized_query = query.lower().strip().rstrip('?.!')
    if normalized_query in GREETINGS:
        print("Detected greeting.")
        return random.choice([
            "Hello! Zomato Genie here 🧞. How can I help you with restaurant information today?",
            "Hi there! Ask me about menus, hours, or features of the restaurants in my database. 🍽️",
            "Hey! What restaurant details are you looking for? 😊",
        ])

    # 1. Load components if needed
    if not load_models_and_data():
         return "Error: Failed to load essential chatbot components. Please check logs. Ensure index/mapping files are in data/processed_data."

    # 2. Check API token status specifically
    if not api_tokens_available:
         return "Error: Cannot connect to Hugging Face API. No API tokens found in HF_API_TOKENS environment variable."

    # 3. Retrieve documents
    try:
        retrieved_docs = retrieve_relevant_documents(query)
    except Exception as retrieval_err:
         print(f"Error during document retrieval: {retrieval_err}")
         traceback.print_exc()
         return "Sorry, I encountered an issue while searching my knowledge base. 🤔"

    # 4. Format context
    context_str = format_context(retrieved_docs)
    print(f"Formatted Context Length: {len(context_str)} chars")

    # 5. Generate response (LLM call with retry logic is inside this function)
    response = generate_response_llm(query, context_str, history)

    return response


# --- Example Usage (Standalone Test - Needs .env in project root) ---
if __name__ == "__main__":
    print(f"\n--- Testing RAG Pipeline (No History Simulation) ---")
    # Standalone execution assumes .env is in PROJECT_ROOT_ENV defined earlier
    print(f"--- Using LLM: {HF_MODEL_ID} ---")
    print(f"--- Expecting .env at: {PROJECT_ROOT_ENV} ---")
    print(f"--- Expecting data at: {PROCESSED_DATA_DIR} ---")

    if not load_models_and_data():
        print("\nExiting due to loading errors.")
        exit(1)

    # Define some test queries
    test_queries = [
        "hello there",
        "What vegetarian main courses does Green Leaf Café have?",
        "Does Green Leaf Café have a Greek Salad?",
        "What is the price of the Greek Salad at Green Leaf Café?",
        "Tell me about the spice level of the Vegan Burger at Green Leaf Café.",
        "Does the vegan burger at Green Leaf Café contain soy?",
        "What are the opening hours for Spice Villa?",
        "Does Spice Villa accept credit cards?",
        "Compare the price of appetizers at Spice Villa and Green Leaf Café.",
        "How can I order from Ocean's Catch?"
    ]
    dummy_history = []

    for q in test_queries:
        print("\n" + "="*60)
        print(f"User Query: {q}")
        answer = get_rag_response(q, dummy_history)
        print(f"\nZomato Genie Response:\n{answer}")
        print("="*60)

    print("\n--- Standalone Testing Complete ---")