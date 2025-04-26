# # ZOMATO_RAG_CHATBOT/nugget_rag_based_chatbot/chatbot/rag_pipeline.py

# import os
# import pickle
# import faiss
# import numpy as np
# from sentence_transformers import SentenceTransformer
# from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
# import torch # Ensure PyTorch is installed

# # --- Configuration ---
# BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) # nugget_rag_based_chatbot directory
# PROCESSED_DATA_DIR = os.path.join(BASE_DIR, "data", "processed_data")
# INDEX_PATH = os.path.join(PROCESSED_DATA_DIR, "restaurant_index.faiss")
# MAPPING_PATH = os.path.join(PROCESSED_DATA_DIR, "index_to_doc_mapping.pkl")

# # Models - Use the same embedding model used for indexing
# EMBEDDING_MODEL_NAME = 'all-MiniLM-L6-v2'
# # Choose a suitable generative model from Hugging Face Hub (free tier)
# # Flan-T5 is a good choice for instruction following / Q&A
# LLM_NAME = "google/flan-t5-base" # Options: "google/flan-t5-small", "google/flan-t5-base", "google/flan-t5-large" (larger needs more resources)

# # Retrieval settings
# NUM_DOCS_TO_RETRIEVE = 5 # How many relevant documents to fetch

# # --- Global Variables (Load models once) ---
# embedding_model = None
# index = None
# doc_mapping = None
# llm_tokenizer = None
# llm_model = None
# device = "cuda" if torch.cuda.is_available() else "cpu" # Use GPU if available

# # --- Initialization Function ---

# def load_models_and_data():
#     """Loads all necessary models and data artifacts."""
#     global embedding_model, index, doc_mapping, llm_tokenizer, llm_model, device

#     print("--- Loading RAG Pipeline Components ---")

#     # 1. Load FAISS Index
#     if not os.path.exists(INDEX_PATH):
#         raise FileNotFoundError(f"FAISS index file not found at {INDEX_PATH}. Please run knowledge_base/indexer.py first.")
#     print(f"Loading FAISS index from {INDEX_PATH}...")
#     index = faiss.read_index(INDEX_PATH)
#     print(f"FAISS index loaded. Contains {index.ntotal} vectors.")

#     # 2. Load Index-to-Document Mapping
#     if not os.path.exists(MAPPING_PATH):
#         raise FileNotFoundError(f"Document mapping file not found at {MAPPING_PATH}. Please run knowledge_base/indexer.py first.")
#     print(f"Loading document mapping from {MAPPING_PATH}...")
#     with open(MAPPING_PATH, 'rb') as f:
#         doc_mapping = pickle.load(f)
#     if not isinstance(doc_mapping, list) or not doc_mapping:
#         raise ValueError("Document mapping is empty or invalid.")
#     print(f"Document mapping loaded. Contains {len(doc_mapping)} entries.")

#     # Verify index and mapping length match
#     if index.ntotal != len(doc_mapping):
#          print(f"Warning: FAISS index size ({index.ntotal}) does not match mapping size ({len(doc_mapping)}). There might be an inconsistency.")
#          # You might want to raise an error here depending on strictness

#     # 3. Load Embedding Model (Sentence Transformer)
#     print(f"Loading embedding model: {EMBEDDING_MODEL_NAME}...")
#     # Specify device for SentenceTransformer
#     embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME, device=device)
#     print("Embedding model loaded.")

#     # 4. Load Generative LLM and Tokenizer (from Hugging Face)
#     print(f"Loading LLM and Tokenizer: {LLM_NAME}...")
#     try:
#         llm_tokenizer = AutoTokenizer.from_pretrained(LLM_NAME)
#         # Load model onto the correct device (GPU or CPU)
#         llm_model = AutoModelForSeq2SeqLM.from_pretrained(LLM_NAME).to(device)
#         llm_model.eval() # Set model to evaluation mode
#         print(f"LLM and Tokenizer loaded successfully onto {device}.")
#     except Exception as e:
#         print(f"Error loading LLM/Tokenizer: {e}")
#         print("Please ensure you have a working internet connection and the model name is correct.")
#         raise

#     print("--- RAG Pipeline Components Loaded Successfully ---")

# # --- Core RAG Functions ---

# def retrieve_relevant_documents(query: str, k: int = NUM_DOCS_TO_RETRIEVE) -> list[dict]:
#     """Retrieves the top-k most relevant document metadata based on the query."""
#     if embedding_model is None or index is None or doc_mapping is None:
#         raise RuntimeError("Models or data not loaded. Call load_models_and_data() first.")
#     if not query:
#         return []

#     print(f"Retrieving documents for query: '{query}'")
#     # 1. Embed the query
#     query_embedding = embedding_model.encode([query], convert_to_numpy=True, device=device)

#     # 2. Search the FAISS index
#     # FAISS returns distances and indices. Indices correspond to the order in doc_mapping.
#     try:
#         distances, indices = index.search(query_embedding.astype(np.float32), k)
#     except Exception as e:
#         print(f"Error during FAISS search: {e}")
#         return []

#     # 3. Get the corresponding documents/metadata from the mapping
#     retrieved_docs_metadata = []
#     if indices.size > 0:
#         for i in range(indices.shape[1]): # Iterate through the top k results
#             doc_index = indices[0, i]
#             # Ensure the index is valid
#             if 0 <= doc_index < len(doc_mapping):
#                 retrieved_docs_metadata.append(doc_mapping[doc_index])
#             else:
#                 print(f"Warning: Retrieved invalid index {doc_index} from FAISS.")

#     print(f"Retrieved {len(retrieved_docs_metadata)} documents.")
#     return retrieved_docs_metadata


# def format_context(retrieved_docs: list[dict]) -> str:
#     """Formats the retrieved document text chunks into a single context string."""
#     if not retrieved_docs:
#         return "No relevant information found in the knowledge base."

#     context = ""
#     for i, doc in enumerate(retrieved_docs):
#         # Ensure 'text_chunk' key exists and is not empty
#         chunk_text = doc.get("text_chunk", "").strip()
#         if chunk_text:
#             context += f"--- Document {i+1} ---\n"
#             context += chunk_text + "\n\n" # Add separation between docs

#     return context.strip() if context else "No relevant text found in retrieved documents."


# def generate_response(query: str, context: str) -> str:
#     """Generates a response using the LLM based on the query and retrieved context."""
#     if llm_tokenizer is None or llm_model is None:
#         raise RuntimeError("LLM components not loaded. Call load_models_and_data() first.")

#     # Construct the prompt for the LLM
#     # This prompt guides the LLM to answer based *only* on the provided context.
#     prompt = f"""Answer the following question based *only* on the provided context. If the context does not contain the answer, say "I cannot answer this question based on the available information."

# Context:
# {context}

# Question: {query}

# Answer:"""

#     print("Generating response...")
#     try:
#         # Tokenize the prompt and generate response
#         inputs = llm_tokenizer(prompt, return_tensors="pt", max_length=1024, truncation=True).to(device)

#         # Generate output tokens - adjust parameters as needed
#         # max_length controls the maximum length of the generated response
#         # num_beams increases quality but slows down generation
#         with torch.no_grad(): # Disable gradient calculations for inference
#              outputs = llm_model.generate(
#                 **inputs,
#                 max_length=256, # Adjust max response length
#                 num_beams=5,    # Use beam search for potentially better quality
#                 early_stopping=True,
#                 temperature=0.7, # Control randomness (lower = more deterministic)
#                 no_repeat_ngram_size=2 # Prevent repeating phrases
#             )

#         # Decode the generated tokens into text
#         response_text = llm_tokenizer.decode(outputs[0], skip_special_tokens=True)

#         print("Response generated.")
#         return response_text.strip()

#     except Exception as e:
#         print(f"Error during LLM generation: {e}")
#         return "Sorry, I encountered an error while generating the response."


# # --- Main Pipeline Function ---

# def get_rag_response(query: str) -> str:
#     """
#     The main RAG pipeline function.
#     Takes a user query, retrieves relevant documents, and generates a response.
#     """
#     # Ensure models are loaded (call if first time)
#     if embedding_model is None:
#          try:
#               load_models_and_data()
#          except Exception as load_err:
#               return f"Error loading RAG components: {load_err}"

#     # 1. Retrieve relevant documents
#     retrieved_docs = retrieve_relevant_documents(query)

#     # Handle case where no documents are found
#     if not retrieved_docs:
#         return "I couldn't find any information related to your query in the restaurant database."

#     # 2. Format the context
#     context_str = format_context(retrieved_docs)

#     # Handle case where retrieved docs had no usable text
#     if "No relevant information found" in context_str or "No relevant text found" in context_str:
#          return "I found some related entries, but couldn't extract specific details to answer your question."

#     # 3. Generate the response using LLM
#     response = generate_response(query, context_str)

#     return response


# # --- Example Usage (for testing) ---
# if __name__ == "__main__":
#     print("Testing RAG Pipeline...")

#     # Load models and data when script is run directly
#     try:
#         load_models_and_data()
#     except Exception as e:
#         print(f"Failed to initialize RAG pipeline: {e}")
#         exit()

#     # Example Queries:
#     test_queries = [
#         "What is the price of Paneer Tikka Masala at Spice Villa?",
#         "Does Green Leaf Café have any vegan options?",
#         "Compare the menus of Spice Villa and Urban Diner for appetizers.", # More complex
#         "What are the operating hours for Ocean's Catch?",
#         "Are there any gluten-free desserts at Sweet Delight?", # Might require checking features/descriptions
#         "Which restaurant serves seafood?",
#         "Tell me about the burgers available.",
#         "What's the cheapest item on the menu at Spice Villa?", # Requires reasoning
#         "Does any restaurant have outdoor seating?", # Check features
#         "Gibberish question that shouldn't match anything?" # Out-of-scope test
#     ]

#     for q in test_queries:
#         print("\n" + "="*40)
#         print(f"Query: {q}")
#         answer = get_rag_response(q)
#         print(f"Answer: {answer}")
#         print("="*40)















# # ZOMATO_RAG_CHATBOT/nugget_rag_based_chatbot/chatbot/rag_pipeline.py


# import os
# import pickle
# import faiss
# import numpy as np
# from sentence_transformers import SentenceTransformer
# import torch
# import requests # Added for Ollama API calls
# import json # Added for handling API response

# # --- Configuration ---
# BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) # nugget_rag_based_chatbot directory
# PROCESSED_DATA_DIR = os.path.join(BASE_DIR, "data", "processed_data")
# INDEX_PATH = os.path.join(PROCESSED_DATA_DIR, "restaurant_index.faiss")
# MAPPING_PATH = os.path.join(PROCESSED_DATA_DIR, "index_to_doc_mapping.pkl")

# # Embedding Model (Consistent with indexing)
# EMBEDDING_MODEL_NAME = 'all-MiniLM-L6-v2'

# # --- Ollama Configuration ---
# OLLAMA_BASE_URL = "http://localhost:11434" # Use base URL for flexibility
# OLLAMA_API_GENERATE_URL = f"{OLLAMA_BASE_URL}/api/generate"
# OLLAMA_API_TAGS_URL = f"{OLLAMA_BASE_URL}/api/tags" # Correct tags endpoint

# # --- MODIFICATION: Changed model name to mistral ---
# # IMPORTANT: Ensure you have run `ollama pull mistral`
# OLLAMA_MODEL_NAME = "mistral" # Using the mistral model (likely mistral:7b)
# # --- End Modification ---

# # Retrieval settings
# NUM_DOCS_TO_RETRIEVE = 5 # How many relevant documents to fetch (keep at 5 for now)

# # --- Global Variables (Load models once) ---
# embedding_model = None
# index = None
# doc_mapping = None
# device = "cuda" if torch.cuda.is_available() else "cpu" # Use GPU if available for embeddings
# ollama_available = False # Flag to track Ollama status

# # --- Initialization Function ---

# def check_ollama_status():
#     """Checks if the Ollama service is reachable and the specified model is available."""
#     global ollama_available
#     try:
#         # 1. Check base Ollama URL
#         response = requests.get(OLLAMA_BASE_URL, timeout=3)
#         response.raise_for_status() # Raise an exception for bad status codes
#         print(f"Ollama service is reachable at {OLLAMA_BASE_URL}")

#         # 2. Check if the specific model tag is available using GET request
#         print(f"Checking available Ollama models at: {OLLAMA_API_TAGS_URL}")
#         response = requests.get(OLLAMA_API_TAGS_URL, timeout=5) # Changed to GET
#         response.raise_for_status()
#         models_data = response.json()

#         # Ollama versions might return list under 'models' key or directly
#         if isinstance(models_data, dict) and "models" in models_data:
#             models = models_data.get("models", [])
#         elif isinstance(models_data, list): # Handle older format if necessary
#              models = models_data
#         else:
#              print("Warning: Unexpected format received from Ollama /api/tags endpoint.")
#              models = []

#         model_names = [m.get('name', 'unknown:unknown').split(':')[0] for m in models] # Get base model names (e.g., 'mistral' from 'mistral:7b')
#         available_tags = [m.get('name', 'unknown:unknown') for m in models] # Get full tags

#         # Check if the exact tag OR the base name exists
#         requested_base_name = OLLAMA_MODEL_NAME.split(':')[0]
#         if OLLAMA_MODEL_NAME not in available_tags and requested_base_name not in model_names:
#              print(f"Warning: Ollama model '{OLLAMA_MODEL_NAME}' (or base name '{requested_base_name}') not found locally.")
#              print(f"Available models: {', '.join(available_tags) if available_tags else 'None found'}")
#              print(f"Please ensure you have pulled the model using: ollama pull {OLLAMA_MODEL_NAME}")
#              print("Will proceed, but generation will likely fail if model is incorrect.")
#              ollama_available = False # Mark as technically unavailable for the specific model
#              return False # Return False if specific model not found
#         else:
#              # Find the exact tag being used if only base name was provided initially
#              if OLLAMA_MODEL_NAME not in available_tags:
#                   found_tag = next((tag for tag in available_tags if tag.startswith(requested_base_name + ':')), None)
#                   if found_tag:
#                        print(f"Found matching tag '{found_tag}' for base name '{requested_base_name}'. Using '{found_tag}'.")
#                        # Optionally update OLLAMA_MODEL_NAME globally? For now, just confirm availability.
#                   else:
#                        # This case should be unlikely if base name matched but tag didn't
#                         print(f"Could not find a specific tag for base model '{requested_base_name}'.")
#                         ollama_available = False
#                         return False

#              print(f"Ollama model '{OLLAMA_MODEL_NAME}' (or a matching tag) is available locally.")
#              ollama_available = True
#              return True # Return True only if service AND model are confirmed

#     except requests.exceptions.RequestException as e:
#         print(f"Error connecting to Ollama service/API: {e}")
#         print(f"Checked URLs: {OLLAMA_BASE_URL}, {OLLAMA_API_TAGS_URL}")
#         print("Please ensure the Ollama service is running and accessible.")
#         ollama_available = False
#         return False
#     except Exception as e:
#         print(f"An unexpected error occurred while checking Ollama status: {e}")
#         ollama_available = False
#         return False

# def load_models_and_data():
#     """Loads embedding model, FAISS index, mapping, and checks Ollama."""
#     global embedding_model, index, doc_mapping, device, ollama_available

#     # Prevent reloading if already loaded and Ollama status is known
#     if embedding_model is not None and index is not None and doc_mapping is not None:
#         # Re-check Ollama status only if it wasn't available before or status unknown
#         if not ollama_available:
#             print("Re-checking Ollama status...")
#             check_ollama_status()
#             if not ollama_available:
#                  print("Warning: Ollama check failed again.")
#                  # Decide if loading should fail entirely
#                  # return False
#         else:
#             print("Models, data, and Ollama connection already verified.")
#             return True

#     print("--- Loading RAG Pipeline Components ---")
#     components_loaded = True

#     # 1. Check Ollama Status (sets ollama_available flag)
#     if not ollama_available: # Check if status needs verification
#         check_ollama_status()
#         if not ollama_available:
#             print("Warning: Ollama check failed or specified model not found. Generation will not work.")
#             # components_loaded = False # Uncomment this line to make Ollama availability mandatory

#     # 2. Load FAISS Index
#     try:
#         if index is None: # Only load if not already loaded
#             if not os.path.exists(INDEX_PATH):
#                 raise FileNotFoundError(f"FAISS index file not found at {INDEX_PATH}. Please run knowledge_base/indexer.py first.")
#             print(f"Loading FAISS index from {INDEX_PATH}...")
#             index = faiss.read_index(INDEX_PATH)
#             print(f"FAISS index loaded. Contains {index.ntotal} vectors.")
#     except Exception as e:
#         print(f"Error loading FAISS index: {e}")
#         components_loaded = False

#     # 3. Load Index-to-Document Mapping
#     try:
#          if doc_mapping is None: # Only load if not already loaded
#             if not os.path.exists(MAPPING_PATH):
#                 raise FileNotFoundError(f"Document mapping file not found at {MAPPING_PATH}. Please run knowledge_base/indexer.py first.")
#             print(f"Loading document mapping from {MAPPING_PATH}...")
#             with open(MAPPING_PATH, 'rb') as f:
#                 doc_mapping = pickle.load(f)
#             if not isinstance(doc_mapping, list) or not doc_mapping:
#                 raise ValueError("Document mapping is empty or invalid.")
#             print(f"Document mapping loaded. Contains {len(doc_mapping)} entries.")
#     except Exception as e:
#         print(f"Error loading document mapping: {e}")
#         components_loaded = False

#     # Verify index and mapping length match (if both loaded successfully)
#     if components_loaded and index is not None and doc_mapping is not None and index.ntotal != len(doc_mapping):
#          print(f"Error: FAISS index size ({index.ntotal}) does not match mapping size ({len(doc_mapping)}). Aborting.")
#          components_loaded = False # Treat mismatch as critical failure

#     # 4. Load Embedding Model (Sentence Transformer)
#     try:
#         if embedding_model is None: # Only load if not already loaded
#             print(f"Loading embedding model: {EMBEDDING_MODEL_NAME}...")
#             embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME, device=device)
#             print(f"Embedding model loaded successfully onto {device}.")
#     except Exception as e:
#         print(f"Error loading embedding model: {e}")
#         components_loaded = False


#     if components_loaded and ollama_available:
#         print("--- RAG Pipeline Components Loaded and Ollama Verified Successfully ---")
#         return True
#     elif components_loaded and not ollama_available:
#          print("--- RAG Pipeline Components Loaded (But Ollama Verification Failed) ---")
#          return False # Return False if Ollama part failed
#     else:
#          print("--- RAG Pipeline Components Failed to Load ---")
#          # Reset globals if loading failed partially to ensure clean state
#          embedding_model = index = doc_mapping = None
#          ollama_available = False
#          return False


# # --- Core RAG Functions ---

# def retrieve_relevant_documents(query: str, k: int = NUM_DOCS_TO_RETRIEVE) -> list[dict]:
#     """Retrieves the top-k most relevant document metadata based on the query."""
#     if embedding_model is None or index is None or doc_mapping is None:
#         print("Attempting to load models/data for retrieval...")
#         if not load_models_and_data():
#              raise RuntimeError("Failed to load necessary models/data for retrieval.")
#         if embedding_model is None or index is None or doc_mapping is None:
#              raise RuntimeError("Essential models or data failed to load. Cannot retrieve.")

#     if not query:
#         return []

#     print(f"Retrieving documents for query: '{query}'")
#     # 1. Embed the query
#     try:
#         query_embedding = embedding_model.encode([query], convert_to_numpy=True, device=device)
#     except Exception as e:
#         print(f"Error encoding query: {e}")
#         return []

#     # 2. Search the FAISS index
#     try:
#         # Ensure embedding is float32 for FAISS
#         distances, indices = index.search(query_embedding.astype(np.float32), k)
#     except Exception as e:
#         print(f"Error during FAISS search: {e}")
#         return []

#     # 3. Get the corresponding documents/metadata from the mapping
#     retrieved_docs_metadata = []
#     if indices.size > 0:
#         # Ensure indices are within bounds
#         valid_indices = indices[0][(indices[0] >= 0) & (indices[0] < len(doc_mapping))]
#         for doc_index in valid_indices:
#             # Basic check on the retrieved metadata structure
#             if isinstance(doc_mapping[doc_index], dict):
#                  retrieved_docs_metadata.append(doc_mapping[doc_index])
#             else:
#                  print(f"Warning: Invalid metadata format found at index {doc_index}. Skipping.")


#     print(f"Retrieved {len(retrieved_docs_metadata)} documents.")
#     return retrieved_docs_metadata


# def format_context(retrieved_docs: list[dict]) -> str:
#     """Formats the retrieved document text chunks into a single context string."""
#     if not retrieved_docs:
#         return "No relevant information was found in the knowledge base." # Slightly more informative

#     context = ""
#     # Use a set to avoid adding the exact same text chunk multiple times if retrieved
#     unique_chunks = set()
#     added_chunks = 0

#     for i, doc in enumerate(retrieved_docs):
#         chunk_text = doc.get("text_chunk", "").strip()
#         # Add chunk only if it has content and hasn't been added before
#         if chunk_text and chunk_text not in unique_chunks:
#             resto_name = doc.get("restaurant_name", "Unknown Restaurant")
#             item_name = doc.get("item_name", "")
#             doc_type = doc.get("doc_type", "unknown")

#             prefix = f"--- Context Snippet {added_chunks+1}"
#             if doc_type == "menu_item" and item_name:
#                  prefix += f" (From Menu: {resto_name} - Item: {item_name})"
#             elif doc_type == "general_info":
#                  prefix += f" (General Info: {resto_name})"
#             prefix += " ---\n"

#             context += prefix
#             context += chunk_text + "\n\n"
#             unique_chunks.add(chunk_text)
#             added_chunks += 1


#     if not context:
#          return "Found related entries, but could not extract usable text." # Edge case

#     # Add a header to the context block
#     final_context = "=== Knowledge Base Context ===\n" + context.strip() + "\n=============================="
#     return final_context

# def generate_response_with_ollama(query: str, context: str) -> str:
#     """Generates a response using the Ollama API based on the query and context."""
#     global ollama_available
#     if not ollama_available:
#         return "Sorry, the connection to the Ollama language model is not available. Please check if Ollama is running and the correct model is specified."

#     # --- Enhanced Prompt Template ---
#     # (Using the same enhanced prompt from before)
#     prompt = f"""You are a helpful and friendly Zomato restaurant assistant chatbot.
# Your goal is to answer the user's question accurately and conversationally based *only* on the provided "Knowledge Base Context".

# **Instructions:**
# 1.  Analyze the user's "Question" and the "Knowledge Base Context" carefully.

# 2.  Formulate a comprehensive and helpful answer. Avoid single-word responses. Explain your reasoning where appropriate (e.g., when comparing or finding cheapest/most expensive).

# 3.  If the question involves comparison (e.g., comparing menus), list the relevant items for each entity mentioned based on the context, and then provide a summary comparison based *only* on those listed items.

# 4.  If the question asks for the cheapest or most expensive item for a specific restaurant, state the item and its price from the context. If multiple items share the same lowest/highest price, mention them.

# 5.  **Crucially:** If the exact information needed to answer the question is *not* found within the "Knowledge Base Context", you MUST explicitly state that you cannot answer based on the available information. Use a phrase like: "Based on the information I have for [Restaurant Name(s)], I couldn't find specific details about [topic of the question]." or "I don't have enough information in the provided context to answer that specific question about [topic]." Do not invent information or use external knowledge.

# 6. 

# {context}

# User Question: {query}

# Chatbot Answer:"""

#     print(f"Sending request to Ollama model '{OLLAMA_MODEL_NAME}'...")
#     try:
#         payload = {
#             "model": OLLAMA_MODEL_NAME,
#             "prompt": prompt,
#             "stream": False, # Keep it simple for now, get the full response at once
#             "options": {
#                "temperature": 0.5 # Slightly lower temperature for more factual responses
#                # Add other options if needed, e.g., num_ctx for context window size if model requires it
#             }
#         }
#         response = requests.post(OLLAMA_API_GENERATE_URL, json=payload, timeout=180) # Increased timeout slightly to 3 minutes just in case for mistral
#         response.raise_for_status() # Raise HTTPError for bad responses (4xx or 5xx)

#         response_data = response.json()

#         # Check if the response format is as expected
#         if "response" not in response_data:
#              print(f"Warning: 'response' key missing in Ollama response. Data: {response_data}")
#              return "Sorry, I received an unexpected response format from the language model."

#         generated_text = response_data.get("response", "").strip()

#         # Basic check if the model just repeated the prompt or gave an empty answer
#         if not generated_text or generated_text.startswith("User Question:") or len(generated_text) < 5 : # Lowered threshold slightly
#             print(f"Warning: Received potentially empty or invalid response from Ollama: '{generated_text[:100]}...'")
#             return "I seem to have trouble formulating a proper response right now. Could you please try rephrasing your question?"

#         print("Response received from Ollama.")
#         return generated_text

#     except requests.exceptions.Timeout:
#         print(f"Error: Request to Ollama model '{OLLAMA_MODEL_NAME}' timed out after 180 seconds.")
#         return f"Sorry, the request to the '{OLLAMA_MODEL_NAME}' model timed out. It might be taking too long to process this request on your hardware. You could try a smaller model or check your system resources."
#     except requests.exceptions.RequestException as e:
#         print(f"Error communicating with Ollama API at {OLLAMA_API_GENERATE_URL}: {e}")
#         return "Sorry, I'm having trouble connecting to the language model right now. Please ensure Ollama is running."
#     except json.JSONDecodeError:
#         print(f"Error: Could not decode JSON response from Ollama.")
#         print(f"Raw Response Text: {response.text[:500]}...") # Log snippet of raw response
#         return "Sorry, I received an invalid response from the language model."
#     except Exception as e:
#         print(f"An unexpected error occurred during Ollama generation: {e}")
#         return "Sorry, I encountered an unexpected error while generating the response."


# # --- Main Pipeline Function ---

# def get_rag_response(query: str) -> str:
#     """
#     The main RAG pipeline function using Ollama.
#     Takes a user query, retrieves relevant documents, and generates a response.
#     """
#     # 1. Ensure models/data are loaded (load_models_and_data returns True on success)
#     if not load_models_and_data():
#          # Specific error if Ollama was the issue vs other components
#          if not ollama_available:
#               return f"Error: Cannot connect to the Ollama language model '{OLLAMA_MODEL_NAME}'. Please ensure Ollama is running and the model is available locally."
#          else:
#               return "Error: Failed to initialize the chatbot components (Embeddings/Index/Mapping). Please check logs."


#     # 2. Retrieve relevant documents
#     try:
#         retrieved_docs = retrieve_relevant_documents(query)
#     except Exception as retrieval_err:
#          print(f"Error during document retrieval phase: {retrieval_err}")
#          return "Sorry, I encountered an error trying to find relevant information."


#     # Handle case where no documents are found early
#     if not retrieved_docs:
#         # Make this response more conversational
#         return f"Hmm, I searched the database but couldn't find specific information related to your query: '{query}'. Could you try asking in a different way or about one of the restaurants I know?"

#     # 3. Format the context
#     context_str = format_context(retrieved_docs)

#     # Handle case where retrieved docs had no usable text
#     if "No relevant information was found" in context_str or "could not extract usable text" in context_str:
#          # Make this more conversational
#          return "I found some potentially related entries, but I couldn't extract the specific details needed to answer your question confidently. Perhaps try rephrasing?"

#     # 4. Generate the response using Ollama
#     response = generate_response_with_ollama(query, context_str)

#     return response


# # --- Example Usage (for testing) ---
# if __name__ == "__main__":
#     print("Testing RAG Pipeline with Ollama...")

#     # Load models and data when script is run directly
#     if not load_models_and_data():
#         print("Exiting due to loading errors.")
#         exit(1) # Exit with non-zero code on error

#     # Example Queries (keep the challenging ones):
#     test_queries = [
#         "What is the price of Paneer Tikka Masala at Spice Villa?",
#         "Does Green Leaf Café have vegan options mentioned in their features or menu?",
#         "Compare the menus of Spice Villa and Urban Diner for appetizers. List the items and prices if available.", # More specific comparison instruction
#         "What are the operating hours for Ocean's Catch?",
#         "Are there any gluten-free options mentioned for Sweet Delight, either as a feature or in the menu descriptions?", # More specific feature check
#         "Which restaurants specialize in seafood based on their features or menu items?",
#         "Tell me about the Classic Cheeseburger at Urban Diner.",
#         "What's the cheapest menu item listed for Spice Villa based on the provided context?", # More specific reasoning instruction
#         "Does the context mention outdoor seating for any restaurant?", # Specific check for absence
#         "Can you recommend a restaurant for a late-night meal?", # Checks hours/features
#         "Compare spice levels mentioned in the menus of Spice Villa and Ocean's Catch" # Requires specific data points
#     ]

#     for q in test_queries:
#         print("\n" + "="*50)
#         print(f"Query: {q}")
#         answer = get_rag_response(q)
#         print(f"\nChatbot Response:\n{answer}")
#         print("="*50)


















# ZOMATO_RAG_CHATBOT/nugget_rag_based_chatbot/chatbot/rag_pipeline.py

import os
import pickle
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
import torch
import requests
import json
import random # Added to vary phrasing
import re

# --- Configuration ---
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) # nugget_rag_based_chatbot directory
PROCESSED_DATA_DIR = os.path.join(BASE_DIR, "data", "processed_data")
INDEX_PATH = os.path.join(PROCESSED_DATA_DIR, "restaurant_index.faiss")
MAPPING_PATH = os.path.join(PROCESSED_DATA_DIR, "index_to_doc_mapping.pkl")

# Embedding Model (Consistent with indexing)
EMBEDDING_MODEL_NAME = 'all-MiniLM-L6-v2'

# --- Ollama Configuration ---
OLLAMA_BASE_URL = "http://localhost:11434" # Use base URL for flexibility
OLLAMA_API_GENERATE_URL = f"{OLLAMA_BASE_URL}/api/generate"
OLLAMA_API_TAGS_URL = f"{OLLAMA_BASE_URL}/api/tags" # Correct tags endpoint

# --- MODIFICATION: Changed model name to tinyllama ---
# IMPORTANT: Ensure you have run `ollama pull tinyllama`
OLLAMA_MODEL_NAME = "llama3.2:1b" # Using the tinyllama model for speed
# --- End Modification ---

# Retrieval settings
NUM_DOCS_TO_RETRIEVE = 4 # Slightly reduced for tinyllama, less context might be better
CONTEXT_RELEVANCE_THRESHOLD = 0.5 # Optional: Add a threshold for embedding distance (experimental)

# --- Global Variables (Load models once) ---
embedding_model = None
index = None
doc_mapping = None
device = "cuda" if torch.cuda.is_available() else "cpu"
ollama_available = False

# ---Phrasing Variations for Groundedness---
GROUNDED_PHRASES = [
    "Looking at the details I have...",
    "From what I can see about {restaurant}...",
    "My information for {restaurant} shows...",
    "Checking the menu details I have for {restaurant}...",
    "Based on the restaurant's description available to me...",
    "According to the data I have...",
]

# --- Initialization Function --- (check_ollama_status and load_models_and_data remain largely the same, ensure check_ollama_status confirms 'tinyllama')

def check_ollama_status():
    """Checks if the Ollama service is reachable and the specified model is available."""
    global ollama_available
    try:
        response = requests.get(OLLAMA_BASE_URL, timeout=3)
        response.raise_for_status()
        print(f"Ollama service is reachable at {OLLAMA_BASE_URL}")

        print(f"Checking available Ollama models at: {OLLAMA_API_TAGS_URL}")
        response = requests.get(OLLAMA_API_TAGS_URL, timeout=5)
        response.raise_for_status()
        models_data = response.json()

        if isinstance(models_data, dict) and "models" in models_data:
            models = models_data.get("models", [])
        elif isinstance(models_data, list):
             models = models_data
        else:
             print("Warning: Unexpected format received from Ollama /api/tags endpoint.")
             models = []

        available_tags = [m.get('name', 'unknown:unknown') for m in models]

        # Check specifically for the requested model tag
        if OLLAMA_MODEL_NAME not in available_tags:
            # Try checking just the base name (e.g., 'tinyllama' from 'tinyllama:1.1b-chat-v1.0-q4_0')
            base_name_check = OLLAMA_MODEL_NAME.split(':')[0]
            matching_tag = next((tag for tag in available_tags if tag.startswith(base_name_check + ':')), None)

            if matching_tag:
                 print(f"Found matching tag '{matching_tag}' for requested model '{OLLAMA_MODEL_NAME}'. Will use '{matching_tag}'.")
                 # Update the global name to the actual tag found? For simplicity, let's assume the API call handles it if base name is unique enough.
                 ollama_available = True
                 return True
            else:
                 print(f"Warning: Ollama model '{OLLAMA_MODEL_NAME}' not found locally.")
                 print(f"Available models: {', '.join(available_tags) if available_tags else 'None found'}")
                 print(f"Please ensure you have pulled the model using: ollama pull {OLLAMA_MODEL_NAME}")
                 ollama_available = False
                 return False
        else:
             print(f"Ollama model '{OLLAMA_MODEL_NAME}' is available locally.")
             ollama_available = True
             return True

    except requests.exceptions.RequestException as e:
        print(f"Error connecting to Ollama service/API: {e}")
        ollama_available = False
        return False
    except Exception as e:
        print(f"An unexpected error occurred while checking Ollama status: {e}")
        ollama_available = False
        return False

def load_models_and_data():
    """Loads embedding model, FAISS index, mapping, and checks Ollama."""
    global embedding_model, index, doc_mapping, device, ollama_available
    if embedding_model is not None and index is not None and doc_mapping is not None:
        if not ollama_available:
            print("Re-checking Ollama status...")
            check_ollama_status()
            if not ollama_available: print("Warning: Ollama check failed again.")
        else:
            print("Models, data, and Ollama connection already verified.")
            return True # Already loaded and verified

    print("--- Loading RAG Pipeline Components ---")
    components_loaded = True
    if not ollama_available: check_ollama_status() # Initial check if not already done

    try: # Load Index
        if index is None:
            if not os.path.exists(INDEX_PATH): raise FileNotFoundError(f"FAISS index not found: {INDEX_PATH}")
            print(f"Loading FAISS index from {INDEX_PATH}...")
            index = faiss.read_index(INDEX_PATH)
            print(f"FAISS index loaded: {index.ntotal} vectors.")
    except Exception as e: print(f"Error loading FAISS index: {e}"); components_loaded = False

    try: # Load Mapping
        if doc_mapping is None:
            if not os.path.exists(MAPPING_PATH): raise FileNotFoundError(f"Mapping file not found: {MAPPING_PATH}")
            print(f"Loading document mapping from {MAPPING_PATH}...")
            with open(MAPPING_PATH, 'rb') as f: doc_mapping = pickle.load(f)
            if not isinstance(doc_mapping, list) or not doc_mapping: raise ValueError("Mapping is invalid.")
            print(f"Document mapping loaded: {len(doc_mapping)} entries.")
    except Exception as e: print(f"Error loading document mapping: {e}"); components_loaded = False

    if components_loaded and index is not None and doc_mapping is not None and index.ntotal != len(doc_mapping):
        print(f"Error: FAISS index size ({index.ntotal}) != mapping size ({len(doc_mapping)}). Aborting.")
        components_loaded = False

    try: # Load Embedding Model
        if embedding_model is None:
            print(f"Loading embedding model: {EMBEDDING_MODEL_NAME}...")
            embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME, device=device)
            print(f"Embedding model loaded onto {device}.")
    except Exception as e: print(f"Error loading embedding model: {e}"); components_loaded = False

    if components_loaded and ollama_available: print("--- RAG Pipeline Components Loaded and Ollama Verified Successfully ---"); return True
    elif components_loaded: print("--- RAG Pipeline Components Loaded (But Ollama Verification Failed) ---"); return False
    else: print("--- RAG Pipeline Components Failed to Load ---"); embedding_model = index = doc_mapping = None; ollama_available = False; return False


# --- Core RAG Functions ---

def retrieve_relevant_documents(query: str, k: int = NUM_DOCS_TO_RETRIEVE) -> list[dict]:
    """Retrieves the top-k most relevant document metadata based on the query."""
    if embedding_model is None or index is None or doc_mapping is None:
        print("Attempting to load models/data for retrieval...")
        if not load_models_and_data(): raise RuntimeError("Failed to load necessary models/data.")
        if embedding_model is None or index is None or doc_mapping is None: raise RuntimeError("Essential models/data missing.")

    if not query: return []
    print(f"Retrieving documents for query: '{query}'")
    try: query_embedding = embedding_model.encode([query], convert_to_numpy=True, device=device)
    except Exception as e: print(f"Error encoding query: {e}"); return []

    try: distances, indices = index.search(query_embedding.astype(np.float32), k)
    except Exception as e: print(f"Error during FAISS search: {e}"); return []

    retrieved_docs_metadata = []
    if indices.size > 0:
        valid_indices = indices[0][(indices[0] >= 0) & (indices[0] < len(doc_mapping))]
        # Optional: Filter by distance threshold if needed
        # valid_indices = valid_indices[distances[0][(indices[0] >= 0) & (indices[0] < len(doc_mapping))] < CONTEXT_RELEVANCE_THRESHOLD]
        for doc_index in valid_indices:
            if isinstance(doc_mapping[doc_index], dict): retrieved_docs_metadata.append(doc_mapping[doc_index])
            else: print(f"Warning: Invalid metadata format at index {doc_index}. Skipping.")

    print(f"Retrieved {len(retrieved_docs_metadata)} documents.")
    return retrieved_docs_metadata


def format_context(retrieved_docs: list[dict]) -> str:
    """Formats the retrieved document text chunks into a single context string."""
    if not retrieved_docs: return "No relevant information was found in the knowledge base."

    context = ""
    unique_chunks = set()
    added_chunks = 0
    known_restaurants = set(doc.get("restaurant_name", "Unknown") for doc in retrieved_docs if doc.get("restaurant_name"))

    for i, doc in enumerate(retrieved_docs):
        chunk_text = doc.get("text_chunk", "").strip()
        if chunk_text and chunk_text not in unique_chunks:
            resto_name = doc.get("restaurant_name", "Unknown Restaurant")
            item_name = doc.get("item_name", "")
            doc_type = doc.get("doc_type", "unknown")
            prefix = f"--- Context Snippet {added_chunks+1}"
            if doc_type == "menu_item" and item_name: prefix += f" (Menu: {resto_name} - Item: {item_name})"
            elif doc_type == "general_info": prefix += f" (Info: {resto_name})"
            prefix += " ---\n"
            context += prefix + chunk_text + "\n\n"
            unique_chunks.add(chunk_text)
            added_chunks += 1

    if not context: return "Found related entries, but could not extract usable text."

    # Include known restaurant names in the header for the LLM's awareness
    restaurant_list_str = ", ".join(filter(lambda x: x != "Unknown", known_restaurants))
    context_header = "=== Context from Restaurant Menus/Info "
    if restaurant_list_str:
        context_header += f"(Primarily focusing on: {restaurant_list_str})"
    context_header += " ===\n"

    final_context = context_header + context.strip() + "\n==================================="
    return final_context

def get_random_grounded_phrase(restaurant_name=None):
    """Returns a randomly chosen phrase to indicate reliance on context."""
    phrase = random.choice(GROUNDED_PHRASES)
    if restaurant_name and "{restaurant}" in phrase:
        return phrase.format(restaurant=restaurant_name)
    # Remove placeholder if no name provided or phrase doesn't use it
    return phrase.replace(" for {restaurant}", "").replace(" about {restaurant}", "")

def generate_response_with_ollama(query: str, context: str) -> str:
    """Generates a response using the Ollama API based on the query and context."""
    global ollama_available
    if not ollama_available:
        return "My apologies, I'm currently unable to connect to the language model needed to generate a full response. Please ensure Ollama is running correctly."

    # --- NEW Enhanced Prompt Template for TinyLlama ---
    # Focus on clearer instructions, slightly simpler language, encouraging tone.
    prompt = f"""You are 'Zomato Genie', a friendly and enthusiastic food guide chatbot.
Your goal is to answer the user's question in a helpful, engaging, and slightly persuasive way, using *only* the information provided in the "Restaurant Context" below. Make the user excited about the food!

**Your Task:**
1.  Understand the User's "Question".
2.  Carefully read the "Restaurant Context".
3.  Craft a response that directly answers the question.
4.  Sound knowledgeable and passionate about the food described (if details are available). Use inviting language (e.g., "Imagine...", "You might love...", "features a delicious...").
5.  **Accuracy is Key:** Stick strictly to the details in the context. Do NOT add information not present (like ingredients, preparation methods, or general opinions if not mentioned).
6.  **Handling Missing Info:** If the context doesn't have the specific detail needed, clearly and politely say so. Use phrases like: "While I don't see that specific detail for [Restaurant Name] in my notes...", or "Hmm, the menu information I have doesn't mention [specific topic] for [Restaurant Name].", followed by suggesting what information IS available, if relevant. Avoid the robotic "Based on the context..." repeatedly unless absolutely necessary for clarity when info is missing.
7.  **Comparisons:** If asked to compare, list the relevant points for each restaurant from the context first, then give a summary comparison.
8.  **Pricing:** When mentioning price, state it clearly (e.g., "priced at ₹X", "costs ₹X"). For cheapest/most expensive, state the item and price.

**Restaurant Context:**
{context}

**User Question:** {query}

**Zomato Genie Answer:**"""

    print(f"Sending request to Ollama model '{OLLAMA_MODEL_NAME}'...")
    try:
        payload = {
            "model": OLLAMA_MODEL_NAME,
            "prompt": prompt,
            "stream": False,
            "options": {
               "temperature": 0.6, # Slightly more creative but still factual
               "num_predict": 150, # Limit output length for tinyllama if needed
               # "top_k": 40, # Other sampling parameters if needed
               # "top_p": 0.9,
            }
        }
        response = requests.post(OLLAMA_API_GENERATE_URL, json=payload, timeout=60) # Reduced timeout for tinyllama
        response.raise_for_status()

        response_data = response.json()

        if "response" not in response_data:
             print(f"Warning: 'response' key missing in Ollama response. Data: {response_data}")
             return "Oh dear, I seem to have received an unusual response format. Could you try asking again?"

        generated_text = response_data.get("response", "").strip()

        # Filter out potential prompt bleed-through or very short/empty responses
        if not generated_text or generated_text.startswith("User Question:") or len(generated_text) < 10:
            print(f"Warning: Received potentially empty/invalid response from TinyLlama: '{generated_text[:100]}...'")
            # Try a simple fallback if retrieval worked but generation failed
            first_doc_text = context.split("--- Context Snippet 1 ---")[1].split("---")[0].strip() if "--- Context Snippet 1 ---" in context else "some relevant details"
            return f"I found {first_doc_text}, but I'm having a little trouble putting together a full sentence right now. Could you try rephrasing?"

        print("Response received from Ollama.")
        # Post-processing: Remove potential self-correction notes if tinyllama adds them
        generated_text = re.sub(r'\(\s*based on context\s*\)|\(\s*from context\s*\)', '', generated_text, flags=re.IGNORECASE).strip()

        return generated_text

    except requests.exceptions.Timeout:
        print(f"Error: Request to Ollama model '{OLLAMA_MODEL_NAME}' timed out after 60 seconds.")
        return f"Apologies! It took a bit too long to get a response from the '{OLLAMA_MODEL_NAME}' model. Maybe try asking again?"
    # Keep other exception handling as before...
    except requests.exceptions.RequestException as e:
        print(f"Error communicating with Ollama API at {OLLAMA_API_GENERATE_URL}: {e}")
        return "My connection to the language model seems to be down. Please ensure Ollama is running correctly."
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON response from Ollama.")
        print(f"Raw Response Text: {response.text[:500]}...")
        return "Something went wrong receiving the response from the language model."
    except Exception as e:
        print(f"An unexpected error occurred during Ollama generation: {e}")
        return "Oops! An unexpected glitch happened while I was thinking. Let's try that again."


# --- Main Pipeline Function ---

# Add conversation history tracking (Simple version)
conversation_history = []

def get_rag_response(query: str) -> str:
    """
    The main RAG pipeline function using Ollama. Includes simple history tracking.
    """
    global conversation_history
    # 1. Ensure models/data are loaded
    if not load_models_and_data():
         if not ollama_available: return f"Error: Cannot connect to the Ollama model '{OLLAMA_MODEL_NAME}'. Is Ollama running?"
         else: return "Error: Failed to load chatbot components. Check logs."

    # 2. Retrieve relevant documents
    try: retrieved_docs = retrieve_relevant_documents(query)
    except Exception as retrieval_err:
         print(f"Error during document retrieval: {retrieval_err}"); return "Sorry, I had trouble searching for that."

    # Handle no documents found
    if not retrieved_docs:
        return f"Hmm, I couldn't find specific details matching '{query}'. Perhaps ask about a specific restaurant menu or feature I know about?"

    # 3. Format the context
    context_str = format_context(retrieved_docs)
    if "No relevant information" in context_str or "could not extract usable text" in context_str:
         return "I found some related info, but not quite enough to answer confidently. Could you refine your question?"

    # 4. Generate the response using Ollama
    response = generate_response_with_ollama(query, context_str)

    # 5. Add to history (basic)
    conversation_history.append({"role": "user", "content": query})
    conversation_history.append({"role": "assistant", "content": response})
    # Limit history size if needed:
    # history_limit = 10 # Keep last 5 pairs
    # if len(conversation_history) > history_limit * 2:
    #     conversation_history = conversation_history[-(history_limit * 2):]

    return response

# --- Example Usage (for testing) ---
if __name__ == "__main__":
    print(f"Testing RAG Pipeline with Ollama model: {OLLAMA_MODEL_NAME}...")

    if not load_models_and_data():
        print("Exiting due to loading errors.")
        exit(1)

    # Example Queries:
    test_queries = [
        "What is the price of Paneer Tikka Masala at Spice Villa?",
        "Does Green Leaf Café sound like a good spot for vegans?", # Changed phrasing
        "Tell me about the appetizer options at Spice Villa versus Urban Diner.", # Changed phrasing
        "When is Ocean's Catch open?", # Changed phrasing
        "I need gluten-free desserts. Does Sweet Delight have anything suitable based on your info?", # Changed phrasing
        "Which restaurants are good for seafood lovers?", # Changed phrasing
        "Describe the Classic Cheeseburger at Urban Diner. Does it sound good?", # Changed phrasing
        "What's a really affordable dish at Spice Villa?", # Changed phrasing
        "Can I sit outside at any of these places?", # Changed phrasing
        "Where can I grab a bite really late at night?", # Changed phrasing
        "How spicy is the food at Spice Villa compared to Ocean's Catch, based on menu notes?" # Changed phrasing
    ]

    for q in test_queries:
        print("\n" + "="*50)
        print(f"User Query: {q}")
        answer = get_rag_response(q)
        print(f"\nZomato Genie Response:\n{answer}")
        print("="*50)

    # Display final history (optional)
    # print("\n--- Conversation History ---")
    # for turn in conversation_history:
    #     print(f"{turn['role'].capitalize()}: {turn['content']}")
    # print("--------------------------")