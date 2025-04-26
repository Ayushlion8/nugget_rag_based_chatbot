# ZOMATO_RAG_CHATBOT/nugget_rag_based_chatbot/ui/app.py

import gradio as gr
import sys
import os
import time

# --- Add Project Root to Python Path ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
sys.path.append(PROJECT_ROOT)
# ----------------------------------------

# --- Import RAG Pipeline ---
try:
    from chatbot.rag_pipeline import get_rag_response, load_models_and_data
    pipeline_loaded = True
except ImportError as e:
    print(f"Error importing RAG pipeline: {e}")
    pipeline_loaded = False
    def load_models_and_data(): print("ERROR: RAG Pipeline not loaded."); return False
    def get_rag_response(query): return "ERROR: Chatbot backend pipeline failed to load."
# --------------------------


# --- Load Models Once at Startup ---
print("Initializing Zomato Genie Chatbot...")
models_ready = False
if pipeline_loaded:
    try:
        models_ready = load_models_and_data()
        if not models_ready: print("Warning: Models or Ollama connection failed during initial load.")
    except Exception as e:
        print(f"Critical error during model loading: {e}")
        def get_rag_response(query): return f"CRITICAL ERROR during model loading: {e}."
print(f"Models Ready: {models_ready}")
# ---------------------------------


# --- Gradio Interaction Function (Updated for messages format) ---
def respond(message, chat_history):
    """
    Function called by Gradio when the user sends a message.
    Args:
        message (str): The user's input message.
        chat_history (list[dict]): List of chat messages [{'role': 'user'/'assistant', 'content': '...'}]
    Returns:
        tuple: (empty_string, updated_chat_history)
    """
    if not models_ready:
        bot_message = "Sorry, the chatbot components failed to load correctly. Please check the startup logs."
    else:
        print(f"User Query: {message}")
        start_time = time.time()
        # Call the RAG pipeline function
        bot_message = get_rag_response(message)
        end_time = time.time()
        print(f"Response Generated in {end_time - start_time:.2f} seconds")
        print(f"Bot Response: {bot_message}")

    # Append user message and bot response in the 'messages' format
    chat_history.append({"role": "user", "content": message})
    chat_history.append({"role": "assistant", "content": bot_message})

    # Return an empty string to clear the input textbox and the updated history
    return "", chat_history
# ---------------------------------


# --- Gradio Interface Definition ---
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown(
        """
        # Zomato Genie Chatbot 🍽️💬
        Ask me questions about restaurants in the database!
        Try asking about menu items, prices, features, opening hours, or comparing restaurants.

        *(Powered by RAG and Ollama)*
        """
    )

    # --- Updated Chatbot Component ---
    chatbot = gr.Chatbot(
        label="Chat Window",
        height=500,
        type='messages' # Use the recommended 'messages' type for history
        # Removed deprecated bubble_full_width
        # avatar_images=(...) # Optional avatars
        )
    # ---------------------------------

    msg = gr.Textbox(
        label="Your Question:",
        placeholder="e.g., What vegan options does Green Leaf Café have?",
        # show_label=False # Keep label for clarity
        )

    submit_btn = gr.Button("Send Message", variant="primary")

    clear_btn = gr.ClearButton([msg, chatbot], value="Clear Chat")

    # --- Event Listeners ---
    # When using type='messages', the function signature for event listeners
    # might expect slightly different inputs/outputs depending on Gradio version,
    # but the `respond` function defined above should generally work.
    # If issues arise, consult the Gradio documentation for the specific version.
    msg.submit(respond, [msg, chatbot], [msg, chatbot])
    submit_btn.click(respond, [msg, chatbot], [msg, chatbot])
    # -----------------------

# --- Launch the Gradio App ---
if __name__ == "__main__":
    print("Launching Gradio interface...")
    demo.launch(server_name="0.0.0.0", share=False)
    print("Gradio interface closed.")
# ---------------------------