# ZOMATO_RAG_CHATBOT/nugget_rag_based_chatbot/ui/app.py

import gradio as gr
import sys, os, time, traceback

# ── Add project root to path ──────────────────────────────────────────────────
SCRIPT_DIR   = os.path.dirname(__file__)
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
sys.path.append(PROJECT_ROOT)
# ─────────────────────────────────────────────────────────────────────────────

# ── Import RAG pipeline ───────────────────────────────────────────────────────
try:
    from chatbot.rag_pipeline import get_rag_response, load_models_and_data
    pipeline_loaded = True
except ImportError as e:
    print(f"Error importing RAG pipeline: {e}")
    pipeline_loaded = False
    def load_models_and_data(): return False
    def get_rag_response(msg, history): return "ERROR: backend pipeline not loaded."
# ─────────────────────────────────────────────────────────────────────────────

# ── Initialize models ─────────────────────────────────────────────────────────
print("Initializing Zomato Genie Chatbot...")
models_ready = False
if pipeline_loaded:
    try:
        models_ready = load_models_and_data()
    except Exception:
        traceback.print_exc()
print("Models ready:", models_ready)
# ─────────────────────────────────────────────────────────────────────────────

# ── The respond function ──────────────────────────────────────────────────────
def respond(message, history):
    """
    message: str
    history: either list of [user_str, bot_str, ...] or list of {'role':..,'content':..} dicts
    """
    # Build hist_dicts in OpenAI style:
    hist_dicts = []
    # If history elements are dicts already:
    if all(isinstance(x, dict) for x in history):
        hist_dicts = history
    else:
        # history in tuples format, but maybe inner lists have extra items
        for entry in history:
            if isinstance(entry, dict):
                hist_dicts.append(entry)
            elif isinstance(entry, (list, tuple)):
                # first element = user, second = bot
                user_msg = entry[0] if len(entry) > 0 else None
                bot_msg  = entry[1] if len(entry) > 1 else None
                if user_msg:
                    hist_dicts.append({"role":"user",      "content":user_msg})
                if bot_msg:
                    hist_dicts.append({"role":"assistant", "content":bot_msg})
            else:
                # unexpected type: skip
                continue

    if not models_ready:
        return "Sorry, chatbot failed to load. Check logs."
    try:
        start = time.time()
        reply = get_rag_response(message, hist_dicts)
        print(f"Response time: {time.time() - start:.2f}s")
    except Exception as e:
        print("Error in RAG pipeline:", e)
        traceback.print_exc()
        reply = f"Error: {e}"
    return reply
# ─────────────────────────────────────────────────────────────────────────────

# ── Build UI ──────────────────────────────────────────────────────────────────
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown(
        "# Zomato Genie Chatbot 🍽️💬\n"
        "Ask questions about restaurant menus, hours, features, etc.\n"
        "*(Powered by RAG & Ollama/HF)*"
    )

    demo_chat = gr.ChatInterface(
        fn=respond,
        type="messages",
        examples=[
            "Which restaurant has the best vegetarian options?",
            "Does any restaurants have gluten-free items?",
            "Give the price range for desserts?",
            "What are the opening hours for Wow Momo?"
        ]
    )

# ── Launch ────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("Launching on http://127.0.0.1:7860")
    demo.launch(server_name="127.0.0.1", server_port=7860, share=False)



















# import gradio as gr
# import sys, os, time, traceback

# # ── Add project root to path ──────────────────────────────────────────────────
# SCRIPT_DIR   = os.path.dirname(__file__)
# PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
# sys.path.append(PROJECT_ROOT)
# # ─────────────────────────────────────────────────────────────────────────────

# # ── Import RAG pipeline ───────────────────────────────────────────────────────
# try:
#     from chatbot.rag_pipeline import get_rag_response, load_models_and_data
#     pipeline_loaded = True
# except ImportError as e:
#     print(f"Error importing RAG pipeline: {e}")
#     pipeline_loaded = False
#     def load_models_and_data(): return False
#     def get_rag_response(msg, history): return "ERROR: backend pipeline not loaded."
# # ─────────────────────────────────────────────────────────────────────────────

# # ── Initialize models ─────────────────────────────────────────────────────────
# print("Initializing Zomato Genie Chatbot...")
# models_ready = False
# if pipeline_loaded:
#     try:
#         models_ready = load_models_and_data()
#     except Exception:
#         traceback.print_exc()
# print("Models ready:", models_ready)
# # ─────────────────────────────────────────────────────────────────────────────

# # ── Typing effect simulation ──────────────────────────────────────────────────
# def respond(message, history):
#     hist_dicts = []
#     if all(isinstance(x, dict) for x in history):
#         hist_dicts = history
#     else:
#         for entry in history:
#             if isinstance(entry, dict):
#                 hist_dicts.append(entry)
#             elif isinstance(entry, (list, tuple)):
#                 user_msg = entry[0] if len(entry) > 0 else None
#                 bot_msg  = entry[1] if len(entry) > 1 else None
#                 if user_msg:
#                     hist_dicts.append({"role": "user", "content": user_msg})
#                 if bot_msg:
#                     hist_dicts.append({"role": "assistant", "content": bot_msg})

#     if not models_ready:
#         yield "Sorry, chatbot failed to load. Check logs."
#         return

#     try:
#         start = time.time()
#         response = get_rag_response(message, hist_dicts)
#         print(f"Response time: {time.time() - start:.2f}s")
#     except Exception as e:
#         print("Error in RAG pipeline:", e)
#         traceback.print_exc()
#         response = f"Error: {e}"

#     # Typing animation
#     typing_text = ""
#     for char in response:
#         typing_text += char
#         yield typing_text
#         time.sleep(0.015)
# # ─────────────────────────────────────────────────────────────────────────────

# # ── Build UI ──────────────────────────────────────────────────────────────────
# with gr.Blocks(
#     theme=gr.themes.Soft(primary_hue="pink", secondary_hue="purple"),
#     css="""
#         body {
#             background: linear-gradient(to bottom right, #ffe4e1, #fff0f5);
#             font-family: 'Poppins', sans-serif;
#         }
#         h1 {
#             text-align: center;
#             font-size: 2.8em;
#             color: #ff4d6d;
#             animation: fadeInDown 1s;
#         }
#         .description {
#             text-align: center;
#             color: #666;
#             animation: fadeInUp 1s;
#         }
#         #chatbot {
#             height: 600px;
#             border-radius: 18px;
#             box-shadow: 0 4px 12px rgba(0,0,0,0.15);
#             background: #ffffff; /* white background for better visibility */
#         }
#         @keyframes fadeInDown {
#             0% {opacity: 0; transform: translateY(-20px);}
#             100% {opacity: 1; transform: translateY(0);}
#         }
#         @keyframes fadeInUp {
#             0% {opacity: 0; transform: translateY(20px);}
#             100% {opacity: 1; transform: translateY(0);}
#         }
#     """
# ) as demo:

#     gr.Markdown(
#         """
#         <img src="https://img.icons8.com/doodle/96/restaurant.png" style="display:block;margin:0 auto 10px;"/>
#         <h1>Zomato Genie Chatbot 🍽️💬</h1>
#         <p class="description">Ask anything about restaurant menus, hours, prices, and more!</p>
#         """
#     )

#     chatbot = gr.Chatbot(
#         label="",
#         elem_id="chatbot",
#         height=600,
#         layout="bubble",
#         avatar_images=("https://img.icons8.com/doodle/48/restaurant.png", None),  # user + bot
#         show_copy_button=True,
#         show_label=False,
#         render_markdown=True,
#         type="messages",  # very important: fixes deprecation warning
#     )

#     with gr.Row():
#         msg = gr.Textbox(
#             placeholder="Type your question here... 🍲",
#             show_label=False,
#             container=True
#         )
#         send = gr.Button("Ask Genie 🧞")
#         clear = gr.Button("Clear Chat 🧹")

#     send.click(respond, [msg, chatbot], chatbot)
#     msg.submit(respond, [msg, chatbot], chatbot)
#     clear.click(lambda: [], None, chatbot)

#     gr.Examples(
#         examples=[
#             "Which restaurant has the best vegetarian options?",
#             "Do any restaurants offer gluten-free desserts?",
#             "What are the opening hours for Urban Diner?",
#             "Compare prices of appetizers at Curry Palace and Spice Villa."
#         ],
#         inputs=msg
#     )

# # ── Launch ────────────────────────────────────────────────────────────────────
# if __name__ == "__main__":
#     print("Launching on http://127.0.0.1:7860")
#     demo.launch(server_name="127.0.0.1", server_port=7860, share=False)