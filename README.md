# ZOMATO_RAG_CHATBOT

A **Retrieval-Augmented Generation** (RAG) chatbot built on top of Zomato restaurant data. It scrapes restaurant listings (and menus), builds FAISS indexes for fast retrieval, and lets you ask natural-language questions via a simple web UI powered by OpenAI’s API.

---

## 📑 Table of Contents

1. [Features](#features)
2. [Architecture & Directory Structure](#architecture--directory-structure)
3. [Prerequisites](#prerequisites)
4. [Installation](#installation)
5. [Configuration](#configuration)
6. [Data Scraping](#data-scraping)
7. [Knowledge-Base Construction](#knowledge-base-construction)
8. [Running the RAG Pipeline](#running-the-rag-pipeline)
9. [Launching the Web UI](#launching-the-web-ui)
10. [Usage Examples](#usage-examples)
11. [Environment Variables](#environment-variables)
12. [Contributing](#contributing)
13. [License](#license)

---

## 🚀 Features

- **Web scraper** for top-50 restaurants in a city + their menus
- **Preprocessing & chunking** of scraped JSON into knowledge-base snippets
- **FAISS** vector indexes for fast retrieval
- **RAG pipeline** (`nugget_rag_based_chatbot/chatbot/rag_pipeline.py`)
- **Simple web UI** (`ui/app.py`) for interactive Q&A
- Modular design: scrape → preprocess → index → query → UI

---

## 🏗️ Architecture & Directory Structure

```
ZOMATO_RAG_CHATBOT/
├── nugget_rag_based_chatbot
│   ├── chatbot
│   │   ├── __init__.py
│   │   └── rag_pipeline.py
│   ├── data
│   │   ├── raw_data/
│   │   │   ├── lucknow_top50_restaurants.json
│   │   │   └── …
│   │   └── processed_data/
│   │       ├── kb_chunks.pkl
│   │       ├── restaurant_index.faiss
│   │       └── …
│   ├── knowledge_base
│   │   ├── preprocess.py
│   │   └── indexer.py
│   └── scraper
│       ├── scrape_Initial_Data.py
│       ├── populate_menu_scraper.py
│       └── scrape_utils.py
├── ui
│   ├── __init__.py
│   └── app.py
├── .env                   # your API keys
├── .gitignore
├── environment.yml        # conda environment spec
├── requirements.txt
├── LICENSE
└── README.md
```

---

## 📦 Prerequisites

- **Python 3.9+**
- **Conda** (recommended) or `pip`
- **Gemini API key**
- **Hugging face tokens**

---

## 🔧 Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/Ayushlion8/nugget_rag_based_chatbot
   cd ZOMATO_RAG_CHATBOT
   ```

2. **Create & activate** your environment:

   - Using **Conda**:
     ```bash
     conda env create -f environment.yml
     conda activate zomato_rag_chatbot
     ```

   - Or with **pip**:
     ```bash
     python3 -m venv venv
     source venv/bin/activate
     pip install -r requirements.txt
     ```

---

## ⚙️ Configuration

1. Copy the example `.env` and fill in your keys:
   ```bash
   cp .env.example .env
   ```
2. Edit `.env`:
   ```dotenv
   OPENAI_API_KEY=sk-...
   ZOMATO_API_KEY=your_zomato_api_key  # optional if using official API
   ```

---

## 🕵️ Data Scraping

1. **Scrape initial restaurant list** (e.g., Lucknow Top-50):
   ```bash
   python nugget_rag_based_chatbot/scraper/scrape_Initial_Data.py
   ```
2. **Populate menus** for each restaurant:
   ```bash
   python nugget_rag_based_chatbot/scraper/populate_menu_scraper.py
   ```

Raw JSON files will be in `nugget_rag_based_chatbot/data/raw_data/`.

---

## 📚 Knowledge-Base Construction

1. **Preprocess & chunk** raw JSON into text snippets:
   ```bash
   python nugget_rag_based_chatbot/knowledge_base/preprocess.py
   ```
2. **Build FAISS indexes** & metadata mappings:
   ```bash
   python nugget_rag_based_chatbot/knowledge_base/indexer.py
   ```

Artifacts are saved in `nugget_rag_based_chatbot/data/processed_data/`.

---

## 🤖 Running the RAG Pipeline

From the project root:
```bash
python -m nugget_rag_based_chatbot.chatbot.rag_pipeline \
  --index-path nugget_rag_based_chatbot/data/processed_data/restaurant_index.faiss \
  --mapping-path nugget_rag_based_chatbot/data/processed_data/index_to_doc_mapping.pkl \
  --query "Which Lucknow restaurant serves the best kebabs?"
```

This retrieves the top‐k relevant chunks and generates an answer via OpenAI.

---

## 🌐 Launching the Web UI

```bash
cd ui
python app.py
```

By default, the Flask (or FastAPI/Gradio) app runs on `http://localhost:8000`. Open it in your browser to chat with your RAG-powered assistant.

---

## 💬 Usage Examples

- **Ask**: “What’s a good budget-friendly place for biryani in Lucknow?”
- **Ask**: “Show me vegetarian options in the top-10 restaurants.”
- **Ask**: “Give me the menu for [Restaurant Name].”

---

## 🔑 Environment Variables

| Variable         | Description                          |
|------------------|--------------------------------------|
| `OPENAI_API_KEY` | Your OpenAI API key for completions  |
| `ZOMATO_API_KEY` | (Optional) Zomato API key            |

---

## 🙏 Contributing

1. Fork the repo
2. Create your feature branch:
   ```bash
   git checkout -b feature/foo
   ```
3. Commit your changes:
   ```bash
   git commit -am 'Add foo'
   ```
4. Push to branch:
   ```bash
   git push origin feature/foo
   ```
5. Open a Pull Request

Please follow the existing code style and include tests where appropriate.

---

## 📄 License

This project is released under the [MIT License](LICENSE).

---

*Built with ❤️ and Python by Your Name*