# CSBot – CS Chatbot (RAG)

CSBot is a Streamlit chatbot that answers questions from a computer science (Java) PDF using Retrieval-Augmented Generation (RAG).

## How it works

1. **Ingest** – PDFs in `data/` are loaded and split into 500-character chunks (50 overlap).
2. **Embed & index** – Chunks are embedded with `sentence-transformers/all-MiniLM-L6-v2` and stored in a FAISS index at `vectorstore/db_faiss`.
3. **Retrieve & answer** – For each question, the top 3 matching chunks are retrieved and passed to `openai/gpt-oss-20b` (via the Groq API). The model answers only from the given context, and the sources (file, page and a short snippet) are shown with the answer. Embeddings run locally, so only the LLM call uses the network.

## Project structure

| File / Folder | Purpose |
|---|---|
| `csbot.py` | Streamlit chat UI (main app) |
| `create_memory_for_llm.py` | Loads PDFs, creates chunks and embeddings, builds the FAISS vector store |
| `connect_memory_with_llm.py` | Command-line version: connects the LLM with FAISS and answers one query |
| `data/` | Source PDF(s) (e.g. `BSCIT_304_slm.pdf`) |
| `vectorstore/db_faiss/` | Generated FAISS index (`index.faiss`, `index.pkl`) |
| `.env` | Holds `GROQ_API_KEY` (git-ignored, do not commit) |
| `.gitignore` | Excludes `.env` and `__pycache__/` from version control |
| `Pipfile` / `Pipfile.lock` | Dependency definitions (pipenv) |

## Prerequisites

- Python 3.14 (as specified in the `Pipfile`)
- [pipenv](https://pipenv.pypa.io/)
- A free [Groq API key](https://console.groq.com/keys)

## Packages and libraries

| Package | Installed version | Used for |
|---|---|---|
| `streamlit` | 1.65.0 | Web chat UI |
| `langchain` | 1.4.4 | LLM application framework |
| `langchain-classic` | 1.0.8 | `RetrievalQA` chain |
| `langchain-community` | 0.4.2 | PDF loaders, FAISS vector store |
| `langchain-core` | 1.6.9 | Prompt templates |
| `langchain-text-splitters` | 1.1.3 | Text chunking |
| `langchain-huggingface` | 1.2.3 | Local Hugging Face embeddings |
| `langchain-groq` | 1.1.3 | Groq chat model (`ChatGroq`) |
| `groq` | 0.37.1 | Groq SDK (installed with `langchain-groq`) |
| `huggingface-hub` | 1.33.0 | Model download for embeddings |
| `sentence-transformers` | 6.1.0 | Embedding model (`all-MiniLM-L6-v2`) |
| `transformers` | 5.18.0 | Model dependency (installed automatically) |
| `torch` / `torchvision` | 2.14.1 / 0.29.1 | Deep-learning backend for embeddings |
| `faiss-cpu` | 1.15.1 | Vector similarity search |
| `pypdf` | 6.19.0 | PDF parsing |
| `python-dotenv` | 1.2.4 | Loads `.env` variables |

## Setup
# Install all dependencies from the Pipfile (creates the virtual environment)
pipenv install
```

To install packages individually:

```powershell
pipenv install langchain langchain-huggingface langchain-groq langchain-community langchain-classic pypdf huggingface-hub streamlit sentence-transformers faiss-cpu torchvision python-dotenv
```

Without pipenv (plain pip):

```powershell
pip install streamlit langchain langchain-huggingface langchain-groq langchain-community langchain-classic sentence-transformers faiss-cpu huggingface-hub python-dotenv pypdf torchvision
```

Create a `.env` file in the project root:

```
GROQ_API_KEY=your_groq_api_key_here
```

## Build the vector store (only when PDFs change)

Place PDFs in `data/`, then run:

```powershell
pipenv run python create_memory_for_llm.py
```

## Run the application

```powershell
pipenv run streamlit run csbot.py
```

Or activate the virtual environment first:

```powershell
pipenv shell
streamlit run csbot.py
```

Open http://localhost:8501 in your browser. Press `Ctrl+C` in the terminal to stop the app.

Without pipenv shortcuts, call the virtualenv's Python directly:

```powershell
& "C:\Users\DELL\.virtualenvs\medical-chatbot-main-Q9ByhsvT\Scripts\python.exe" -m streamlit run csbot.py
```

### Command-line version (no UI)

```powershell
pipenv run python connect_memory_with_llm.py
```

## Troubleshooting

- `pipenv streamlit run ...` fails – the command needs `run`: `pipenv run streamlit run csbot.py`.
- `No module named streamlit` – you are using the global Python; run through `pipenv run` or activate the virtualenv.
- Authentication or model errors – check `GROQ_API_KEY` in `.env`. A `model_not_found` error means Groq retired the model; pick a current one from https://console.groq.com/docs/models and update `GROQ_MODEL` in `csbot.py` and `connect_memory_with_llm.py`.
- Rate limit (429) errors – the Groq free tier has per-minute/day limits; wait and retry.
- Port already in use – run with `--server.port 8502`.

## Notes

- The LLM model is set by `GROQ_MODEL` (currently `openai/gpt-oss-20b`) in `csbot.py` and `connect_memory_with_llm.py`. Groq retires models from time to time, so update it if needed.
- Embeddings (`all-MiniLM-L6-v2`) run locally; changing the embedding model requires rebuilding the FAISS index.
- The FAISS index is loaded with `allow_dangerous_deserialization=True`; only load indexes you created yourself.
- Answers are limited to the PDF content and are not a substitute for official course material or your instructor's guidance.
