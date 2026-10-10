# Interview Prep: RAG Chatbot (LangChain + Groq + HuggingFace Embeddings + FAISS + Streamlit)

> My project: I uploaded a **Java study material PDF** (`data/BSCIT_304_slm.pdf`) and built a chatbot that answers questions **only from that PDF**.
> (The project started from a medical-chatbot template. I replaced the data with Java material and renamed it to CSBot. Mention this honestly if asked. The project folder is still named `medical-chatbot-main`.)

---

## 0. The Story (how to present it in 60–90 seconds)

**Hook (the problem):**
"While studying Java, I had a long PDF of study material. Searching it by keyword was slow, and when I asked ChatGPT-style models Java questions, they sometimes answered from general knowledge, or even made things up, instead of from *my* syllabus. I wanted a chatbot that answers **only from my material** and **shows me where the answer came from**."

**Idea (the solution):**
"So I built a **RAG (Retrieval-Augmented Generation)** application. Think of it as an *open-book exam*. The LLM is the student, my PDF is the book, and the vector database is the index that finds the right pages quickly."

**How it works (the flow):**
1. **Ingestion:** I load the Java PDF with LangChain's `PyPDFLoader`.
2. **Chunking:** I split it into small pieces (500 characters, 50 overlap) so each piece holds one idea.
3. **Embeddings:** Each chunk is turned into a vector by the HuggingFace model `all-MiniLM-L6-v2`, so the computer can search by *meaning*.
4. **Vector store:** The vectors are saved in **FAISS**, a fast similarity-search database, stored locally.
5. **Question time:** The user asks a question in a **Streamlit** chat UI. The question is embedded, FAISS returns the **top 3 most similar chunks**, and these go with the question into a prompt.
6. **Generation:** The prompt goes to **`openai/gpt-oss-20b`** through the **Groq API** (I first used Llama 3.1 8B on the Hugging Face Inference API, but the free tier ran out, and Groq later retired its Llama models). It answers *only from the context*, or says "I don't know".
7. **Transparency:** The app also shows the **sources** (file, page and a short snippet) used.

**Result / learning:**
"Answers are grounded in my material, hallucinations are reduced, and I can verify each answer from the source. I learned the whole RAG pipeline end to end: loading, chunking, embedding, indexing, retrieval, prompting, and UI."

**Closing line:** "Next I'd improve it with chat memory, better chunking, re-ranking and evaluation."

### Analogy to remember
- **LLM** = a smart student who has not read your book
- **PDF** = the textbook
- **Chunks** = index cards / pages with one topic each
- **Embeddings** = a "meaning coordinate" for each card
- **FAISS** = the librarian who finds the 3 closest cards in milliseconds
- **Prompt** = "answer using only these cards"
- **LangChain** = the manager who connects all of them

---

## 1. What is a RAG application?

**RAG = Retrieval-Augmented Generation.** It combines:
- **Retrieval:** find relevant information from your own data (documents, PDFs, DB).
- **Augmented:** add that information to the prompt as context.
- **Generation:** the LLM writes the answer using that context.

**Why RAG?**
- LLMs have a **knowledge cutoff** and do not know your private data.
- LLMs can **hallucinate** (confidently make things up).
- Retraining or fine-tuning is expensive. RAG only needs you to update the data and re-index.
- You can show **sources/citations**.

**Flow:** `Documents → Chunks → Embeddings → Vector DB` (offline, done once), then `Question → Embedding → Search → Top-k chunks + Question → LLM → Answer` (online, per query).

**In my project:** `create_memory_for_llm.py` is the offline part. `connect_memory_with_llm.py` (CLI) and `csbot.py` (Streamlit) are the online part.

**One-liner:** "RAG gives the LLM an open book instead of making it answer from memory."

---

## 2. What is an LLM?

**LLM = Large Language Model.** A neural network (Transformer architecture) trained on huge amounts of text to **predict the next token**. Because of that, it can answer questions, summarize, translate, write code, etc.

- "Large" = billions of parameters (e.g., `gpt-oss-20b` = 20 billion parameters; Llama 3.1 8B = 8 billion).
- Examples: GPT-4 (OpenAI), Claude (Anthropic), Gemini (Google), Llama (Meta), Mistral, Falcon.
- **Limits:** knowledge cutoff, hallucinations, limited context window, no access to private data. RAG addresses these.
- **Instruct/chat models** (like `gpt-oss-20b` or `Llama-3.1-8B-Instruct`) are fine-tuned to follow instructions and hold conversations.
- **Temperature** (I used 0.5) controls randomness. Lower is more factual and higher is more creative.
- **max_tokens** (I used 512) caps the length of the answer.

---

## 3. What is the use of LangChain?

LangChain is a **framework/orchestration library** for building LLM applications. It gives ready-made building blocks so you don't write the plumbing yourself:

| Need | LangChain component (used in my project) |
|---|---|
| Load documents | `PyPDFLoader`, `DirectoryLoader` |
| Split text | `RecursiveCharacterTextSplitter` |
| Embeddings | `HuggingFaceEmbeddings` |
| Vector store | `FAISS` wrapper (`from_documents`, `save_local`, `load_local`) |
| LLM wrapper | `ChatGroq` (from `langchain-groq`) |
| Prompts | `PromptTemplate` |
| Chain (retrieve + prompt + LLM) | `RetrievalQA` |
| Retriever | `db.as_retriever(search_kwargs={'k':3})` |

**Benefits:** a unified interface (swap Llama for GPT or FAISS for Pinecone with little code change), faster development, and support for chains, agents, memory and tools.

**Honest note:** `RetrievalQA` is a legacy chain (imported here from `langchain_classic`). Newer code uses **LCEL** (LangChain Expression Language) or **LangGraph**. Saying this shows you know the ecosystem.

---

## 4. After ingestion, why convert to chunks? What are chunk size and chunk overlap?

### Why chunks?
1. **LLM context window is limited.** You can't send a whole book in every prompt.
2. **Cost and speed:** fewer tokens means cheaper and faster.
3. **Retrieval accuracy:** a small chunk about one topic gets a focused embedding. A giant page mixes many topics, so the vector becomes "blurry".
4. **Embedding models have input limits.** `all-MiniLM-L6-v2` handles roughly 256 word-pieces by default (longer text gets truncated).
5. We send only the **relevant** pieces, so the answer is more precise and there is less noise.

### Chunk size
The **maximum length of each chunk** (characters here, tokens in some splitters). Mine is **500 characters**.
- Too small: loses context, such as half a definition.
- Too large: mixed topics, noisy retrieval, more tokens.

### Chunk overlap
The number of characters **repeated between consecutive chunks**. Mine is **50**.
- Prevents a sentence or idea from being **cut at the boundary** and losing meaning.
- Too much overlap: duplicates, a bigger index, wasted tokens.

**Example:** chunk 1 = chars 0–500, chunk 2 = chars 450–950. The last 50 characters of chunk 1 appear again at the start of chunk 2.

### Why `RecursiveCharacterTextSplitter`?
It tries to split on natural boundaries in order: paragraphs (`\n\n`) → lines (`\n`) → spaces → characters. So chunks stay as semantically intact as possible.

**Tuning tip:** common values are 500–1000 with 10–20% overlap. Test with real questions and check whether the retrieved chunks actually contain the answer.

---

## 5. What does vector embedding mean?

An **embedding** converts text into a **list of numbers (a vector)** that captures its **meaning**. Texts with similar meaning get vectors that are **close together** in space.

- `all-MiniLM-L6-v2` produces **384-dimensional** vectors.
- "What is inheritance in Java?" and "Explain the concept of a subclass reusing a parent class" end up close together, even with few shared words. This is **semantic search**, as opposed to keyword search.
- **Similarity** is measured by cosine similarity or L2 (Euclidean) distance. FAISS here uses L2 distance by default.
- The **same embedding model must be used** for indexing and querying. My code uses the same model in both places.

**Simple explanation:** "It's GPS coordinates for meaning. Sentences about the same topic live in the same neighborhood."

---

## 6. What is the use of Hugging Face?

Hugging Face is the **"GitHub of AI models"**: a hub and platform with:
- **Model Hub:** thousands of open models (Llama, Mistral, sentence-transformers, BERT).
- **Datasets** and **Spaces** (host demos).
- **Libraries:** `transformers`, `sentence-transformers`, `datasets`.
- **Inference API / Endpoints:** run models on their servers by calling an API with an **HF token**, so I don't need a GPU. (I used this at first, but the free tier has a usage cap, so I moved the LLM to Groq.)

**In my project it's used for one thing now:**
- **Embedding model** `sentence-transformers/all-MiniLM-L6-v2` (downloaded from the Hub once, then runs locally, free, no token needed).

**The LLM is served by Groq:** `openai/gpt-oss-20b`, called remotely via `ChatGroq` with `GROQ_API_KEY` stored in `.env`. Groq is a fast inference provider that runs open models on its own hardware with a free tier.

> Security tip: never commit `.env` or the API key to GitHub. `.env` is listed in `.gitignore`. If a key was ever shared or committed, **revoke and regenerate** it.

---

## 7. What is the use of FAISS?

**FAISS = Facebook AI Similarity Search**, an open-source library by Meta for **fast similarity search over vectors**.

- Stores all chunk vectors and quickly finds the **k nearest neighbours** to a query vector.
- Runs **locally and free**, with no server and no setup. It saves to disk as `index.faiss` (vectors) and `index.pkl` (the documents and metadata).
- Scales to millions of vectors. It supports approximate search (IVF, HNSW) for speed.
- In my project: `FAISS.from_documents(...)` builds it, `save_local()` saves it, `load_local()` loads it, and `as_retriever(k=3)` returns the top 3 chunks.

**Why a vector DB and not SQL?** SQL matches exact values and keywords. A vector DB matches by **meaning** using nearest-neighbour search.

**Alternatives:** Chroma, Pinecone, Weaviate, Qdrant, Milvus, pgvector.
**FAISS limits:** it is a library, not a full database. It has no built-in auth, no easy metadata filtering, and it is single-machine.

**Security note (good to mention):** I used `allow_dangerous_deserialization=True` because `index.pkl` is a **pickle file**. This is safe only because I created the file myself. Never load pickle files from untrusted sources.

---


-

---

## 9. Why Groq, Hugging Face and LangChain? Are there other options?

**Why I chose them:**
- **Groq:** free tier, very fast inference, simple API key, no GPU needed. I switched to it when the Hugging Face free inference quota ran out. Because of LangChain, only the LLM wrapper changed (`HuggingFaceEndpoint` → `ChatGroq`). The embeddings and the FAISS index stayed the same.
- **Hugging Face:** free and open models and a huge catalogue. I still use its `sentence-transformers` embedding model locally.
- **LangChain:** ready-made RAG components and a large community, so I could build the pipeline fast.

They are **not competitors**: Hugging Face and Groq provide the *models*, LangChain *orchestrates* them.

**Alternatives**

| Category | Options |
|---|---|
| Orchestration frameworks | **LlamaIndex** (strong for RAG and data indexing), **Haystack**, **LangGraph** (agents/workflows), Semantic Kernel, DSPy, or plain Python |
| LLM providers | OpenAI (GPT), Anthropic (Claude), Google (Gemini), **Groq (mine)**, Hugging Face Inference, Together AI, Mistral AI, Azure OpenAI, AWS Bedrock |
| Run LLM locally | **Ollama**, LM Studio, llama.cpp, vLLM |
| Embedding models | OpenAI `text-embedding-3`, Cohere, Google, BGE, E5, `all-mpnet-base-v2` |
| Vector DBs | Chroma, Pinecone, Weaviate, Qdrant, Milvus, pgvector |
| UI | Streamlit (mine), Gradio, Chainlit, React frontend |

**How to justify:** "I picked open-source and free tools so I could learn each piece without cost. Thanks to LangChain's abstractions, I can swap the LLM provider or FAISS for Pinecone with minimal changes. I already did this once, moving from Hugging Face to Groq when the free quota ended."

---

## 10. Walk through my code (be ready for this)

| File | Role |
|---|---|
| `create_memory_for_llm.py` | **Ingestion pipeline:** load PDFs → chunk (500/50) → embed → store in FAISS (`vectorstore/db_faiss`). Run once, or again when data changes. |
| `connect_memory_with_llm.py` | **CLI version:** loads FAISS, connects `gpt-oss-20b` via Groq, builds `RetrievalQA`, asks one question. |
| `csbot.py` | **Streamlit chat UI:** same logic, plus chat history display (`st.session_state`), vector store and chain cached with `@st.cache_resource`, clean source citations, errors shown in the UI. |
| `.env` | Holds `GROQ_API_KEY` (secret, git-ignored). |
| `data/` | The Java PDF. |

**Key details to remember:**
- `chain_type="stuff"`: all retrieved chunks are "stuffed" into one prompt (simplest, good for k=3).
- `k=3`: retrieve the top 3 chunks.
- **Custom prompt:** "Use only the context; if you don't know, say you don't know." This is the main anti-hallucination guard.
- `return_source_documents=True`: shows the sources for trust and verification.
- `temperature=0.5`: balanced. For strict factual answers, go lower, such as 0.1–0.2.

**Known limitation to be honest about:** the Streamlit chat history is only *displayed*. It is not sent to the LLM, so the bot has **no conversational memory** (follow-up questions like "explain it more" won't work well). Also, the retriever is built once and cached, but the chat history is still not used in the prompt.

---

## 11. Likely interview questions with short answers

1. **RAG vs fine-tuning?** RAG injects knowledge at query time (cheap, easy to update, gives sources). Fine-tuning changes model weights (better for style or behaviour, costly, harder to update).
2. **How do you reduce hallucination?** Strict prompt, good retrieval, low temperature, showing sources, "I don't know" fallback, evaluation.
3. **What if the answer isn't in the PDF?** The prompt tells the model to say it doesn't know.
4. **How did you choose chunk size?** Started at 500/50. Tune by testing questions and checking whether the retrieved chunks contain the answer.
5. **Why `k=3`?** A balance of enough context against token cost and noise. Tunable.
6. **What if retrieval returns wrong chunks?** Better chunking, a better embedding model, hybrid search (BM25 + vectors), re-ranking, metadata filters, query rewriting.
7. **Cosine vs L2 distance?** Cosine measures angle (direction/meaning). L2 measures straight-line distance. For normalized vectors they give equivalent rankings.
8. **How would you evaluate the RAG?** Build a Q&A test set. Measure retrieval hit rate/recall@k, and answer faithfulness and relevance. Tools: RAGAS, TruLens, LangSmith.
9. **How to scale?** A managed vector DB (Pinecone, Qdrant), async processing, caching, batch embedding, a hosted API.
10. **How to add memory?** Keep the chat history and use a history-aware retriever, or condense the question with history (LangChain message history or LangGraph).
11. **Does the PDF data leave your machine?** Embeddings and FAISS are local. The question and the retrieved chunks are sent to the Groq API. For fully private use, run the model locally (Ollama).
13. **Embedding vs LLM?** The embedding model converts text to vectors for search. The LLM generates the answer text. They are different models with different jobs.
14. **Scanned PDFs?** `PyPDFLoader` only reads text PDFs. Scanned ones need OCR (Tesseract, Unstructured).

---

## 12. Suggestions to improve the project (talk about these as "future work")

1. **Add conversational memory** so follow-ups work.
2. **Rename the project and UI** ("Java Study Assistant") and update the README so it matches the Java material.
3. **Use a hybrid search + re-ranker** (BM25 + vectors + cross-encoder) for better retrieval.
4. **Better chunking:** larger chunks (e.g., 800–1000), split by headings or chapters, and keep page numbers in metadata.
5. ~~Show clean citations~~ **Done:** sources now show file, page number and a snippet.
6. **Stream the response** for better UX.
7. **Evaluate** with a set of 20–30 Java questions and track accuracy (RAGAS).
8. **Upgrade to LCEL/LangGraph** instead of the legacy `RetrievalQA`.
9. **Support multiple PDFs and upload from the UI.**
10. **Security hygiene:** `.env` is now git-ignored. Still to do: rotate the key regularly and avoid loading untrusted pickle files.
11. **Deploy** to Streamlit Cloud, Hugging Face Spaces or Docker.
12. ~~Cache the chain and LLM~~ **Done:** the chain is cached with `st.cache_resource`.
13. **Handle Groq rate limits (429)** with a friendly message or retry.

---

## 13. More interview preparation tips

**Topics to revise**
- Python basics, virtual environments (**Pipenv** is used here, see `Pipfile`), `.env` handling.
- Transformer basics, attention (high level), tokenization.
- Embeddings and similarity metrics.
- Prompt engineering: system prompts, few-shot, "answer only from context".
- Vector DB concepts: ANN, indexing, metadata filtering.
- RAG evaluation, hallucination, guardrails.
- Agents and tool calling (be aware of the next trend).
- Basic Git and REST APIs.

**Presentation tips**
- Start with the **problem, not the tech**. Use the open-book exam analogy.
- Keep a **30-second**, **2-minute** and **5-minute** version of your story.
- Have a **live demo** ready: ask 2 questions from the PDF and 1 outside the PDF (to show "I don't know").
- Draw the architecture on paper or a whiteboard: `PDF → chunks → embeddings → FAISS → retriever → prompt → LLM → answer`.
- Be honest about limitations (no memory, template origin). Interviewers value that, and then tell them how you would fix them.
- Use numbers: PDF page count, number of chunks (the scripts print them), chunk size 500/50, k=3, 384-dim vectors, 20B-parameter model (`gpt-oss-20b`).
- Prepare for "what was the hardest part?". Ideas: LangChain version and import changes (`langchain_classic`), choosing chunk size, the Hugging Face free-tier limit and migrating to Groq (including a retired model returning `model_not_found`).

**Practice**
- Explain each step aloud in simple words, as if to a non-technical person.
- Run the project once before the interview and make sure the demo works and the Groq API key is valid.
- Read the LangChain RAG tutorial and try LlamaIndex or Ollama once, so you can speak about alternatives.

---

## 14. Issues we faced and how we resolved them

| # | Issue | Cause | Resolution |
|---|---|---|---|
| 1 | Hugging Face LLM calls stopped working | The **free-tier Inference API quota/credits were used up** | Moved the LLM to **Groq** (free tier). Only the LLM wrapper changed (`HuggingFaceEndpoint`/`ChatHuggingFace` → `ChatGroq`). The embeddings and the FAISS index were reused, so no rebuild was needed. |
| 2 | Groq key was stored as `HF_TOKEN` in `.env` | The old variable name was kept when the key was swapped | Renamed it to `GROQ_API_KEY` and read it with `os.environ.get("GROQ_API_KEY")`. This avoids sending a Groq key to Hugging Face by mistake. |
| 3 | `404 model_not_found` for `llama-3.1-8b-instant` | Groq had **retired the Llama models** for this account | Listed the models available to the key through the Groq API (`client.models.list()`), then picked `openai/gpt-oss-20b` and tested it end to end. |
| 4 | Parameter names differed between the wrappers | `HuggingFaceEndpoint` uses `max_new_tokens` and `task="conversational"`; `ChatGroq` uses `max_tokens` and needs no `task` | Rewrote `load_llm` for `ChatGroq` (model, temperature, `max_tokens`, `api_key`). |
| 5 | New dependency was not in the project | `langchain-groq` was missing and `Pipfile.lock` was out of date | Added `langchain-groq` and `python-dotenv` to the `Pipfile`, ran `pipenv install` to re-lock and install. |
| 6 | Chain and LLM were rebuilt on every question | They were created inside the chat handler | Cached the chain with `@st.cache_resource`. |
| 7 | The "vector store failed to load" check did not stop execution | It showed an error but had no `return` | Removed the dead check. A missing `GROQ_API_KEY` now shows a clear error and returns early. |
| 8 | Sources were printed as raw `Document` objects | `str(source_documents)` was used | Now shows the file name, page number and a 150-character snippet. |
| 9 | Secret handling | No `.gitignore`, so `.env` could be committed | Added `.gitignore` (`.env`, `__pycache__/`). If a key was ever shared, rotate it. |
| 10 | Docs out of sync | README and this file still described Hugging Face | Updated `README.md` and `prep.md`. |

Earlier setup issues (from the original build): LangChain import changes (`RetrievalQA` now comes from `langchain_classic`), choosing chunk size, and getting the Hugging Face endpoint `task` right.

**How I verified the fix:** ran the CLI script and the Streamlit app (with Streamlit's `AppTest`) on a real question ("What is polymorphism?"). Both returned a correct answer from the PDF with sources, and no errors.

---

## 15. Why we moved from Hugging Face to Groq

**Short answer:** the Hugging Face free Inference API ran out of quota, and Groq's free tier gave a working LLM at no cost with minimal code changes.

**What actually changed**
- Only the **LLM** moved. Embeddings (`all-MiniLM-L6-v2`) still run locally through Hugging Face `sentence-transformers`, so the FAISS index stayed valid.
- The code change was small because LangChain hides the provider behind a common chat-model interface.

**Hugging Face Inference API vs Groq: comparison**

| Aspect | Hugging Face Inference API | Groq |
|---|---|---|
| What it is | Hub and platform. It routes requests to many inference providers | Dedicated, very fast inference provider that runs open models on its own hardware (LPUs) |
| Free tier | Limited monthly credits. It ran out for me | Free tier with per-minute and per-day rate limits |
| Speed | Varies by model and provider. Can be slower, with cold starts | Generally much faster responses (low latency, high tokens/second) |
| Model catalogue | Very large (thousands of models) | Smaller, curated set. Models get retired (my Llama model did) |
| Setup | `HF_TOKEN`, gated models need access approval (Llama) | One `GROQ_API_KEY`, no model access approval |
| Failure mode I hit | Quota exhausted | `404 model_not_found` (retired model), possible `429` rate limits |
| Best for | Exploring and trying many different models | Fast, low-latency chat and demos |

**Which one is more efficient, and why?**
- **For this project, Groq is the more efficient choice.** It is generally faster per response and it worked on the free tier when the Hugging Face quota did not. Setup is simpler: one key, no gated-model approval.
- **Hugging Face is more flexible**, not more efficient. It wins when you need a specific or niche model, fine-tunes, or want to self-host the same model later.
- **Caveats (be honest):** I did not run a formal benchmark. "Faster" is based on the provider's design and general experience, not on measured numbers from this project. Groq's free tier has rate limits and its model list changes, so for production I would use a paid plan, handle `429` errors with retries, and avoid depending on one model name.
- **Embeddings stay on Hugging Face** because running `all-MiniLM-L6-v2` locally is free, private and fast enough for this size of PDF.

**How to justify it in an interview:** "My LLM provider's free quota ran out, so I swapped providers. Because of LangChain's abstractions I changed only the LLM wrapper and the key, and reused the embeddings and vector store. I tested it end to end, and then hit a retired-model error, which I fixed by listing the models available to my key."

---

**Good luck!**
