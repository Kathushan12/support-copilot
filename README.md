# Trusted Customer Support Copilot — Ticket Triage + RAG (OpenAI)

An end-to-end support assistant that **classifies customer tickets**, assigns **priority**, and generates **KB-grounded replies with citations** to reduce hallucinations.

## ✨ Key Features
- **Ticket Triage (ML):** Predicts a support category + confidence score  
- **Priority Routing (Rules):** High / Medium / Low based on risk keywords  
- **RAG Answering (KB-only):** Drafts responses using only internal policies/FAQs  
- **Citations:** Each reply includes the exact KB snippet used  
- **FastAPI + Gradio UI:** API endpoint + clean interactive demo interface

---

## 🧠 Architecture (High Level)
1. **User submits ticket** (Gradio UI or API)
2. **ML triage model** predicts category (TF-IDF + Logistic Regression)
3. **Rule-based priority** flags urgency (fraud/unauthorized → High)
4. **RAG pipeline**
   - Embed query (OpenAI embeddings)
   - Retrieve top KB chunks via **FAISS**
   - Generate reply with **strict KB-only prompt**
   - Return structured JSON + citations
5. API returns a single structured response

---

## 📦 Tech Stack
- **Python**
- **FastAPI** (API)
- **Gradio** (UI)
- **scikit-learn** (TF-IDF + Logistic Regression)
- **OpenAI API** (Embeddings + LLM)
- **FAISS** (Vector similarity search)
- **Pandas** (data prep)

---

## 📁 Project Structure
```text
support-copilot/
├─ src/
│  ├─ api/        # FastAPI app
│  ├─ ingest/     # dataset preprocessing
│  ├─ triage/     # training + prediction
│  ├─ rag/        # retrieval + answer generation + indexing
│  └─ ui/         # Gradio UI
├─ data/
│  ├─ kb/         # Knowledge Base documents (tracked)
│  ├─ raw/        # raw dataset (ignored)
│  └─ processed/  # processed dataset (ignored)
├─ models/        # saved model (ignored)
├─ indexes/       # FAISS index (ignored)
└─ .env           # secrets (ignored)

```
# Setup & Demo Guide — Trusted Customer Support Copilot

This guide covers **environment setup**, **KB indexing**, **model training**, **running the API/UI**, and **demo test inputs**.

---

## ⚙️ Setup

### 1) Create & activate virtual environment
```bash
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```
### 2) Install dependencies
```
pip install -r requirements.txt
```
### 3) Add environment variables
Create a .env file in the project root:
```bash
OPENAI_API_KEY=your_key_here
OPENAI_MODEL=gpt-4o-mini
OPENAI_EMBED_MODEL=text-embedding-3-small
```
Note: .env is ignored by git

## 🗂️ Knowledge Base (KB)
Add markdown documents inside:
```bash
data/kb/
```
After editing KB docs, rebuild the index:
```bash
python -m src.rag.build_index
```

## 🧪 Train the Triage Model (ML)
1) Preprocess dataset
   ```bash
   python -m src.ingest.preprocess
    ```
2) Train
   ```bash
   python -m src.triage.train
   ```
   
## 🚀 Run the API
Start the FastAPI server:
```bash
uvicorn src.api.main:app --reload --port 8000
```

## 🖥️ Run the UI (Gradio)

Make sure the API is running first, then:
```bash
python src/ui/gradio_app.py
```
Open the local Gradio URL shown in the terminal (usually):
```bash
http://127.0.0.1:7860
```

