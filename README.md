# Trusted Customer Support Copilot (Triage + RAG)

End-to-end project that:

- Trains a ticket classifier (category/priority)
- Uses RAG over KB docs to generate grounded replies with citations
- Provides a Gradio UI + FastAPI backend

## Setup

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env


#Run

python -m src.ingest.download_cfpb
python -m src.ingest.preprocess
python -m src.triage.train
python -m src.rag.build_index

uvicorn src.api.main:app --reload --port 8000
python -m src.ui.gradio_app
```
