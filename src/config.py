import os
from dotenv import load_dotenv

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4.1-mini")
OPENAI_EMBED_MODEL = os.getenv("OPENAI_EMBED_MODEL", "text-embedding-3-small")

BASE_DIR = os.path.dirname(os.path.dirname(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
RAW_DIR = os.path.join(DATA_DIR, "raw")
PROCESSED_DIR = os.path.join(DATA_DIR, "processed")
KB_DIR = os.path.join(DATA_DIR, "kb")

MODELS_DIR = os.path.join(BASE_DIR, "models")
INDEX_DIR = os.path.join(BASE_DIR, "indexes")

for d in [RAW_DIR, PROCESSED_DIR, KB_DIR, MODELS_DIR, INDEX_DIR]:
    os.makedirs(d, exist_ok=True)
