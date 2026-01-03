from typing import Dict, Any, List
import json

from openai import OpenAI
from src.config import OPENAI_API_KEY, OPENAI_MODEL
from src.rag.retrieve import retrieve

client = OpenAI(api_key=OPENAI_API_KEY)

RESPONSE_SCHEMA = {
    "name": "support_reply",
    "schema": {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "found_in_kb": {"type": "boolean"},
            "final_reply": {"type": "string"},
            "citations": {
                "type": "array",
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "doc_id": {"type": "string"},
                        "title": {"type": "string"},
                        "snippet": {"type": "string"},
                    },
                    "required": ["doc_id", "title", "snippet"],
                },
            },
        },
        "required": ["found_in_kb", "final_reply", "citations"],
    },
}

def build_prompt(ticket_text: str, retrieved: List[Dict]) -> str:
    context_lines = []
    for r in retrieved:
        snippet = r["chunk"][:600]
        context_lines.append(f"[{r['doc_id']} | {r['title']}]\n{snippet}\n")
    context = "\n---\n".join(context_lines)

    return f"""
You are a customer support assistant.

RULES:
- Use ONLY the provided KB context to answer.
- If KB does not contain the answer, set found_in_kb=false and ask 1-2 clarifying questions.
- Never invent policies or steps.
- Be concise and professional.

TICKET:
{ticket_text}

KB CONTEXT:
{context}
""".strip()

def _extract_json_text(resp) -> str:
    try:
        return resp.output[0].content[0].text
    except Exception:
        if hasattr(resp, "output_text"):
            return resp.output_text
        raise

def _fallback_not_found() -> Dict[str, Any]:
    return {
        "found_in_kb": False,
        "final_reply": (
            "I couldn’t find this information in our knowledge base. "
            "Could you share a bit more detail (what happened, relevant dates/amounts, and any error messages)?"
        ),
        "citations": [],
    }

def generate_grounded_reply(ticket_text: str) -> Dict[str, Any]:
    # ✅ Add threshold to avoid weak/irrelevant retrieval
    retrieved = retrieve(ticket_text, k=4, min_score=0.25)

    # ✅ Strict fallback when nothing relevant is found
    if not retrieved:
        return _fallback_not_found()

    prompt = build_prompt(ticket_text, retrieved)

    resp = client.responses.create(
        model=OPENAI_MODEL,
        input=prompt,
        text={
            "format": {
                "type": "json_schema",
                "name": RESPONSE_SCHEMA["name"],
                "schema": RESPONSE_SCHEMA["schema"],
                "strict": True,
            }
        },
    )

    raw = _extract_json_text(resp)

    # Parse JSON
    try:
        out = json.loads(raw)
    except json.JSONDecodeError:
        start = raw.find("{")
        end = raw.rfind("}")
        if start != -1 and end != -1 and end > start:
            out = json.loads(raw[start : end + 1])
        else:
            # If parsing fails, be safe and fallback
            return _fallback_not_found()

    # ✅ If model says found_in_kb=false, don't return citations
    if not out.get("found_in_kb", False):
        out["citations"] = []
        return out

    # ✅ Ensure citations exist (auto-fill from retrieved if missing)
    if out.get("found_in_kb") and not out.get("citations"):
        out["citations"] = [
            {"doc_id": r["doc_id"], "title": r["title"], "snippet": r["chunk"][:240]}
            for r in retrieved[:2]
        ]

    return out
