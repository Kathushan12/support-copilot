from typing import Dict, Any, List
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
            "snippet": {"type": "string"}
          },
          "required": ["doc_id", "title", "snippet"]
        }
      }
    },
    "required": ["found_in_kb", "final_reply", "citations"]
  }
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

def generate_grounded_reply(ticket_text: str) -> Dict[str, Any]:
    retrieved = retrieve(ticket_text, k=4)

    # If retrieval is weak, still run but likely “not found”
    prompt = build_prompt(ticket_text, retrieved)

    resp = client.responses.create(
        model=OPENAI_MODEL,
        input=prompt,
        text={
            "format": {
                "type": "json_schema",
                "name": RESPONSE_SCHEMA["name"],
                "schema": RESPONSE_SCHEMA["schema"],
                "strict": True
            }
        }
    )

    # Parse output
    out = resp.output[0].content[0].parsed
    # Attach citations snippets from retrieved (ensure they match doc ids)
    if out["found_in_kb"] and len(out["citations"]) == 0:
        out["citations"] = [{"doc_id": r["doc_id"], "title": r["title"], "snippet": r["chunk"][:240]} for r in retrieved[:2]]
    return out
