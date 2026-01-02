from fastapi import FastAPI
from src.api.schemas import TicketRequest, TicketResponse
from src.triage.predict import predict_category
from src.rag.answer import generate_grounded_reply

app = FastAPI(title="Trusted Support Copilot")

def simple_priority_rule(text: str) -> str:
    t = text.lower()
    if any(x in t for x in ["fraud", "unauthorized", "identity theft", "stolen"]):
        return "High"
    if any(x in t for x in ["refund", "charge", "billing", "payment"]):
        return "Medium"
    return "Low"

@app.post("/analyze", response_model=TicketResponse)
def analyze(req: TicketRequest):
    category, conf = predict_category(req.text)
    priority = simple_priority_rule(req.text)

    rag = generate_grounded_reply(req.text)

    return TicketResponse(
        category=category,
        category_confidence=conf,
        priority=priority,
        reply=rag["final_reply"],
        found_in_kb=rag["found_in_kb"],
        citations=rag["citations"]
    )
