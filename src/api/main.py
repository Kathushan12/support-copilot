from fastapi import FastAPI
from fastapi.responses import JSONResponse

from src.api.schemas import TicketRequest, TicketResponse
from src.triage.predict import predict_category
from src.rag.answer import generate_grounded_reply

app = FastAPI(title="Trusted Support Copilot")


def simple_priority_rule(text: str) -> str:
    t = (text or "").lower()
    if any(x in t for x in ["fraud", "unauthorized", "identity theft", "stolen", "hacked"]):
        return "High"
    if any(x in t for x in ["refund", "charge", "billing", "payment", "charged twice", "double charged"]):
        return "Medium"
    return "Low"


@app.get("/")
def root():
    return {
        "status": "ok",
        "service": "Trusted Support Copilot API",
        "endpoints": {
            "analyze": "POST /analyze",
            "docs": "/docs",
            "health": "/health",
        },
    }


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/analyze", response_model=TicketResponse)
def analyze(req: TicketRequest):
    try:
        category, conf = predict_category(req.text)
        priority = simple_priority_rule(req.text)
        rag = generate_grounded_reply(req.text)

        return TicketResponse(
            category=category,
            category_confidence=float(conf) if conf is not None else 0.0,
            priority=priority,
            reply=rag.get("final_reply", ""),
            found_in_kb=bool(rag.get("found_in_kb", False)),
            citations=rag.get("citations", []),
        )

    except Exception as e:
        # Return a clean JSON error instead of a big crash traceback during demo
        return JSONResponse(
            status_code=500,
            content={"error": "Internal Server Error", "detail": str(e)},
        )
