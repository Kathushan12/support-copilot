from pydantic import BaseModel, Field
from typing import List, Optional

class TicketRequest(BaseModel):
    text: str = Field(min_length=10, max_length=5000)

class Citation(BaseModel):
    doc_id: str
    title: str
    snippet: str

class TicketResponse(BaseModel):
    category: str
    category_confidence: Optional[float] = None
    priority: str
    reply: str
    found_in_kb: bool
    citations: List[Citation]
