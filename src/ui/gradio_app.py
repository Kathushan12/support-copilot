import os
import time
from typing import Any, Dict, List, Tuple

import gradio as gr
import requests


API_URL = os.getenv("API_URL", "http://127.0.0.1:8000/analyze")


# ----------------------------
# Styling (clean + attractive)
# ----------------------------
CSS = r"""
/* ===== Page background ===== */
.gradio-container {
  min-height: 100vh !important;
  background:
    radial-gradient(1200px 700px at 10% 15%, rgba(99, 102, 241, 0.22), transparent 60%),
    radial-gradient(1200px 700px at 90% 20%, rgba(16, 185, 129, 0.16), transparent 62%),
    radial-gradient(900px 600px at 50% 90%, rgba(59, 130, 246, 0.16), transparent 60%),
    linear-gradient(180deg, #070A12, #060812 55%, #050615) !important;
}

/* Typography */
* {
  font-family: ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto, Helvetica, Arial, "Apple Color Emoji","Segoe UI Emoji";
}
a { color: #93C5FD; }
code {
  background: rgba(255,255,255,0.08);
  padding: 2px 6px;
  border-radius: 8px;
  border: 1px solid rgba(255,255,255,0.10);
}

/* Layout container */
#shell {
  max-width: 1180px;
  margin: 0 auto;
  padding: 22px 18px 28px 18px;
}

/* Smooth entrance animations */
@keyframes fadeUp {
  from { opacity: 0; transform: translateY(10px); filter: blur(4px); }
  to { opacity: 1; transform: translateY(0); filter: blur(0); }
}
.fadeUp { animation: fadeUp 520ms ease both; }

/* Cards (glass) */
.card {
  border-radius: 18px !important;
  background: rgba(255,255,255,0.06) !important;
  border: 1px solid rgba(255,255,255,0.10) !important;
  backdrop-filter: blur(12px);
  box-shadow: 0 16px 46px rgba(0,0,0,0.40);
}
.card-pad { padding: 14px 14px 12px 14px; }

/* Header */
#hero {
  padding: 18px 18px 14px 18px;
}
#hero-title {
  margin: 0;
  font-size: 28px;
  font-weight: 900;
  letter-spacing: 0.2px;
  color: #EEF2FF;
}
#hero-sub {
  margin-top: 8px;
  opacity: 0.85;
  font-size: 14px;
  color: #E5E7EB;
}
.badges {
  display: flex;
  flex-wrap: wrap;
  gap: 10px;
  margin-top: 14px;
}
.badge {
  padding: 7px 10px;
  border-radius: 999px;
  background: rgba(255,255,255,0.08);
  border: 1px solid rgba(255,255,255,0.12);
  font-size: 12px;
  color: #E5E7EB;
}

/* Sidebar/Main spacing */
#layout { margin-top: 14px; }
#sidebar, #mainpanel { gap: 12px; }

/* Inputs */
textarea, input {
  border-radius: 14px !important;
}
textarea:focus, input:focus {
  outline: none !important;
  box-shadow: 0 0 0 4px rgba(59,130,246,0.18) !important;
  border-color: rgba(59,130,246,0.55) !important;
}

/* Buttons */
#btn-analyze button {
  border-radius: 14px !important;
  font-weight: 800 !important;
  letter-spacing: 0.2px;
  transition: transform 0.12s ease, box-shadow 0.12s ease, filter 0.12s ease;
  box-shadow: 0 16px 36px rgba(59,130,246,0.26);
}
#btn-analyze button:hover {
  transform: translateY(-1px);
  filter: brightness(1.06);
}
#btn-analyze button:active {
  transform: translateY(1px) scale(0.99);
  box-shadow: 0 10px 24px rgba(59,130,246,0.18);
}

/* KPI cards */
.kpi-grid {
  display: grid;
  grid-template-columns: repeat(4, minmax(0,1fr));
  gap: 12px;
  margin-top: 6px;
}
.kpi {
  padding: 12px 12px 10px 12px;
  border-radius: 16px;
  background: rgba(255,255,255,0.06);
  border: 1px solid rgba(255,255,255,0.10);
}
.kpi .k {
  font-size: 12px;
  opacity: 0.78;
  color: #E5E7EB;
}
.kpi .v {
  margin-top: 6px;
  font-size: 14px;
  font-weight: 900;
  color: #F3F4F6;
  word-break: break-word;
}

/* Status pill */
.pill {
  display: inline-flex;
  align-items: center;
  gap: 8px;
  padding: 7px 10px;
  border-radius: 999px;
  font-size: 12px;
  font-weight: 800;
  border: 1px solid rgba(255,255,255,0.12);
  background: rgba(255,255,255,0.08);
}

/* Citations formatting */
.cite-item {
  padding: 12px;
  border-radius: 16px;
  background: rgba(255,255,255,0.06);
  border: 1px solid rgba(255,255,255,0.10);
  margin-bottom: 10px;
}
.cite-title {
  font-weight: 900;
  margin-bottom: 6px;
}
.cite-snippet {
  opacity: 0.9;
  white-space: pre-wrap;
}

/* Responsive */
@media (max-width: 980px) {
  .kpi-grid { grid-template-columns: repeat(2, minmax(0,1fr)); }
}
@media (max-width: 540px) {
  .kpi-grid { grid-template-columns: 1fr; }
}
"""


# ----------------------------
# Helpers
# ----------------------------
def _kpi_html(category: str, conf: Any, priority: str, found_in_kb: Any) -> str:
    conf_str = ""
    try:
        conf_str = f"{float(conf):.2f}"
    except Exception:
        conf_str = str(conf) if conf is not None else ""

    found_str = str(found_in_kb)

    return f"""
<div class="kpi-grid">
  <div class="kpi">
    <div class="k">Category</div>
    <div class="v">{category or "-"}</div>
  </div>
  <div class="kpi">
    <div class="k">Confidence</div>
    <div class="v">{conf_str or "-"}</div>
  </div>
  <div class="kpi">
    <div class="k">Priority</div>
    <div class="v">{priority or "-"}</div>
  </div>
  <div class="kpi">
    <div class="k">Found in KB</div>
    <div class="v">{found_str or "-"}</div>
  </div>
</div>
""".strip()


def _citations_html(citations: List[Dict[str, Any]]) -> str:
    if not citations:
        return '<div class="pill">No citations</div>'

    blocks = []
    for c in citations[:4]:
        doc_id = c.get("doc_id", "unknown")
        title = c.get("title", "KB")
        snippet = c.get("snippet", "")
        blocks.append(
            f"""
<div class="cite-item">
  <div class="cite-title">{title} <span style="opacity:.7; font-weight:700;">({doc_id})</span></div>
  <div class="cite-snippet">{snippet}</div>
</div>
""".strip()
        )
    return "\n".join(blocks)


def call_api(ticket_text: str) -> Tuple[str, str, str, Dict[str, Any]]:
    text = (ticket_text or "").strip()

    if len(text) < 10:
        return (
            _kpi_html("-", "-", "-", "-"),
            "Please enter a longer ticket message (10+ characters).",
            '<div class="pill">No citations</div>',
            {"error": "Ticket too short"},
        )

    # Small delay so the UI loading state feels smooth (optional)
    time.sleep(0.12)

    try:
        r = requests.post(API_URL, json={"text": text}, timeout=90)
        r.raise_for_status()
        data = r.json()
    except Exception as e:
        return (
            _kpi_html("Error", "-", "-", "-"),
            f"API error: {e}",
            '<div class="pill">No citations</div>',
            {"error": str(e)},
        )

    category = data.get("category", "")
    conf = data.get("category_confidence", "")
    priority = data.get("priority", "")
    reply = data.get("reply", "")
    found = data.get("found_in_kb", "")

    citations = data.get("citations", []) or []

    return (
        _kpi_html(category, conf, priority, found),
        reply,
        _citations_html(citations),
        data,
    )


# ----------------------------
# UI
# ----------------------------
theme = gr.themes.Soft(
    primary_hue="blue",
    neutral_hue="slate",
)

with gr.Blocks(theme=theme, css=CSS, title="Trusted Support Copilot") as demo:
    gr.HTML(
        """
        <div id="shell">
          <div id="hero" class="card fadeUp">
            <div id="hero-title">Trusted Customer Support Copilot</div>
            <div id="hero-sub">
              End-to-end ticket triage (ML) + grounded answers (RAG) with citations.
              Designed to reduce hallucinations with strict KB-only behavior.
            </div>
            <div class="badges">
              <span class="badge">FastAPI</span>
              <span class="badge">Gradio UI</span>
              <span class="badge">TF-IDF + LogisticRegression</span>
              <span class="badge">OpenAI Embeddings</span>
              <span class="badge">RAG + Citations</span>
            </div>
          </div>
        </div>
        """
    )

    with gr.Row(elem_id="layout"):
        # Sidebar
        with gr.Column(scale=5, elem_id="sidebar"):
            with gr.Group(elem_classes=["card", "fadeUp"]):
                gr.Markdown("### Ticket Input", elem_classes=["card-pad"])
                ticket = gr.Textbox(
                    lines=10,
                    label="",
                    placeholder="Paste a customer ticket here...\n\nExample: I see an unauthorized charge on my card. Please help.",
                )

                with gr.Row():
                    analyze_btn = gr.Button("Analyze", variant="primary", elem_id="btn-analyze")
                    clear_btn = gr.Button("Clear")

                gr.Markdown("#### Quick Tests", elem_classes=["card-pad"])
                gr.Examples(
                    examples=[
                        ["I see an unauthorized charge on my card. Please help."],
                        ["I was charged twice for my subscription. Can you refund the duplicate?"],
                        ["I can’t log in. Password reset doesn’t work and my account is locked."],
                        ["My delivery is delayed and tracking hasn’t updated."],
                    ],
                    inputs=[ticket],
                    label="",
                )

                gr.Markdown(
                    "Tip: Keep the API running at `http://127.0.0.1:8000`.",
                    elem_classes=["card-pad"],
                )

        # Main panel
        with gr.Column(scale=7, elem_id="mainpanel"):
            with gr.Group(elem_classes=["card", "fadeUp"]):
                gr.Markdown("### Overview", elem_classes=["card-pad"])
                kpis = gr.HTML(value=_kpi_html("-", "-", "-", "-"), elem_classes=["card-pad"])

            with gr.Group(elem_classes=["card", "fadeUp"]):
                with gr.Tabs():
                    with gr.TabItem("Draft Reply"):
                        reply = gr.Textbox(
                            lines=10,
                            label="Grounded reply (from KB)",
                            placeholder="Reply will appear here...",
                        )

                    with gr.TabItem("Citations"):
                        cites = gr.HTML(value='<div class="pill">No citations</div>')

                    with gr.TabItem("Raw JSON"):
                        raw = gr.JSON(value={})

    analyze_btn.click(
        fn=call_api,
        inputs=[ticket],
        outputs=[kpis, reply, cites, raw],
    )

    clear_btn.click(
        fn=lambda: (_kpi_html("-", "-", "-", "-"), "", '<div class="pill">No citations</div>', {}),
        inputs=[],
        outputs=[kpis, reply, cites, raw],
    )

if __name__ == "__main__":
    # If you want a share link: demo.launch(share=True)
    demo.launch()
