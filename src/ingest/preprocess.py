import os
import re
import pandas as pd
from src.config import RAW_DIR, PROCESSED_DIR

PII_PATTERNS = [
    (re.compile(r"\b[\w\.-]+@[\w\.-]+\.\w+\b"), "[EMAIL]"),
    (re.compile(r"\b(?:\+?\d{1,3}[-.\s]?)?(?:\(?\d{2,3}\)?[-.\s]?)?\d{3,4}[-.\s]?\d{4}\b"), "[PHONE]"),
]

TEXT_COL_CANDIDATES = [
    "consumer_complaint_narrative",
    "Consumer complaint narrative",
    "Consumer Complaint Narrative",
]

LABEL_COL_CANDIDATES = [
    "product",
    "Product",
    "issue",
    "Issue",
]

# Map CFPB "Product" labels into realistic support categories
SUPPORT_MAP = {
    # Fraud / unauthorized
    "Credit card": "Fraud/Unauthorized",
    "Credit card or prepaid card": "Fraud/Unauthorized",
    "Prepaid card": "Fraud/Unauthorized",

    # Credit reporting
    "Credit reporting": "Credit Reporting",
    "Credit reporting or other personal consumer reports": "Credit Reporting",
    "Credit reporting, credit repair services, or other personal consumer reports": "Credit Reporting",

    # Banking
    "Checking or savings account": "Banking",
    "Bank account or service": "Banking",

    # Debt / collections
    "Debt collection": "Debt/Collections",
    "Debt or credit management": "Debt/Collections",

    # Loans
    "Student loan": "Loans",
    "Vehicle loan or lease": "Loans",
    "Payday loan": "Loans",
    "Payday loan, title loan, or personal loan": "Loans",
    "Payday loan, title loan, personal loan, or advance loan": "Loans",
    "Consumer Loan": "Loans",
    "Mortgage": "Loans",

    # Payments / transfers
    "Money transfer, virtual currency, or money service": "Payments/Transfers",
    "Money transfers": "Payments/Transfers",
}

def scrub_pii(text: str) -> str:
    if not isinstance(text, str):
        return ""
    t = text
    for pat, rep in PII_PATTERNS:
        t = pat.sub(rep, t)
    return t.strip()

def pick_col(df: pd.DataFrame, candidates):
    for c in candidates:
        if c in df.columns:
            return c
    return None

def preprocess(input_csv: str) -> str:
    df = pd.read_csv(input_csv, low_memory=False)

    text_col = pick_col(df, TEXT_COL_CANDIDATES)
    label_col = pick_col(df, LABEL_COL_CANDIDATES)

    if not text_col or not label_col:
        raise ValueError(
            f"Missing required columns.\n"
            f"Found columns: {list(df.columns)[:40]}\n"
            f"Expected one of text={TEXT_COL_CANDIDATES} and label={LABEL_COL_CANDIDATES}"
        )

    df = df[[text_col, label_col]].rename(columns={text_col: "text", label_col: "label"})

    df["text"] = df["text"].fillna("").astype(str).map(scrub_pii)
    df["label"] = df["label"].fillna("Unknown").astype(str)

    # Keep only rows with meaningful narratives
    df = df[df["text"].str.len() >= 30]

    # Map to support categories (unknown/rare -> Other)
    df["label"] = df["label"].map(lambda x: SUPPORT_MAP.get(x, "Other"))

    # Remove 'Other' from training (keep it as a fallback at inference time)
    df = df[df["label"] != "Other"]

    out_path = os.path.join(PROCESSED_DIR, "cfpb_clean.csv")
    df.to_csv(out_path, index=False)
    return out_path

if __name__ == "__main__":
    # pick newest csv in raw/
    files = [f for f in os.listdir(RAW_DIR) if f.lower().endswith(".csv") and "cfpb_sample" in f.lower()]
    if not files:
        files = [f for f in os.listdir(RAW_DIR) if f.lower().endswith(".csv")]
    if not files:
        raise SystemExit("No CSV found in data/raw. Run download script first.")

    files.sort()
    in_path = os.path.join(RAW_DIR, files[-1])
    out = preprocess(in_path)
    print("Saved:", out)
