import os
import time
import zipfile
import requests
import pandas as pd
from tqdm import tqdm

from src.config import RAW_DIR

CSV_ZIP_URL = "https://files.consumerfinance.gov/ccdb/complaints.csv.zip"  # official ZIP :contentReference[oaicite:1]{index=1}

def download_with_resume(url: str, dest_path: str, max_retries: int = 30) -> None:
    os.makedirs(os.path.dirname(dest_path), exist_ok=True)

    for attempt in range(1, max_retries + 1):
        existing = os.path.getsize(dest_path) if os.path.exists(dest_path) else 0
        headers = {"Range": f"bytes={existing}-"} if existing > 0 else {}

        try:
            with requests.get(url, stream=True, timeout=120, headers=headers) as r:
                # 200 = full file, 206 = partial content (resume)
                if r.status_code not in (200, 206):
                    r.raise_for_status()

                # total remaining bytes (content-length is remaining when using Range)
                remaining = int(r.headers.get("content-length", 0))
                total = existing + remaining if remaining else None

                mode = "ab" if existing > 0 else "wb"
                desc = f"Downloading (resume @ {existing/1024/1024:.1f} MB)"

                with open(dest_path, mode) as f, tqdm(
                    total=total,
                    initial=existing,
                    unit="B",
                    unit_scale=True,
                    desc=desc,
                ) as pbar:
                    for chunk in r.iter_content(chunk_size=1024 * 1024):
                        if chunk:
                            f.write(chunk)
                            pbar.update(len(chunk))

            # quick integrity check: can we open as zip?
            with zipfile.ZipFile(dest_path, "r") as z:
                _ = z.namelist()
            return

        except Exception as e:
            wait = min(60, 2 * attempt)
            print(f"[Attempt {attempt}/{max_retries}] Download failed: {e}")
            print(f"Retrying in {wait}s... (will resume)")
            time.sleep(wait)

    raise RuntimeError("Download failed after many retries. Try Option A with curl.exe.")

def download_and_sample(sample_rows: int = 50000) -> str:
    zip_path = os.path.join(RAW_DIR, "complaints.csv.zip")
    out_csv = os.path.join(RAW_DIR, f"cfpb_sample_{sample_rows}.csv")

    download_with_resume(CSV_ZIP_URL, zip_path)

    # Extract CSV
    with zipfile.ZipFile(zip_path, "r") as z:
        csv_name = [n for n in z.namelist() if n.lower().endswith(".csv")][0]
        extracted_csv_path = os.path.join(RAW_DIR, csv_name)
        if not os.path.exists(extracted_csv_path):
            z.extract(csv_name, path=RAW_DIR)

    # Sample in chunks (keep rows that have narratives)
    narrative_col_candidates = [
        "consumer_complaint_narrative",
        "Consumer complaint narrative",
        "Consumer Complaint Narrative",
    ]

    kept = []
    kept_count = 0

    for chunk in pd.read_csv(extracted_csv_path, chunksize=20000, low_memory=False):
        narrative_col = next((c for c in narrative_col_candidates if c in chunk.columns), None)

        if narrative_col:
            chunk = chunk[chunk[narrative_col].notna() & (chunk[narrative_col].astype(str).str.len() > 30)]

        if len(chunk) == 0:
            continue

        remaining = sample_rows - kept_count
        if remaining <= 0:
            break

        chunk = chunk.head(remaining)
        kept.append(chunk)
        kept_count += len(chunk)

        if kept_count >= sample_rows:
            break

    if not kept:
        raise RuntimeError("No rows collected. Column names may have changed.")

    df = pd.concat(kept, ignore_index=True)
    df.to_csv(out_csv, index=False)
    return out_csv

if __name__ == "__main__":
    path = download_and_sample(sample_rows=50000)
    print("Saved sample CSV:", path)
