"""
Day 2, Steps 6 and 7 - turn the extracted pages into searchable chunks.

What it does:
  - reads data/pages.json (made by ingest.py)
  - splits each document into "chunks", one per clause / table / figure
  - builds two search indexes over those chunks:
        BM25       = keyword search (good for codes like "Class M", "MGP10")
        embeddings = meaning search (good for reworded questions)
  - saves everything into data/ for the app to use

Run:  python build_index.py
"""
import json
import re
import pickle
from pathlib import Path

import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

DATA = Path("data")

# A line is treated as a NEW clause heading if it looks like:
#   "3.2.4 Something"      (numbered clause)
#   "Table 5.10 ..."       (a table)
#   "Figure 8.3 ..."       (a figure)
HEADING = re.compile(r"^\s*((\d+(\.\d+){1,4})|((Table|Figure|Appendix)\s+[A-Z]?\d))\b", re.IGNORECASE)

MAX_WORDS = 800   # if a chunk is longer than this, split it but keep the clause id


def clause_id(line: str) -> str:
    """Pull the clause/table number out of a heading line."""
    m = HEADING.match(line)
    return m.group(1).strip() if m else ""


def split_long(text: str):
    """Break a very long chunk into pieces of <= MAX_WORDS words."""
    words = text.split()
    if len(words) <= MAX_WORDS:
        return [text]
    return [" ".join(words[i:i + MAX_WORDS]) for i in range(0, len(words), MAX_WORDS)]


def chunk_document(pages):
    """Turn all pages of ONE document into clause chunks."""
    chunks = []
    cur = {"clause": "", "heading": "", "lines": [], "pages": set()}

    def flush():
        text = "\n".join(cur["lines"]).strip()
        if not text:
            return
        for piece in split_long(text):
            chunks.append({
                "clause": cur["clause"],
                "heading": cur["heading"],
                "text": piece,
                "pages": sorted(cur["pages"]),
            })

    for pg in pages:
        page_no = pg["page"]
        for line in pg["text"].split("\n"):
            if HEADING.match(line):          # a new clause starts here
                flush()
                cur = {"clause": clause_id(line), "heading": line.strip()[:120],
                       "lines": [line], "pages": {page_no}}
            else:
                cur["lines"].append(line)
                cur["pages"].add(page_no)
    flush()
    return chunks


def main():
    pages = json.loads((DATA / "pages.json").read_text(encoding="utf-8"))

    # group pages by document, keep page order
    by_doc = {}
    for p in pages:
        by_doc.setdefault(p["doc"], []).append(p)
    for doc in by_doc:
        by_doc[doc].sort(key=lambda p: p["page"])

    # chunk every document
    all_chunks = []
    for doc, doc_pages in by_doc.items():
        doc_key = doc_pages[0]["doc_key"]
        c = chunk_document(doc_pages)
        for ch in c:
            ch["doc"] = doc
            ch["doc_key"] = doc_key
        all_chunks.extend(c)
        print(f"{doc}: {len(c)} chunks")

    print(f"\nTotal chunks: {len(all_chunks)}")

    # a searchable string for each chunk (heading helps matching)
    texts = [f"{c['doc']} {c['heading']} {c['text']}" for c in all_chunks]

    # ---- BM25 keyword index ----
    tokenized = [t.lower().split() for t in texts]
    bm25 = BM25Okapi(tokenized)

    # ---- embedding (meaning) index ----
    print("Loading embedding model (first run downloads ~90 MB)...")
    model = SentenceTransformer("all-MiniLM-L6-v2")
    print("Encoding chunks...")
    embeddings = model.encode(texts, show_progress_bar=True, normalize_embeddings=True)

    # ---- save everything ----
    (DATA / "chunks.json").write_text(json.dumps(all_chunks, ensure_ascii=False), encoding="utf-8")
    with open(DATA / "bm25.pkl", "wb") as f:
        pickle.dump(bm25, f)
    np.save(DATA / "embeddings.npy", embeddings)

    print(f"\nSaved chunks.json, bm25.pkl, embeddings.npy to data/")


if __name__ == "__main__":
    main()