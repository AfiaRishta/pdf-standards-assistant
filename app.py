import os
import io
import json
import pickle
from pathlib import Path

import numpy as np
import streamlit as st
import fitz  # PyMuPDF
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer
from google import genai
from google.genai import types
import PIL.Image

DATA = Path("data")
DOCS = Path("docs")
MODEL = "gemini-3.6-flash"      # cheap, fast, reads images and text
TOP_K = 6                       # how many chunks to send to Gemini


#  load everything once
@st.cache_resource
def load_all():
    chunks = json.loads((DATA / "chunks.json").read_text(encoding="utf-8"))
    with open(DATA / "bm25.pkl", "rb") as f:
        bm25 = pickle.load(f)
    embeddings = np.load(DATA / "embeddings.npy")
    embedder = SentenceTransformer("all-MiniLM-L6-v2")
    client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
    return chunks, bm25, embeddings, embedder, client


# ---------- retrieval ----------
def retrieve(question, chunks, bm25, embeddings, embedder, k=TOP_K):
    # keyword scores
    bm_scores = bm25.get_scores(question.lower().split())
    bm_rank = np.argsort(bm_scores)[::-1][:10]

    # meaning scores
    q_vec = embedder.encode([question], normalize_embeddings=True)[0]
    em_scores = embeddings @ q_vec
    em_rank = np.argsort(em_scores)[::-1][:10]

    # merge: give each chunk points based on how high it ranked in each list
    points = {}
    for rank, idx in enumerate(bm_rank):
        points[idx] = points.get(idx, 0) + (10 - rank)
    for rank, idx in enumerate(em_rank):
        points[idx] = points.get(idx, 0) + (10 - rank)

    best = sorted(points, key=points.get, reverse=True)[:k]
    return [chunks[i] for i in best]


# ----------  ask Gemini ----------
def build_prompt(question, retrieved):
    blocks = []
    for i, c in enumerate(retrieved, 1):
        pages = ", ".join(str(p) for p in c["pages"])
        blocks.append(
            f"[SOURCE {i}]\nDocument: {c['doc']}\nClause: {c['clause']}\n"
            f"Page(s): {pages}\nText:\n{c['text']}\n"
        )
    sources = "\n".join(blocks)
    return f"""You are helping a registered building inspector. Answer the QUESTION using ONLY the SOURCES below. Do not use any outside knowledge.

Rules:
- The "quote" must be copied WORD FOR WORD from a source's Text.
- Use the page number given for that source.
- If the sources do not answer the question, set "found" to false and explain that it was not found in the supplied documents.
- Always name the document (including its year/edition).

Reply with JSON only, in exactly this shape:
{{
  "answer": "plain-English answer for the inspector",
  "found": true,
  "citations": [
    {{"document": "AS 2870-2011", "clause": "3.2.4", "page": 27, "quote": "exact text copied from a source"}}
  ]
}}

QUESTION: {question}

SOURCES:
{sources}
"""


def looks_like_table(chunk):
    h = chunk["heading"].lower()
    return h.startswith("table") or "table" in chunk["text"][:200].lower()


def ask_gemini(client, question, retrieved):
    prompt = build_prompt(question, retrieved)
    contents = [prompt]

    # for table chunks, also send the page image so Gemini can read the table visually
    seen = set()
    for c in retrieved:
        if looks_like_table(c) and c["pages"]:
            key = (c["doc_key"], c["pages"][0])
            if key in seen:
                continue
            seen.add(key)
            img_path = DATA / "images" / c["doc_key"] / f"p{c['pages'][0]:04d}.png"
            if img_path.exists():
                contents.append(PIL.Image.open(img_path))

    resp = client.models.generate_content(
        model=MODEL,
        contents=contents,
        config=types.GenerateContentConfig(response_mime_type="application/json"),
    )
    try:
        return json.loads(resp.text)
    except Exception:
        return {"answer": resp.text, "found": False, "citations": []}


# ----------  verify + highlight ----------
def find_pdf(document_name):
    """Match Gemini's document name back to a file in docs/."""
    for p in DOCS.glob("*.pdf"):
        stem = p.stem.lower()
        key = document_name.lower().replace(" ", "").replace("-", "")
        if key[:6] in stem.replace(" ", "").replace("-", "").replace("_", ""):
            return p
    return None


def verify_and_highlight(citation):
    """Returns (highlighted_png_bytes, verified_bool). None image if not verified."""
    pdf = find_pdf(citation.get("document", ""))
    quote = citation.get("quote", "").strip()
    page_no = citation.get("page")
    if not pdf or not quote or not page_no:
        return None, False

    doc = fitz.open(pdf)
    # search the cited page first, then its neighbours (page numbers can be off by one)
    candidates = [page_no - 1, page_no, page_no - 2]
    for pi in candidates:
        if pi < 0 or pi >= len(doc):
            continue
        page = doc[pi]
        rects = page.search_for(quote)
        if not rects:                                   # try the first 8 words
            short = " ".join(quote.split()[:8])
            if len(short) > 15:
                rects = page.search_for(short)
        if rects:
            for r in rects:
                page.add_highlight_annot(r)
            pix = page.get_pixmap(dpi=130)
            png = pix.tobytes("png")
            doc.close()
            return png, True
    doc.close()
    return None, False


# ---------- the screen ----------
st.set_page_config(page_title="PDF Standards Assistant", layout="wide")
st.title("GGBI Standards Assistant")
st.caption("Answers come only from the four uploaded standards. Every answer is checked against the page it came from.")

chunks, bm25, embeddings, embedder, client = load_all()

docs_loaded = sorted({c["doc"] for c in chunks})
with st.expander("Documents loaded"):
    for d in docs_loaded:
        st.write("•", d)

question = st.text_input("Ask a question about the standards",
                         placeholder="e.g. What is the minimum floor joist bearing length?")

if question:
    with st.spinner("Searching the documents..."):
        retrieved = retrieve(question, chunks, bm25, embeddings, embedder)
        result = ask_gemini(client, question, retrieved)

    st.subheader("Answer")
    st.write(result.get("answer", "(no answer)"))

    if not result.get("found", False):
        st.warning("Not found in the supplied documents.")

    for cit in result.get("citations", []):
        png, verified = verify_and_highlight(cit)
        label = f"{cit.get('document','?')} — clause {cit.get('clause','?')}, page {cit.get('page','?')}"
        if verified:
            st.success(f"Verified source: {label}")
            st.image(png, use_container_width=True)
        else:
            st.error(f"Unverified (quote not found on page): {label}")
            st.write(f"Claimed quote: “{cit.get('quote','')}”")
