PDF Standards Assistant

A citation-grounded question-answering system for technical PDFs. It retrieves the relevant clauses for a question, has an LLM answer using only those clauses, then verifies the model's quote against the source page before showing it highlighting the exact text on the page. If a quote can't be confirmed, the answer is flagged as unverified. If the documents don't cover the question, it says so instead of guessing.

The principle behind the whole system: the AI reads, the code checks, the human decides.

The problem

Large language models are confident even when they're wrong. That's a dealbreaker when your source is a regulation, a standard, or any document where a made-up answer has real consequences. Standard retrieval-augmented generation (RAG) helps by giving the model relevant text but nothing stops the model from misquoting it, citing the wrong page, or smoothly inventing a detail.

This project adds a verification layer on top of retrieval: every answer must be backed by a quote that is programmatically confirmed to exist on the cited page. The model is never the final authority on a fact deterministic code is.

How it works
<img width="437" height="781" alt="image" src="https://github.com/user-attachments/assets/a30a7f54-a8b5-49d4-81c0-8a48285db19f" />

1 · Retrieve. For a question, the system scores every chunk with two methods and merges the results: BM25 keyword search (wins on exact codes and identifiers) and semantic embeddings (wins on reworded questions that share no keywords). The top few clauses go forward.

2 · Answer. Only those clauses are sent to the LLM, with strict instructions: answer using this text alone, copy out the exact sentence relied on, name the source, and say "not found" if the text doesn't support an answer. Table pages are also sent as images so the model reads them visually rather than trusting imperfect OCR.

3 · Verify. Before anything is shown, the system reopens the real PDF, searches the cited page for the model's exact quote, and highlights it. If the exact quote isn't found, it retries with a shorter fragment; if it still can't be confirmed, the citation is discarded and the answer is marked unverified. The model cannot claim a source it can't prove.

4 · Show. The answer appears with its clause number, page, and the highlighted page image — so a reader can verify it in seconds.

Design decisions

The interesting part of this project is why it's built this way:

Hybrid retrieval, not embeddings alone. Semantic search misses exact codes like Class M or MGP10; keyword search misses paraphrased questions. Technical documents contain both, so the system uses both and fuses the rankings.
Chunk by clause, not by fixed length. A clause (or table, or figure) is the natural unit of a standard. Splitting mid-clause or mid-table destroys both meaning and the ability to cite cleanly, so headings drive the chunk boundaries.
Verify in code, don't trust the model. This is the core idea. Grounding an answer in retrieved text isn't enough if the model can still misquote it. Confirming the quote against the page converts "the AI says so" into "the source says so, and here it is."
Explicit "not found." Refusing to answer beyond the documents is a feature. In a high-stakes setting, a confident wrong answer is worse than an honest "I don't know."
Send tables as images. OCR mangles multi-column tables; a multimodal model reads the image far more reliably.
Tech stack
Layer	Tool	Role
PDF parsing & highlighting	PyMuPDF (fitz)	text, block positions, page images, annotations
OCR	Tesseract + pytesseract	read scanned pages
Keyword search	rank_bm25	exact-term retrieval
Semantic search	sentence-transformers (all-MiniLM-L6-v2)	meaning-based retrieval
Vectors	numpy	store and compare embeddings
LLM	multimodal LLM via API	read clauses, produce grounded answers
UI	Streamlit	the web app
Language	Python	everything

Everything except the LLM call runs locally. Only the short, relevant excerpts for a given question are ever sent to the API.

Running it
bash
# 1. install dependencies
pip install -r requirements.txt

# 2. add one or more PDFs to the docs/ folder
#    (any text-based or scanned PDF works)

# 3. set your LLM API key as an environment variable
#    (PowerShell)  $env:LLM_API_KEY = "your-key"
#    (bash)        export LLM_API_KEY="your-key"

# 4. extract the pages, then build the search indexes
python ingest.py
python build_index.py

# 5. launch the app
streamlit run app.py

Then open the local URL Streamlit prints, and ask a question about your documents.

Note: the docs/ and data/ folders are git-ignored, so no source documents or extracted content are ever committed. Bring your own PDF to try it.

Limitations

Stated honestly, because knowing the edges matters:

Scanned documents have imperfect text layers, so answers drawn from them can be correct but harder to highlight precisely — these surface as unverified. Better OCR or a clean digital source resolves it.
Complex tables with many conditions (e.g. multiple variables per row) are the hardest case; the system reads the table as an image and asks the model to state the row and column used, but full robustness here is future work.
Cross-document questions are supported by retrieving across all documents, but multi-hop reasoning is the next area to strengthen.
Non-determinism: an LLM may phrase a quote slightly differently between runs, so the same question can verify one run and not the next. Page-neighbour and fragment fallbacks reduce this; normalised fuzzy matching would reduce it further.
What's next
Narrow to one document type, measure accuracy against a known answer set, then widen.
A human review step so every answer is confirmed before it's used downstream.
Extending from question-answering to generating full, citation-backed reports.

Built as a learning project exploring retrieval-augmented generation, OCR on scanned documents, and most of all why grounding and verification matter more than raw model quality.
