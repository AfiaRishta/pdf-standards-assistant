"""
Day 3 - run a list of test questions automatically and save a scorecard.

It reuses everything from app.py (retrieval, Gemini answer, verification),
runs each question in questions.txt, and writes scorecard.csv.

Run:  python evaluate.py
(Needs GEMINI_API_KEY set in the same terminal, same as the app.)

Open scorecard.csv in Excel afterwards and fill in the last columns by hand.
"""
import csv
import time
from pathlib import Path

# borrow the functions already written in app.py
from app import (
    retrieve, ask_gemini, verify_and_highlight,
    chunks, bm25, embeddings, embedder, client,
)

QUESTIONS_FILE = Path("questions.txt")
OUT = Path("scorecard.csv")


def run():
    questions = [q.strip() for q in QUESTIONS_FILE.read_text(encoding="utf-8").splitlines()
                 if q.strip() and not q.strip().startswith("#")]

    rows = []
    for i, q in enumerate(questions, 1):
        print(f"[{i}/{len(questions)}] {q}")
        try:
            retrieved = retrieve(q, chunks, bm25, embeddings, embedder)
            result = ask_gemini(client, q, retrieved)
        except Exception as e:
            print(f"    error: {e}")
            rows.append({"question": q, "answer": f"ERROR: {e}", "found": "",
                         "citation": "", "highlight_verified": ""})
            time.sleep(3)
            continue

        answer = result.get("answer", "")
        found = result.get("found", False)

        # take the first citation and check whether its highlight verified
        citations = result.get("citations", [])
        cite_str, verified = "", ""
        if citations:
            c = citations[0]
            cite_str = f"{c.get('document','?')} clause {c.get('clause','?')} p{c.get('page','?')}"
            _, ok = verify_and_highlight(c)
            verified = "YES" if ok else "NO"

        rows.append({
            "question": q,
            "answer": answer,
            "found": "found" if found else "NOT FOUND",
            "citation": cite_str,
            "highlight_verified": verified,
        })
        time.sleep(2)   # be gentle on the API

    # write the CSV with blank columns for you to fill in
    with open(OUT, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "Question", "System answer", "Found?", "Citation", "Highlight verified?",
            "Answer correct? (you fill)", "Citation correct? (you fill)", "Notes (you fill)",
        ])
        for r in rows:
            writer.writerow([
                r["question"], r["answer"], r["found"], r["citation"], r["highlight_verified"],
                "", "", "",
            ])

    print(f"\nDone. Saved {len(rows)} results to {OUT}")
    print("Open scorecard.csv in Excel and fill in the last three columns.")


if __name__ == "__main__":
    run()