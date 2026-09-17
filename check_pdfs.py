from pathlib import Path
import fitz  # this is PyMuPDF

DOCS = Path("docs")

for pdf_path in sorted(DOCS.glob("*.pdf")):
    doc = fitz.open(pdf_path)
    total_pages = len(doc)

    # Sample a few pages spread through the document, not just the cover
    sample_pages = sorted(set([0, min(19, total_pages - 1), total_pages // 2, total_pages - 1]))

    pages_with_text = 0
    for i in sample_pages:
        text = doc[i].get_text().strip()
        if len(text) > 200:      # a real page of a standard has far more than 200 characters
            pages_with_text += 1

    if pages_with_text == len(sample_pages):
        verdict = "TEXT LAYER OK  - no OCR needed"
    elif pages_with_text == 0:
        verdict = "SCANNED        - needs OCR"
    else:
        verdict = "MIXED          - some pages need OCR"

    print(f"{pdf_path.name}")
    print(f"   pages: {total_pages}   sampled: {[p+1 for p in sample_pages]}   {verdict}")

    # Show a preview of page 20 so you can eyeball it
    preview_page = min(19, total_pages - 1)
    preview = doc[preview_page].get_text().strip().replace("\n", " ")[:300]
    print(f"   page {preview_page+1} preview: {preview!r}\n")
    doc.close() 
