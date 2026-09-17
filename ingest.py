import json
from pathlib import Path
import fitz  # PyMuPDF

DOCS = Path("docs")
DATA = Path("data")
IMAGES = DATA / "images"
DPI = 150
MIN_TEXT_CHARS = 50   # below this we assume the page is a scan and OCR it

# Optional OCR - only used if a page has no text layer
try:
    import pytesseract
    from PIL import Image
    pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
    OCR_AVAILABLE = True
except ImportError:
    OCR_AVAILABLE = False


def short_name(pdf_path: Path) -> str:
    """Turn 'AS_1684.2-2021-0.pdf' into a safe folder name."""
    return pdf_path.stem.replace(" ", "_").replace(".", "_")


def extract_blocks(page):
    """Return text blocks with their positions on the page."""
    blocks = []
    for b in page.get_text("dict")["blocks"]:
        if b.get("type") != 0:          # 0 = text, 1 = image
            continue
        text = " ".join(
            span["text"] for line in b["lines"] for span in line["spans"]
        ).strip()
        if text:
            x0, y0, x1, y1 = b["bbox"]
            blocks.append({"text": text, "bbox": [round(x0, 1), round(y0, 1), round(x1, 1), round(y1, 1)]})
    return blocks


def ocr_page(png_path: Path):
    """Step 5 - OCR a scanned page. Returns (text, blocks)."""
    if not OCR_AVAILABLE:
        return "", []
    img = Image.open(png_path)
    data = pytesseract.image_to_data(img, output_type=pytesseract.Output.DICT)
    scale = 72 / DPI   # convert image pixels back to PDF points so bboxes match PyMuPDF
    words, blocks = [], []
    for i, w in enumerate(data["text"]):
        if w.strip():
            words.append(w)
            x, y, wd, ht = data["left"][i], data["top"][i], data["width"][i], data["height"][i]
            blocks.append({"text": w, "bbox": [round(x*scale,1), round(y*scale,1), round((x+wd)*scale,1), round((y+ht)*scale,1)]})
    return " ".join(words), blocks


def main():
    DATA.mkdir(exist_ok=True)
    IMAGES.mkdir(exist_ok=True)
    records = []

    for pdf_path in sorted(DOCS.glob("*.pdf")):
        name = short_name(pdf_path)
        out_dir = IMAGES / name
        out_dir.mkdir(exist_ok=True)
        doc = fitz.open(pdf_path)
        print(f"\n{pdf_path.name}  ({len(doc)} pages)")

        ocr_count = 0
        for idx, page in enumerate(doc):
            page_no = idx + 1

            # Save the page picture
            png_path = out_dir / f"p{page_no:04d}.png"
            if not png_path.exists():
                page.get_pixmap(dpi=DPI).save(png_path)

            # Get the words and where they are
            text = page.get_text().strip()
            blocks = extract_blocks(page)
            used_ocr = False

            if len(text) < MIN_TEXT_CHARS:          # Step 5 - scanned page
                text, blocks = ocr_page(png_path)
                used_ocr = True
                ocr_count += 1

            records.append({
                "doc": pdf_path.name,
                "doc_key": name,
                "page": page_no,
                "text": text,
                "blocks": blocks,
                "image": str(png_path).replace("\\", "/"),
                "ocr": used_ocr,
                "width": round(page.rect.width, 1),
                "height": round(page.rect.height, 1),
            })

            if page_no % 25 == 0:
                print(f"   ... page {page_no}")

        print(f"   done. pages needing OCR: {ocr_count}")
        doc.close()

    with open(DATA / "pages.json", "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False)

    print(f"\nSaved {len(records)} pages to data/pages.json")
    if not OCR_AVAILABLE:
        print("Note: pytesseract not installed - scanned pages (if any) were left empty.")


if __name__ == "__main__":
    main()
