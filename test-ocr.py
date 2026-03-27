from pathlib import Path
from typing import Optional
import fitz  # pymupdf
import easyocr
import numpy as np

_reader: Optional[easyocr.Reader] = None


def get_reader() -> easyocr.Reader:
    global _reader
    if _reader is None:
        print("🔄 Initializing EasyOCR reader...")
        _reader = easyocr.Reader(["th", "en"], gpu=True)
    return _reader


def is_scanned_pdf(pdf_path: Path, text_threshold: int = 20) -> bool:
    """
    ตรวจว่า PDF เป็นแบบ scanned หรือมี text layer อยู่แล้ว
    โดยนับจำนวนตัวอักษรใน 3 หน้าแรก
    """
    doc = fitz.open(str(pdf_path))
    sample_pages = min(3, len(doc))
    total_chars = sum(
        len(doc[i].get_text("text").strip())
        for i in range(sample_pages)
    )
    doc.close()

    print(f"🔍 Total characters (sample): {total_chars}")
    return total_chars < text_threshold  # น้อยเกินไป = scanned


def extract_text_pdf(pdf_path: Path) -> str:
    """
    ดึง text จาก PDF ที่มี text layer อยู่แล้ว (ไม่ต้อง OCR)
    """
    print("📄 Extracting text (no OCR)...")
    doc = fitz.open(str(pdf_path))
    pages_text = []

    for page_num, page in enumerate(doc, start=1):
        text = page.get_text("text").strip()
        if text:
            pages_text.append(f"--- Page {page_num} ---\n{text}")

    doc.close()
    return "\n\n".join(pages_text)


def ocr_pdf_with_easyocr(pdf_path: Path, dpi: int = 200) -> str:
    """
    Render แต่ละหน้าเป็น image แล้วให้ EasyOCR อ่าน
    ใช้ fitz render แทน pdf2image
    """
    print("🧠 Running OCR...")
    reader = get_reader()

    doc = fitz.open(str(pdf_path))
    matrix = fitz.Matrix(dpi / 72, dpi / 72)

    pages_text = []
    for page_num, page in enumerate(doc, start=1):
        print(f"➡️ OCR Page {page_num}")

        pix = page.get_pixmap(matrix=matrix, colorspace=fitz.csRGB)
        img_np = np.frombuffer(pix.samples, dtype=np.uint8).reshape(
            pix.height, pix.width, 3
        )

        results = reader.readtext(img_np, detail=0, paragraph=True)
        page_text = "\n".join(results)

        pages_text.append(f"--- Page {page_num} ---\n{page_text}")

    doc.close()
    return "\n\n".join(pages_text)


def process_pdf_per_page(pdf_path: Path) -> tuple[str, list[bool]]:
    """
    ตรวจทีละหน้า:
    - ถ้ามี text → extract
    - ถ้าไม่มี → OCR
    คืนค่า:
    - text รวมทั้งหมด
    - list บอกว่าแต่ละหน้าใช้ OCR หรือไม่
    """
    doc = fitz.open(str(pdf_path))
    reader = get_reader()

    pages_text = []
    ocr_flags = []

    for page_num, page in enumerate(doc, start=1):
        print(f"\n📄 Page {page_num}")

        if is_scanned_page(page):
            print("🧠 OCR this page")
            pix = page.get_pixmap(matrix=fitz.Matrix(200 / 72, 200 / 72), colorspace=fitz.csRGB)
            img_np = np.frombuffer(pix.samples, dtype=np.uint8).reshape(
                pix.height, pix.width, 3
            )

            results = reader.readtext(img_np, detail=0, paragraph=True)
            page_text = "\n".join(results)
            ocr_flags.append(True)
        else:
            print("📄 Extract text directly")
            page_text = page.get_text("text").strip()
            ocr_flags.append(False)

        pages_text.append(f"--- Page {page_num} ---\n{page_text}")

    doc.close()
    return "\n\n".join(pages_text), ocr_flags

def is_scanned_page(page, text_threshold: int = 20) -> bool:
    text = page.get_text("text").strip()
    return len(text) < text_threshold

# =========================
# 🔽 TEST SECTION
# =========================
if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("❌ Usage: python test-ocr.py <path_to_pdf>")
        sys.exit(1)

    pdf_file = Path(sys.argv[1])

    if not pdf_file.exists():
        print(f"❌ File not found: {pdf_file}")
        sys.exit(1)

    print(f"📂 Processing: {pdf_file}")

    text, ocr_flags = process_pdf_per_page(pdf_file)

    print("\n" + "=" * 50)
    print(f"✅ Used OCR: {ocr_flags}")
    print("=" * 50)

    print(text)  # แสดงบางส่วนกันยาวเกิน