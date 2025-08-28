import argparse
import os
from pathlib import Path
from typing import Iterable, List
import numpy as np

import cv2
from PIL import Image
import pytesseract

def preprocess_for_ocr(pil_img: Image.Image) -> Image.Image:
    """Lightweight preprocessing that usually boosts OCR accuracy."""
    # Convert PIL -> OpenCV
    cv_img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
    # Grayscale
    gray = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)
    # Binarize (adaptive threshold helps with uneven lighting)
    thr = cv2.adaptiveThreshold(gray, 255,
                                cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                cv2.THRESH_BINARY, 31, 10)
    # Optional: slight dilation to connect characters
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 1))
    proc = cv2.morphologyEx(thr, cv2.MORPH_OPEN, kernel, iterations=1)
    # Back to PIL
    return Image.fromarray(proc)

def image_paths_from(source: Path) -> Iterable[Path]:
    exts = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}
    if source.is_file() and source.suffix.lower() in exts:
        yield source
    elif source.is_dir():
        for p in sorted(source.rglob("*")):
            if p.suffix.lower() in exts:
                yield p
    else:
        raise ValueError(f"Unsupported source: {source}")

def ocr_image(p: Path, lang: str = "eng", preprocess: bool = True) -> str:
    img = Image.open(p)
    if preprocess:
        try:
            import numpy as np  # lazy import to keep optional
        except ImportError:
            raise SystemExit("Please `pip install numpy opencv-python` for preprocessing, "
                             "or rerun with --no-preprocess")
        img = preprocess_for_ocr(img)

    # Point to where Homebrew installed tesseract
    pytesseract.pytesseract.tesseract_cmd = "/opt/homebrew/bin/tesseract"  # for M1/M2 Macs
    # or, if Intel Mac:
    # pytesseract.pytesseract.tesseract_cmd = "/usr/local/bin/tesseract"
    text = pytesseract.image_to_string(img, lang=lang)
    return text

def save_text(out_dir: Path, img_path: Path, text: str) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / (img_path.stem + ".txt")
    out_path.write_text(text, encoding="utf-8")
    return out_path

def main():
    parser = argparse.ArgumentParser(description="Convert image(s) to text via Tesseract OCR.")
    parser.add_argument("source", type=Path, help="Image file or folder of images")
    parser.add_argument("--lang", default="eng", help="Tesseract language code (e.g., eng, deu, fra)")
    parser.add_argument("--out", type=Path, default=Path("output"), help="Output directory for .txt files")
    parser.add_argument("--no-preprocess", action="store_true", help="Disable image preprocessing")
    args = parser.parse_args()

    failures: List[str] = []
    for img_path in image_paths_from(args.source):
        try:
            text = ocr_image(img_path, args.lang, preprocess=not args.no_preprocess)
            out_file = save_text(args.out, img_path, text)
            print(f"[OK] {img_path} -> {out_file}")
        except Exception as e:
            print(f"[FAIL] {img_path}: {e}")
            failures.append(str(img_path))

    if failures:
        print("\nSome files failed:")
        for f in failures: print(" -", f)

if __name__ == "__main__":
    main()
