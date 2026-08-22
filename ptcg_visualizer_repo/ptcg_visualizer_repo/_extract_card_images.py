"""Extract per-card images from "Card_ID List_{EN,JP}.pdf" into ./assets/cards[_jp]/.

The PDF is an index table whose "View Image" cells are internal GOTO links to a
per-card page that embeds one JPEG. We map each row's Card ID -> target page via
the link's vertical position, then export (optionally downscaled) <cardId>.jpg.

Place "Card_ID List_EN.pdf" / "Card_ID List_JP.pdf" next to this script (or one
level up), then run from the folder that holds this script + index.html:

  python _extract_card_images.py --lang en   # EN -> assets/cards (default)
  python _extract_card_images.py --lang jp   # JP -> assets/cards_jp
  python _extract_card_images.py --limit 3   # test a few
  python _extract_card_images.py --shrink 0  # full resolution (660x920)
"""
import argparse
import re
from pathlib import Path

try:
    import pymupdf as fitz          # PyMuPDF >= 1.24.3 (current import name)
except ImportError:
    try:
        import fitz                 # PyMuPDF (legacy import name)
    except ImportError:
        raise SystemExit(
            "PyMuPDF is required but not installed.\n"
            "  Install it with:  python -m pip install pymupdf\n"
            "  (Install the 'pymupdf' package - NOT the unrelated 'fitz' package.)"
        )

HERE = Path(__file__).resolve().parent            # folder holding this script + index.html
PDF_NAMES = {"en": "Card_ID List_EN.pdf", "jp": "Card_ID List_JP.pdf"}
OUTS = {"en": HERE / "assets" / "cards", "jp": HERE / "assets" / "cards_jp"}


def find_pdf(name: str) -> Path:
    """Locate the source PDF: next to this script (standalone repo) or one level up (monorepo)."""
    for d in (HERE, HERE.parent):
        if (d / name).exists():
            return d / name
    return HERE / name


def build_mapping(doc) -> dict:
    """cardId -> target page index (the page that embeds that card's image)."""
    mapping = {}
    for pno in range(doc.page_count):
        pg = doc[pno]
        links = pg.get_links()
        if not links:
            continue
        words = pg.get_text("words")  # (x0,y0,x1,y1,text,...)
        id_words = [(w[1], w[3], w[4]) for w in words
                    if re.fullmatch(r"\d+", w[4]) and w[0] < 150]  # Card ID column (left)
        if not id_words:
            continue
        for l in links:
            if l["kind"] != fitz.LINK_GOTO:
                continue
            r = l["from"]
            if r.x0 < 400:          # only the right-column "View Image" links
                continue
            ymid = (r.y0 + r.y1) / 2
            best, bestd = None, 1e9
            for (y0, y1, txt) in id_words:
                d = abs((y0 + y1) / 2 - ymid)
                if d < bestd:
                    bestd, best = d, txt
            if best is not None and bestd < 8:
                mapping[int(best)] = l["page"]
    return mapping


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lang", choices=["en", "jp"], default="en", help="which Card_ID List PDF to extract")
    ap.add_argument("--limit", type=int, default=0, help="extract only the first N cards (0=all)")
    ap.add_argument("--shrink", type=int, default=1, help="halve dimensions N times (0=full res)")
    args = ap.parse_args()

    PDF, OUT = find_pdf(PDF_NAMES[args.lang]), OUTS[args.lang]
    if not PDF.exists():
        raise SystemExit(f"PDF not found: '{PDF_NAMES[args.lang]}'. "
                         f"Place it next to this script (or one level up). See README.")
    print(f"lang={args.lang}  pdf={PDF}  out={OUT}")
    doc = fitz.open(PDF)
    mapping = build_mapping(doc)
    ids = sorted(mapping)
    print(f"mapped {len(mapping)} cards (id range {ids[0]}..{ids[-1]})")

    OUT.mkdir(parents=True, exist_ok=True)
    todo = ids[:args.limit] if args.limit else ids
    total_bytes = 0
    written = 0
    for cid in todo:
        imgs = doc[mapping[cid]].get_images(full=True)
        if not imgs:
            continue
        pix = fitz.Pixmap(doc, imgs[0][0])
        if pix.alpha:                      # drop alpha for jpeg
            pix = fitz.Pixmap(fitz.csRGB, pix)
        for _ in range(args.shrink):
            pix.shrink(1)                  # halve once per call
        data = pix.tobytes("jpeg", jpg_quality=82)
        (OUT / f"{cid}.jpg").write_bytes(data)
        total_bytes += len(data)
        written += 1
    print(f"wrote {written} images to {OUT}")
    print(f"size: {total_bytes/1e6:.1f} MB  | avg {total_bytes/max(1,written)/1024:.1f} KB"
          f"  | dims now {pix.width}x{pix.height}")


if __name__ == "__main__":
    main()
