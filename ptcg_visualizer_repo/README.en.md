# PTCGABCSVisualizer

[日本語](README.md) ・ **English**

A single-file, dependency-free, **offline** replay viewer (pure HTML + Vanilla JS) for the **PTCG AI Battle Challenge** (cabt engine). It loads the JSON produced by `visualize_data()` and steps through a game with a full board view. Open `index.html` directly via `file://` — no build, no server.

> ⚠️ **Card images are NOT bundled** (IP + size). You generate them locally from the official PDFs (→ [Card image setup](#card-image-setup)). Without images the tool still works, showing card name / HP / energy as text.

## Features

- God's-eye board: both players' active / bench / hand, deck/prize/discard counts, stadium.
- Active-turn player highlight; **hover any card for a large preview** (with attached energy & tools).
- Click the **Discard** count to open a **scrollable grid** of the whole discard pile.
- Right pane: the action chosen on that step plus the **decoded log** (with card names & game result).
- **EN / JP toggle**: switches UI labels, log text, **and the card images** (choice is persisted).

## Quick start

1. Clone / download this repository.
2. Open `index.html` in a browser (double-click is fine).
3. **Drag & drop** a replay JSON, or use "Open file".
   - To try it instantly, use a bundled sample: `samples/MegaLucario_vs_MegaAbomasnow.json` (short; stadium & energy) or `samples/MegaLucario_vs_BeginnerGuide.json` (has a Tool).
4. Step with **◀ Prev / Next ▶** (or ←/→ keys) and the slider.

At this point card art is not shown yet (text view). To get the images, continue below.

## Card image setup

Card art is extracted from the JPEGs embedded inside the official PDFs (`Card_ID List_EN.pdf` / `Card_ID List_JP.pdf`). This repo ships neither the PDFs nor the images, so set them up yourself:

### 1. Get the official PDFs and place them

From the official PTCG AI Battle Challenge (Kaggle) distribution, obtain:

- `Card_ID List_EN.pdf` (English card art)
- `Card_ID List_JP.pdf` (Japanese card art)

Put them in the **same folder as `index.html` (the repo root)**:

```
PTCGABCSVisualizer/
├── index.html
├── _extract_card_images.py
├── Card_ID List_EN.pdf   ← place here
└── Card_ID List_JP.pdf   ← place here
```

### 2. Install Python + PyMuPDF

- Python 3.8+ (check with `python --version`)
- Install **PyMuPDF** (it provides the `fitz` module used by `import fitz`):

  ```bash
  python -m pip install pymupdf
  ```

  > ⚠️ The package is **`pymupdf`** (do NOT install the unrelated `fitz` package). Using **`python -m pip`** (not bare `pip`) installs into the same Python you run the script with. Verify: `python -c "import fitz; print('ok')"`

### 3. Run the extractor

Run from the repo root (where the script lives):

```bash
python _extract_card_images.py --lang en   # English  -> assets/cards/     (1267 images)
python _extract_card_images.py --lang jp   # Japanese -> assets/cards_jp/  (1267 images)
```

Resulting layout:

```
assets/
├── cards/      # EN: 1.jpg .. 1267.jpg
└── cards_jp/   # JP: 1.jpg .. 1267.jpg
```

Options:

| Option                    | Description                                                |
| ------------------------- | ---------------------------------------------------------- |
| `--lang en` / `--lang jp` | Which PDF to extract (default `en`)                        |
| `--limit N`               | Extract only the first N cards (quick test)                |
| `--shrink 0`              | Full resolution (660×920). Default is 1 (halved, ~434×606) |

> Rough numbers: ~1 min per language for a full extract; `assets/cards_jp` is ~100 MB (larger with `--shrink 0`).

### 4. Done

Reload `index.html` — card art now appears. The language button (top-right) switches the UI **and the card image set** between EN and JP.

### Troubleshooting

- `ModuleNotFoundError: No module named 'fitz'` → PyMuPDF isn't installed, or it's in a **different Python** than the one you run. Install it into the same interpreter:

  ```bash
  python -m pip install pymupdf
  python -c "import fitz; print('ok')"      # prints "ok" when ready
  ```

  On Windows, if `python` isn't found, use the launcher `py -3` (e.g. `py -3 -m pip install pymupdf` → `py -3 _extract_card_images.py --lang en`).

- The package is **`pymupdf`**. Do NOT `pip install fitz` — that's a different, unrelated package.

## Language (EN / JP)

- The header language button switches **UI labels, log text, and card images** between English and Japanese (saved to `localStorage`; default is Japanese).
- ⚠️ **Card name text** comes from the replay JSON (English), so names in the log stay English even in JP mode — only the **card images** switch.

## Input format

- Input is a **JSON array of snapshots** (one per decision) emitted by the cabt engine's `visualize_data()`.
- Each element has: `select` (the choice) / `logs` (events so far) / `current` (board state — **god view**, both players' hands/deck/prizes revealed) / `selected` (the chosen indices).
- ⚠️ In `visualize_data()` enums are serialized as **strings** (log `type`="Draw", `select.type`="Main", …), while `area` / `energies` stay **integers**. The decoder assumes this.

## Repository layout

```
PTCGABCSVisualizer/
├── index.html                  # the app (open this)
├── _extract_card_images.py     # card-image extractor (requires PyMuPDF)
├── README.md / README.en.md    # docs (Japanese / English)
├── .gitignore                  # PDFs & generated images are not committed
├── samples/                    # bundled replay JSON for a quick try
├── (Card_ID List_EN/JP.pdf)    # you provide; not committed
└── assets/cards{,_jp}/         # generated by the extractor; not committed
```

## Notes / limitations

- Works from `file://` (no server, no `fetch`).
- Cards without an image fall back to **name / HP / energy text**.
- Attacks are shown by ID, not name (no attack-name dictionary bundled).
- Some browsers scope `localStorage` per file path.

## Credits & disclaimer

- Pokémon card images and `Card_ID List_*.pdf` are © The Pokémon Company / Nintendo / Creatures / GAME FREAK. **This repository does not bundle or redistribute them.** Extract them locally from files you obtain officially.
- This is an unofficial tool, not affiliated with or endorsed by the competition organizers.
- **This software is provided without any warranty; the author accepts no responsibility or liability whatsoever for any damage, loss, or trouble arising from its use. Use at your own risk.**
