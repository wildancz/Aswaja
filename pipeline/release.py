#!/usr/bin/env python3
"""Packs the public export into a Hugging Face dataset folder: Parquet shards (train/test by family),
books metadata, and a dataset card. Needs pyarrow.

Input: KDS_PUBLIC (default <root>/round1/public) with books.jsonl and pages/<book_id>.jsonl.gz;
<root> = KDS_ROOT, default the folder above this script.

  release.py <version>   ->  <root>/release/<version>/
        README.md                      dataset card (HF YAML header)
        data/train-NNNNN.parquet       pages, ~50k rows per shard
        data/test-NNNNN.parquet        held-out families (book split "eval")
        books/books.parquet            one row per book
Uploading is done by the user (huggingface-cli upload ... --repo-type dataset).
"""
from __future__ import annotations

import gzip
import json
import os
import sys
from collections import Counter
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

ROOT = Path(os.environ.get("KDS_ROOT", Path(__file__).resolve().parent.parent))
SRC = Path(os.environ.get("KDS_PUBLIC", ROOT / "round1" / "public"))
ROWS = 50000
PAGE_SCHEMA = pa.schema([
    ("book_id", pa.string()), ("page", pa.int32()), ("printed_page", pa.int32()),
    ("text", pa.string()), ("text_norm", pa.string()), ("headings", pa.list_(pa.string())),
    ("script", pa.string()), ("ocr_engine", pa.string()), ("grade", pa.string()),
    ("valid_word_rate", pa.float32()), ("junk_ratio", pa.float32()), ("diacritic_ratio", pa.float32()),
    ("title_ar", pa.string()), ("author", pa.string()), ("death_ah", pa.int32()), ("field", pa.string()),
])
BOOK_FIELDS = ["id", "title_ar", "title_latin", "author", "death_ah", "field", "genre", "creed", "language",
               "script", "engine", "pages", "public_pages", "grade", "vocalized_pages", "family", "split",
               "license", "metadata_source"]


class Shards:
    def __init__(self, d: Path, prefix: str):
        self.d, self.prefix, self.rows, self.i, self.n = d, prefix, [], 0, 0

    def add(self, row: dict) -> None:
        self.rows.append(row)
        if len(self.rows) >= ROWS:
            self.flush()

    def flush(self) -> None:
        if self.rows:
            t = pa.Table.from_pylist(self.rows, schema=PAGE_SCHEMA)
            pq.write_table(t, self.d / f"{self.prefix}-{self.i:05d}.parquet", compression="zstd")
            self.n += len(self.rows)
            self.i += 1
            self.rows = []


def main(version: str) -> None:
    out = ROOT / "release" / version
    (out / "data").mkdir(parents=True, exist_ok=True)
    (out / "books").mkdir(parents=True, exist_ok=True)
    books = [json.loads(l) for l in (SRC / "books.jsonl").open(encoding="utf8")]
    books.sort(key=lambda b: b["id"])
    shards = {"train": Shards(out / "data", "train"), "test": Shards(out / "data", "test")}
    st, chars = Counter(), 0
    for b in books:
        split = "test" if b["split"] == "eval" else "train"
        st[f"books_{split}"] += 1
        for line in gzip.open(SRC / "pages" / f"{b['id']}.jsonl.gz", "rt", encoding="utf8"):
            p = json.loads(line)
            q = p["quality"]
            shards[split].add({"book_id": p["book_id"], "page": p["page"], "printed_page": p["printed_page"],
                               "text": p["text"], "text_norm": p["text_norm"], "headings": p["headings"],
                               "script": p["script"], "ocr_engine": p["ocr_engine"], "grade": q["grade"],
                               "valid_word_rate": q.get("valid_word_rate"), "junk_ratio": q.get("junk_ratio"),
                               "diacritic_ratio": q.get("diacritic_ratio"), "title_ar": b.get("title_ar"),
                               "author": b.get("author"), "death_ah": b.get("death_ah"), "field": b.get("field")})
            st[f"pages_{split}"] += 1
            st["grade_" + q["grade"]] += 1
            chars += len(p["text"])
    for s in shards.values():
        s.flush()
    bt = pa.Table.from_pylist([{k: b.get(k) for k in BOOK_FIELDS} for b in books])
    pq.write_table(bt, out / "books" / "books.parquet", compression="zstd")
    st["chars"] = chars
    fields = Counter(b.get("field") or "?" for b in books)
    centuries = Counter(((b.get("death_ah") or 0) // 100 + 1) for b in books)
    engines = Counter(b.get("engine") for b in books)
    (out / "README.md").write_text(card(version, st, fields, centuries, engines, len(books)), encoding="utf8")
    (out / "stats.json").write_text(json.dumps({**st, "books": len(books)}, indent=1), encoding="utf8")
    print("release", out, dict(st), flush=True)


def card(version: str, st: Counter, fields: Counter, centuries: Counter, engines: Counter, nbooks: int) -> str:
    pages = st["pages_train"] + st["pages_test"]
    f_rows = "\n".join(f"| {k} | {v} |" for k, v in fields.most_common())
    c_rows = "\n".join(f"| {k} | {centuries[k]} |" for k in sorted(centuries))
    e_rows = "\n".join(f"| {k} | {v} |" for k, v in engines.most_common())
    return f"""---
license: cdla-sharing-1.0
language:
- ar
pretty_name: An-Nuqthah Kitab OCR (classical Islamic texts, Ahlussunnah)
size_categories:
- 100K<n<1M
task_categories:
- text-generation
- fill-mask
- image-to-text
tags:
- arabic
- classical-arabic
- islamic-texts
- ocr
- kitab-kuning
- pesantren
configs:
- config_name: pages
  default: true
  data_files:
  - split: train
    path: data/train-*.parquet
  - split: test
    path: data/test-*.parquet
- config_name: books
  data_files:
  - split: train
    path: books/books.parquet
---

# An-Nuqthah Kitab OCR — {version}

Clean, page-level OCR text of **{nbooks:,} classical Islamic books** ({pages:,} pages, {st['chars'] / 1e6:,.0f} M characters),
mostly Arabic, from the library work of Pondok Pesantren An-Nuqthah (Indonesia). The selection follows the
Ahlussunnah wal Jama'ah tradition taught in pesantren: Ash'ari and Maturidi creed, the four Sunni madhhabs
(mainly Shafi'i), and Sunni tasawuf.

*Ringkasan (ID): teks OCR bersih per halaman dari {nbooks:,} kitab klasik Ahlussunnah (Asy'ari/Maturidi, tasawuf Sunni),
hasil kerja perpustakaan Pondok Pesantren An-Nuqthah. Hanya karya pengarang yang wafat ≤ 1374 H (domain publik).*

## What is in it

- **Only public-domain authors**: every book's author died in or before 1374 AH (before 1956 CE), so the work is
  out of copyright under life+70 rules (Indonesia, UU 28/2014).
- **Only the author's text**: editor (muhaqqiq) footnotes, running headers/footers and pages detected as the
  editor's introduction are removed, because modern edition apparatus can still be under copyright.
- **No scans**, no source file names. Each page keeps its PDF page index and the printed page number when it
  could be read, so text can be traced back to a printed edition.
- Books graded A or B (≥ 60% of their pages A/B); pages graded A–C are kept, D pages are dropped (see *Quality*).

| field | books |
|---|---|
{f_rows}

| author's death century (AH) | books |
|---|---|
{c_rows}

| OCR engine | books |
|---|---|
{e_rows}

## Configs and fields

`pages` (default) — one row per page. Splits: `train` ({st['pages_train']:,} pages, {st['books_train']:,} books) and
`test` ({st['pages_test']:,} pages, {st['books_test']:,} books). The test split holds out whole *families* (all copies,
volumes and editions of a work), so no text of a test work appears in train.

| column | meaning |
|---|---|
| `book_id` | stable book id (join with `books`) |
| `page` | 1-based page index in the scanned PDF |
| `printed_page` | page number printed on the page, if read |
| `text` | clean body text; letters and harakat exactly as OCR'd |
| `text_norm` | normalized layer: no harakat/tatweel, unified alef (أ إ آ ٱ → ا) and ى → ي, ASCII digits; ة ؤ ئ kept |
| `headings` | headings the OCR engine marked |
| `script` | `ar`, `fa`, `ur`, `latin`, `mixed` |
| `ocr_engine` | `paddleocr-vl-1.6`, `dots.mocr` (re-OCR of weak pages), `pdf-text-layer` |
| `grade` | page quality A/B/C (see below) |
| `valid_word_rate` | share of words found in an Arabic lexicon (0–1) |
| `junk_ratio`, `diacritic_ratio` | non-text characters per character; harakat per Arabic letter |
| `title_ar`, `author`, `death_ah`, `field` | copied from `books` for convenience |

`books` — one row per book: titles, author, death year (AH), field, genre (matn/syarah/hasyiyah/kitab), creed,
language, OCR engine, page counts, grade, family id, split, licence reason, metadata source.

```python
from datasets import load_dataset
pages = load_dataset("anocta/Aswaja", "pages", split="train")
books = load_dataset("anocta/Aswaja", "books", split="train")
```

## How it was made

1. PDFs collected by the pesantren library (Telegram kitab-sharing channels, archive.org, own scans), sorted by
   an LLM title pass and an Ahlussunnah content policy.
2. OCR with PaddleOCR-VL 1.6; pages scoring below 0.85 on the lexicon check re-OCR'd with dots.mocr; PDFs with a
   usable text layer used directly.
3. Cleaning (`clean.py`, in the companion GitHub repository [wildancz/Aswaja](https://github.com/wildancz/Aswaja)): markup removal, running header/footer and page-number
   detection, footnote split (separator rule or numbered markers), encoding fixes, normalized layer.
4. Deduplication by 5-word shingles: exact re-uploads collapse to the best copy; a juz or treatise fully contained
   in a better book is dropped.
5. Metadata (title, author, death year, field, creed) guessed by a local LLM ("kyAI") from the first pages and
   checked against Shamela/OpenITI author lists where the name matched exactly.

## Quality

- Page grade A: lexicon rate ≥ 0.9 and < 2% junk; B: ≥ 0.8; C: ≥ 0.6 (filter on `grade` if you need cleaner text).
  In this release: A {st['grade_A']:,} · B {st['grade_B']:,} · C {st['grade_C']:,} pages.
- **Harakat are not reliable.** OCR models often invent or misplace them. Do not train diacritization on this text
  without verification.
- The lexicon check cannot see *hallucinated* text: fluent words that are not on the page. This is rare in Arabic
  naskh but exists.
- Nested layouts (matn inside sharh, marginal hashiyah) may interleave texts in reading order.
- Some footnotes are missed when the OCR dropped their numbers; some running chapter titles stay in `text`.

## Metadata caveats and takedown

Author, death year and creed are **machine guesses** (`metadata_source`) and have not all been checked by a person.
The licence decision rests on the death year; for a sharh or hashiyah the year should be the commentator's.
If you find a book that is still under copyright, or wrong metadata, please open a discussion on this repository
and it will be removed or fixed in the next version.

## Licence

The underlying texts are in the public domain. The OCR, cleaning, page alignment and metadata are
© 2026 Pondok Pesantren An-Nuqthah, Anocta and wildancz, released under the
**[Community Data License Agreement – Sharing, Version 1.0](https://cdla.dev/sharing-1-0/)** (CDLA-Sharing-1.0).

- You may use, modify and share the data, also commercially.
- If you publish the data or a modified version of it, publish it under CDLA-Sharing-1.0 too and keep this notice.
- What you compute from the data (trained models, analyses, statistics: "Results" in the licence) is not bound by it.

Please credit: *Pondok Pesantren An-Nuqthah, Anocta, wildancz — An-Nuqthah Kitab OCR*.

## Citation

```
@misc{{annuqthah_kitab_ocr,
  title  = {{An-Nuqthah Kitab OCR: clean page-level OCR of classical Islamic texts}},
  author = {{Pondok Pesantren An-Nuqthah and Anocta and wildancz}},
  year   = {{2026}},
  note   = {{Version {version}}}
}}
```
"""


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "v0.1")
