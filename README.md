# Aswaja — cleaning pipeline

Code behind the **Aswaja** dataset: clean, page-level OCR text of classical Islamic books
(Ahlussunnah: Ash'ari/Maturidi creed, Sunni fiqh and tasawuf) from the library of Pondok Pesantren An-Nuqthah,
Indonesia.

- Dataset (v0.2): 2,051 books, 716,680 pages, ~1.0 B characters, Parquet — on Hugging Face: [anocta/Aswaja](https://huggingface.co/datasets/anocta/Aswaja)
- Only books whose author died ≤ 1374 AH (public domain); editor footnotes and introductions removed; no scans.

*Ringkasan (ID): kode pembersih untuk dataset Aswaja. Datanya ada di Hugging Face.*

## What the pipeline does

`pipeline/clean.py` (stdlib only, Python ≥ 3.8) turns one OCR'd page into three layers that are never mixed:

| layer | content |
|---|---|
| `text_raw` | the OCR output as produced |
| `text` | body text: markup removed, running headers/footers, page numbers, converter watermarks and footnotes split out into their own fields; encoding fixes only (presentation forms, zero-width marks). Letters are not changed. |
| `text_norm` | for search/training: no harakat or tatweel, alef forms → ا, ى → ي, ASCII digits; ة ؤ ئ kept. Persian ی/ک are mapped only for Arabic-language books, so Persian, Urdu, Jawi and Pegon spelling survives. |

It also grades pages A–D (lexicon valid-word rate + junk, repetition and loop signals), detects the script
(ar/fa/ur/jawi/latin), the printed page number, and pages of an editor's introduction (مقدمة المحقق).

Running headers are learned per book: a short line that keeps appearing at the top or bottom of pages (and not
in the body) is a running head; fuzzy matching absorbs OCR variation. Footnotes are split on a printed separator
rule, or on a bottom block of `(١) (٢) …` lines whose numbering runs in order. `*****` lines are deliberately not
treated as footnote rules, because in works like the sharh of al-Hikam they separate matn from sharh.

```python
import clean
pages = [...]                         # raw OCR text of every page of one book
ctx = clean.BookContext(pages)        # learns this book's running headers
out = clean.clean_page(pages[10], ctx, lang="ar")
out["text"], out["footnotes"], out["header"], out["printed_page"], out["text_norm"]
```

Around `clean.py` the dataset build also deduplicates books by 5-word shingles (`edition` = exact re-uploads,
`family` = volumes, parts and other editions of a work, used for the train/test split) and sorts books into
licence buckets by the author's death year; that driver is tied to the pesantren's own servers and is not
published. `pipeline/release.py` packs an export (`books.jsonl` + `pages/<book_id>.jsonl.gz`) into Parquet with
the dataset card. `pipeline/report_sample.py` prints before/after examples from a folder of cleaned pages.

`sample/pages-sample.jsonl` — 50 random pages of the public release.

## Selection rules

1. Creed/field: Ash'ari, Maturidi or Sunni author, or tasawuf by a non-sectarian author (LLM guess, checked
   against Shamela/OpenITI author lists where the name matches exactly).
2. Content policy of the pesantren: nothing promoting LGBT, opposing Islam, or atheist/agnostic philosophy;
   polemics against Ash'ari/Maturidi kept out.
3. Copyright: author died ≤ 1374 AH (life + 70); body text only.
4. Quality: book grade A/B; page grades A–C.

## Known limitations

OCR harakat are a bit unreliable; lexicon scoring cannot detect hallucinated text; nested sharh/hashiyah layouts may
interleave; metadata is machine-assisted. See the dataset card for details. Reports of wrong metadata or of a book
still under copyright are welcome as issues.

## Licence

© 2026 Pondok Pesantren An-Nuqthah, Anocta and wildancz.

- Code: MIT (see `LICENSE`).
- Dataset (and `sample/`): [CDLA-Sharing-1.0](https://cdla.dev/sharing-1-0/). Republished data stays under the
  same licence; models and analyses made from it are free.
- Please credit *Pondok Pesantren An-Nuqthah, Anocta, wildancz — Aswaja*.
