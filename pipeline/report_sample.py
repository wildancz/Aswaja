#!/usr/bin/env python3
"""Writes SAMPEL-SEBELUM-SESUDAH.md from a folder of cleaned pages (pages.jsonl.gz, books.jsonl, stats.json).
Usage: report_sample.py <sample dir> <out.md>"""
import collections
import gzip
import json
import random
import sys
from pathlib import Path

sd, out_path = Path(sys.argv[1]), Path(sys.argv[2])
P = [json.loads(l) for l in gzip.open(sd / "pages.jsonl.gz", "rt", encoding="utf8")]
B = {b["id"]: b for b in (json.loads(l) for l in (sd / "books.jsonl").open(encoding="utf8"))}
S = json.loads((sd / "stats.json").read_text())
rnd = random.Random(21)
cats = [
    ("Catatan kaki dipisah lewat garis pemisah", [p for p in P if p["footnote_split"] == "rule" and p["quality"]["grade"] in "AB"]),
    ("Catatan kaki dipisah lewat penanda (١) (٢) + nomor halaman di header", [p for p in P if p["footnote_split"] == "markers" and p["quality"]["grade"] == "A" and p["header"]]),
    ("Watermark konverter di header", [p for p in P if p["header"] and "Converted" in p["header"] and p["quality"]["grade"] in "AB"]),
    ("Judul berjalan di header (buku Latin/Indonesia)", [p for p in P if p["script"] == "latin" and p["header"] and p["quality"]["grade"] in "AB" and len(p["header"]) > 12]),
    ("Teks berharakat: harakat tetap di `text`, hilang di `text_norm`", [p for p in P if p["quality"]["diacritic_ratio"] > 0.6 and p["quality"]["grade"] in "AB" and p["script"] == "ar"]),
    ("Halaman muqaddimah muhaqqiq ditandai `editor_intro` (tidak ikut rilis)", [p for p in P if p.get("editor_intro")]),
    ("Halaman grade D (tidak dipakai)", [p for p in P if p["quality"]["grade"] == "D" and p["quality"]["chars"] > 300 and p["script"] != "latin"]),
]
n = S["pages"]
out = ["# Sampel pembersihan — sebelum / sesudah (putaran 1)", "",
       f"Sampel: **{len(B)} kitab, {n:,} halaman**, diambil bergilir per kombinasi mesin OCR × skrip × grade "
       "(jadi grade D sengaja terwakili lebih banyak dari aslinya).", "",
       "| hasil | halaman | % |", "|---|---|---|"]
for k, lab in [("header", "header berjalan / nomor / watermark dipisah"), ("footer", "footer dipisah"),
               ("footnotes", "catatan kaki dipisah"), ("printed_page", "nomor halaman cetak terbaca")]:
    out.append(f"| {lab} | {S[k]:,} | {S[k] / n:.0%} |")
ed = sum(1 for p in P if p.get("editor_intro"))
out.append(f"| halaman muqaddimah muhaqqiq ditandai | {ed:,} | {ed / n:.1%} |")
g = collections.Counter(p["quality"]["grade"] for p in P)
out += ["", f"Grade halaman di sampel: A {g['A']:,} · B {g['B']:,} · C {g['C']:,} · D {g['D']:,}.", "",
        "Kode: `pipeline/clean.py` (pembersih), `pipeline/report_sample.py` (laporan ini). "
        f"Data sampel lengkap (semua kolom): `data/sample/pages.jsonl.gz`.", ""]
for title, c in cats:
    if not c:
        continue
    p = rnd.choice(c)
    b = B[p["book_id"]]
    judul = b["meta"].get("judul_ar") or b["meta"].get("judul_latin") or b["name"]
    q = p["quality"]
    out += [f"## {title}", "",
            f"*{judul}* — `{p['book_id']}` hlm PDF {p['page']} · {p['ocr']['engine']} · grade **{q['grade']}** · skrip {p['script']} · "
            f"kata-valid {p['ocr']['valid_word_rate']} · harakat {q['diacritic_ratio']:.0%} · sampah {q['junk_ratio']:.1%}", "",
            "**Sebelum (`text_raw`)**", "", "```", p["text_raw"][:900].strip(), "```", "",
            "| kolom | isi |", "|---|---|",
            f"| `header` | {(p['header'] or '—').replace('|', '/')} |", f"| `footer` | {(p['footer'] or '—').replace('|', '/')} |",
            f"| `printed_page` | {p['printed_page'] if p['printed_page'] is not None else '—'} |",
            f"| `footnote_split` | {p['footnote_split'] or '—'} |", f"| `editor_intro` | {p.get('editor_intro')} |", ""]
    if p["footnotes"]:
        out += ["**`footnotes`**", "", "```", p["footnotes"][:500].strip(), "```", ""]
    out += ["**Sesudah (`text`)**", "", "```", p["text"][:900].strip(), "```", "",
            "**`text_norm`**", "", "```", p["text_norm"][:400].strip(), "```", ""]
out += ["## Yang masih kurang (diketahui)", "",
        "- Catatan kaki tanpa nomor/garis di hasil OCR tidak terdeteksi; catatan pertama kadang tertinggal di badan teks bila nomornya hilang saat OCR.",
        "- PaddleOCR-VL kadang tidak menyalin catatan kaki sama sekali; dots.mocr lebih lengkap.",
        "- Garis `*****` / `.....` sengaja tidak dianggap pemisah catatan kaki: di kitab seperti Syarah Hikam itu memisahkan matn dari syarah.",
        "- Hasyiyah di pinggir/di atas matn tercampur urutannya; butuh kotak layout dots (putaran 2).",
        "- Harakat hasil OCR banyak yang salah (lihat contoh berharakat) — jangan dipakai sebagai data latih harakat tanpa verifikasi.",
        "- Skor kata-valid dihitung per halaman; matn berharakat yang rusak bisa lolos bila syarah di halaman yang sama bersih.",
        "- Halaman Urdu/Persia nasta'liq kualitasnya rendah, dan ada halaman yang *dikarang* model OCR (teks lancar tapi tidak ada di scan). "
        "Skor kata-valid tidak menangkap ini → perlu cek silang dua mesin (Paddle vs dots).",
        "- Judul bab di baris pertama yang berulang ≥ 3 halaman bisa ikut dianggap header berjalan; tetap tersimpan di kolom `header`, tidak hilang.", ""]
out_path.write_text("\n".join(out), encoding="utf8")
print(out_path, len(P), "halaman")
