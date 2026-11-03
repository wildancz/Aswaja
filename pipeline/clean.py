"""Page cleaning for the kitab OCR dataset (stdlib only, Python 3.8+ so it also runs on the pod).

Three text layers per page, never mixed:
  text_raw   exactly what the OCR engine produced (kept for audit and re-cleaning)
  text       body text: markup, running headers/footers, page numbers and footnotes taken out;
             encoding junk fixed (presentation forms, zero-width chars). Letters are NOT changed.
  text_norm  search/training layer made from `text`: no harakat or tatweel, alef/ya unified,
             digits to ASCII. Persian ی/ک are mapped only for Arabic-language books, so Jawi,
             Pegon, Persian and Urdu spelling survives.
Headers, footers and footnotes are kept in their own fields, not thrown away.
"""
from __future__ import annotations

import difflib
import re
import unicodedata
from collections import Counter

AR_LETTER = re.compile("[ء-يٱ-ۓەۺ-ۼݐ-ݿ]")
HARAKAT = re.compile("[ً-ٰٟۖ-ۭ]")  # tashkil, dagger alef, Qur\'anic marks (not the digits at U+0660)
TATWEEL = "ـ"
LATIN = re.compile(r"[A-Za-zÀ-ɏ]")
DIGITS = "0123456789٠١٢٣٤٥٦٧٨٩۰۱۲۳۴۵۶۷۸۹"
TO_ASCII = str.maketrans("٠١٢٣٤٥٦٧٨٩۰۱۲۳۴۵۶۷۸۹", "01234567890123456789")
ZERO_WIDTH = re.compile("[​-‏‪-‮⁠﻿­]")
PRESENTATION = re.compile("[ﭐ-﷏ﹰ-ﻼ]")  # letter forms; not the word ligatures ﷺ ﷲ ﷻ (U+FDF0-FDFF)
WORD_LIGATURE = re.compile("[ﷰ-﷿]")
ALEF = str.maketrans({"أ": "ا", "إ": "ا", "آ": "ا", "ٱ": "ا", "ى": "ي"})
PERSIAN_TO_AR = str.maketrans({"ی": "ي", "ک": "ك", "ە": "ه"})
OK_PUNCT = set(" \n\t.,:;!?()[]{}«»\"'-–—/*%+=_،؛؟٪٫٬۔")

# A line that is only a page number: "١٢", "- 12 -", "(١٢)", "[ص ١٢]", "12 |".
PAGE_NUM = re.compile(r"^[\s\-–—_ـ()\[\]{}«»|.ـ*•]*(?:ص|صـ|صفحة|page|hal\.?|p\.)?[\s.:]*([0-9٠-٩۰-۹]{1,4})[\s\-–—_ـ()\[\]{}«»|.*•]*$", re.I)
# A footnote separator rule printed above the footnotes.
# Edge junk that is never text: print timestamps, URLs, 1-3 stray Latin letters left by watermarks.
EDGE_JUNK = re.compile(r"^\s*(?:\d{1,2}/\d{1,2}/\d{2,4}\s+\d{1,2}:\d{2}(?::\d{2})?\s*(?:AM|PM)?|[A-Za-z]{1,3}|(?:https?://|www\.)\S+)\s*$", re.I)
RULE = re.compile(r"^\s*[_\-–—ـ]{5,}\s*$")  # not *** or ...: those separate matn from sharh or poems
# "(١) ..." / "[1] ..." / "١ - ..." at the start of a footnote.
FN_START = re.compile(r"^\s*[(\[]\s*([0-9٠-٩۰-۹]{1,3})\s*[)\]]\s*\S|^\s*([0-9٠-٩۰-۹]{1,3})\s*[-–—)]\s+\S")
# The editor's (muhaqqiq's) own introduction: modern text, so it is kept out of any public release.
EDITOR_INTRO = re.compile(r"مقدم[ةه]\s+(?:ال)?(?:محقق|تحقيق)|منهج\s+(?:ال)?تحقيق|ترجم[ةه]\s+(?:ال)?محقق|وصف\s+(?:ال)?نسخ")
MD_HEADING = re.compile(r"^\s{0,3}#{1,6}\s+")
MD_IMG = re.compile(r"!\[[^\]]*\]\([^)]*\)")
HTML_TAG = re.compile(r"</?[A-Za-z][^>]{0,200}>")
MD_EMPH = re.compile(r"\*\*|__|(?<!\w)\*(?!\s)|(?<!\s)\*(?!\w)")
LATEX = re.compile(r"\$+([^$]{0,80})\$+")


def fix_encoding(s: str) -> str:
    """OCR/encoding artefacts only: presentation forms -> base letters, zero-width/bidi marks out."""
    s = PRESENTATION.sub(lambda m: unicodedata.normalize("NFKC", m.group()), s)
    s = ZERO_WIDTH.sub("", s)
    s = re.sub("[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f�]", "", s)  # control chars, U+FFFD from bad decodes
    return unicodedata.normalize("NFC", s).replace("\r", "")


def strip_markup(s: str) -> tuple[str, list[str]]:
    """Removes markdown/HTML the OCR engines emit; returns (text, headings found)."""
    s = MD_IMG.sub("", s)
    s = LATEX.sub(lambda m: m.group(1), s)
    s = re.sub(r"<br\s*/?>", "\n", s, flags=re.I)
    s = HTML_TAG.sub("", s)
    heads, out = [], []
    for line in s.split("\n"):
        if MD_HEADING.match(line):
            line = MD_HEADING.sub("", line)
            heads.append(line.strip())
        if not RULE.match(line):  # "__________" is a footnote rule, not markdown emphasis
            line = MD_EMPH.sub("", line)
        if re.fullmatch(r"\s*\|?\s*:?-{3,}:?\s*(\|\s*:?-{3,}:?\s*)*\|?\s*", line):
            continue  # markdown table rule
        out.append(line.rstrip())
    return "\n".join(out), [h for h in heads if h]


def line_key(s: str) -> str:
    """Letters only, unified: used to recognise the same running header across pages."""
    s = HARAKAT.sub("", s).replace(TATWEEL, "").translate(ALEF).translate(PERSIAN_TO_AR)
    s = re.sub(r"[ةه]", "ه", s)
    return "".join(ch for ch in s if ch.isalpha()).lower()


def page_number(line: str) -> int | None:
    m = PAGE_NUM.match(line)
    return int(m.group(1).translate(TO_ASCII)) if m else None


def nonempty(lines: list[str]) -> list[int]:
    return [i for i, l in enumerate(lines) if l.strip()]


EDGE = 2  # lines from the top/bottom that may be running header/footer


class BookContext:
    """Per-book statistics gathered in a first pass: which edge lines repeat (running heads)."""

    def __init__(self, pages: list[str]):
        self.edge, self.body = Counter(), Counter()
        for t in pages:
            lines = strip_markup(fix_encoding(t))[0].split("\n")
            idx = nonempty(lines)
            edge = set(idx[:EDGE] + idx[-EDGE:])
            for i in idx:
                k = line_key(lines[i])
                if not k or len(k) > 80:
                    continue
                (self.edge if i in edge else self.body)[k] += 1
        n = max(len(pages), 1)
        need = max(3, int(0.04 * n))
        self.running = {k for k, c in self.edge.items() if c >= need and self.body[k] * 2 < c}
        self._top = sorted(self.running, key=lambda k: -self.edge[k])[:30]

    def is_running(self, line: str) -> bool:
        k = line_key(line)
        if len(k) < 5 or len(line.split()) > 12:  # "فصل", "توضيح" are section heads, not running heads
            return False
        if k in self.running:
            return True
        return any(abs(len(k) - len(r)) <= 4 and difflib.SequenceMatcher(None, k, r).ratio() >= 0.85 for r in self._top)


def split_footnotes(lines: list[str]) -> tuple[list[str], list[str], str | None]:
    """(body, footnotes, how). Uses a printed rule first; else a bottom block of "(١) ..." lines
    whose markers also appear in the body. Only the lower part of the page is searched."""
    idx = nonempty(lines)
    if len(idx) < 4:
        return lines, [], None
    lo = idx[int(len(idx) * 0.35)]
    for i in idx:
        if i >= lo and RULE.match(lines[i]):
            fn = [l for l in lines[i + 1:] if l.strip()]
            if fn and len("".join(fn)) <= len("".join(lines[:i])):  # notes are shorter than the body
                return lines[:i], fn, "rule"
    cands = []  # (line index, footnote number, bracketed?) in the lower part of the page
    for i in idx:
        m = FN_START.match(lines[i]) if i >= lo else None
        if m:
            cands.append((i, int((m.group(1) or m.group(2)).translate(TO_ASCII)), bool(m.group(1))))
    for k, (i, n, bracketed) in enumerate(cands):
        seq = [c[1] for c in cands[k:]]
        if any(b - a not in (1, 2) for a, b in zip(seq, seq[1:])):
            continue  # numbers must run 1,2,3 (one OCR-lost number allowed)
        in_body = n in body_markers("\n".join(lines[:i]))
        # Body markers are often OCR'd in another form ((٢) vs (2) vs ²), so a bracketed run from 1 with
        # at least two notes is enough; "١ - ..." numbering (also used for numbered hadith) needs the body marker.
        if in_body or (bracketed and n == 1 and len(seq) >= 2):
            fn = [l for l in lines[i:] if l.strip()]
            if len(fn) <= 0.6 * len(idx):
                return lines[:i], fn, "markers"
            break
    return lines, [], None


SUPERSCRIPT = str.maketrans("⁰¹²³⁴⁵⁶⁷⁸⁹", "0123456789")
MARK_IN_BODY = re.compile(r"[(\[]\s*([0-9٠-٩۰-۹]{1,3})\s*[)\]]|([⁰¹²³⁴⁵⁶⁷⁸⁹]{1,3})")


def body_markers(text: str) -> set[int]:
    return {int((a or b).translate(TO_ASCII).translate(SUPERSCRIPT)) for a, b in MARK_IN_BODY.findall(text)}


def normalize(text: str, lang: str | None) -> str:
    """The normalized layer (see module doc). Ta marbuta and hamza carriers ؤ ئ are kept."""
    s = WORD_LIGATURE.sub(lambda m: unicodedata.normalize("NFKC", m.group()), text)  # ﷺ -> صلى الله عليه وسلم
    s = HARAKAT.sub("", s).replace(TATWEEL, "").translate(ALEF).translate(TO_ASCII)
    if (lang or "ar").startswith("ar"):
        s = s.translate(PERSIAN_TO_AR)
    s = re.sub(r"[ \t]+", " ", s)
    return re.sub(r"\n{3,}", "\n\n", s).strip()


def script_of(text: str) -> str:
    """Rough script/language tag for one page: ar, fa, ur, jawi, latin, mixed, empty."""
    a, l = len(AR_LETTER.findall(text)), len(LATIN.findall(text))
    if a + l < 20:
        return "empty"
    if l > 0.7 * (a + l):
        return "latin"
    if a > 0.7 * (a + l):
        urdu = len(re.findall(r"[ہےٹڈڑںھۓ]", text))
        jawi = len(re.findall(r"[ڠڤڽڬۏ]", text))  # nga, pa, nya, ga, va: Jawi/Pegon letters
        fa = len(re.findall(r"[پچژگکی]", text))
        if urdu > 0.02 * a:
            return "ur"
        if jawi > 0.004 * a:
            return "jawi"
        if fa > 0.03 * a:
            return "fa"
        return "ar"
    return "mixed"


def signals(text: str) -> dict:
    """Language-independent page signals for quality grading."""
    chars = re.sub(r"\s", "", text)
    n = max(len(chars), 1)
    a = len(AR_LETTER.findall(text))
    lat = len(LATIN.findall(text))
    har = len(HARAKAT.findall(text))
    junk = sum(1 for ch in chars if not (AR_LETTER.match(ch) or LATIN.match(ch) or ch in DIGITS or ch in OK_PUNCT
                                          or HARAKAT.match(ch) or ch == TATWEEL))
    words = text.split()
    top = Counter(words[:600]).most_common(1)[0][1] / len(words[:600]) if words else 0.0
    grams = [" ".join(words[i:i + 4]) for i in range(0, max(len(words) - 3, 0))]
    rep4 = (1 - len(set(grams)) / len(grams)) if len(grams) >= 20 else 0.0
    return {"chars": len(chars), "words": len(words), "arabic_ratio": round(a / n, 3), "latin_ratio": round(lat / n, 3),
            "junk_ratio": round(junk / n, 3), "diacritic_ratio": round(har / max(a, 1), 3),
            "top_word_ratio": round(top, 3), "repeat_4gram": round(rep4, 3)}


def grade(valid: float | None, sig: dict, flags: list[str]) -> str:
    """A (clean) .. D (unusable). `valid` = dictionary valid-word rate from the OCR scorer;
    unscored pages are capped at B because only the shape heuristics vouch for them."""
    if sig["chars"] < 80 or "loop" in flags or "low_quality" in flags:
        return "D"
    if sig["junk_ratio"] > 0.08 or sig["repeat_4gram"] > 0.35 or sig["top_word_ratio"] > 0.15:
        return "D"
    if valid is None:
        return "B" if sig["junk_ratio"] < 0.03 and sig["repeat_4gram"] < 0.15 else "C"
    if valid >= 0.9 and sig["junk_ratio"] < 0.02:
        return "A"
    if valid >= 0.8:
        return "B"
    if valid >= 0.6:
        return "C"
    return "D"


def clean_page(raw: str, ctx: BookContext, lang: str | None = None) -> dict:
    """One page -> dict with the three layers plus header/footer/footnotes/printed page."""
    text, headings = strip_markup(fix_encoding(raw or ""))
    lines = text.split("\n")
    header, footer, printed = [], [], None
    idx = nonempty(lines)
    drop = set()
    for pos, i in enumerate(idx[:EDGE]):
        n = page_number(lines[i])
        if n is not None or ctx.is_running(lines[i]) or EDGE_JUNK.match(lines[i]):
            header.append(lines[i].strip()); drop.add(i)
            printed = n if n is not None else printed
        elif pos == 0 and (m := re.match(r"^\s*([0-9٠-٩۰-۹]{1,4})\s+(.{2,60})$|^(.{2,60}?)\s+([0-9٠-٩۰-۹]{1,4})\s*$", lines[i])):
            rest = m.group(2) or m.group(3)
            if ctx.is_running(rest):  # "١٢  كتاب الطهارة" running head with the number on it
                header.append(lines[i].strip()); drop.add(i)
                printed = int((m.group(1) or m.group(4)).translate(TO_ASCII))
        else:
            break
    for i in reversed(idx[-EDGE:]):  # page numbers at the foot; running footers only after footnotes are out
        if i in drop:
            continue
        n = page_number(lines[i])
        if n is None and not EDGE_JUNK.match(lines[i]):
            break
        footer.insert(0, lines[i].strip()); drop.add(i)
        printed = printed if printed is not None else n
    body = [l for i, l in enumerate(lines) if i not in drop]
    body, notes, fn_how = split_footnotes(body)
    if not notes:
        bidx = nonempty(body)
        if bidx and ctx.is_running(body[bidx[-1]]) and not FN_START.match(body[bidx[-1]]):
            footer.insert(0, body[bidx[-1]].strip())
            body = body[:bidx[-1]]
    clean = re.sub(r"\n{3,}", "\n\n", "\n".join(l.rstrip() for l in body)).strip()
    clean = re.sub(r"[ \t]{2,}", " ", clean)
    top = " ".join(header + (headings or []) + [l for l in body[:3]])
    editor = bool(EDITOR_INTRO.search(top))
    return {"text": clean, "text_norm": normalize(clean, lang), "header": " | ".join(header) or None, "editor_intro": editor,
            "footer": " | ".join(footer) or None, "footnotes": "\n".join(n.strip() for n in notes) or None,
            "footnote_split": fn_how, "printed_page": printed, "headings": headings or None}
