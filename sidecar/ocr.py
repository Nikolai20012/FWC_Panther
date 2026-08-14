"""Targeted banner OCR for the Panther Detector sidecar.

Replaces the old whole-frame OCR: the user draws a box per banner field once
(calibration), and we only ever OCR those tiny crops. Each field has an
allowlist and a parser, so a misread can't silently produce garbage.

Fields: cameraId, temperature, clock (cross-check), moon (crop only, no OCR).
"""

import datetime
import re
import threading

import cv2

FIELDS = ("cameraId", "temperature", "clock", "moon")

_reader = None
_reader_lock = threading.Lock()

_ALLOWLISTS = {
    "cameraId": "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789",
    "temperature": "0123456789F-",
    "clock": "0123456789/:APM ",
}


def _get_reader():
    global _reader
    with _reader_lock:
        if _reader is None:
            import easyocr  # deferred: heavy import
            _reader = easyocr.Reader(["en"], gpu=False, verbose=False)
        return _reader


def crop(frame, box):
    """box is normalized [x1, y1, x2, y2] in 0..1 of the frame."""
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = box
    x1, x2 = sorted((max(0, int(x1 * w)), min(w, int(x2 * w))))
    y1, y2 = sorted((max(0, int(y1 * h)), min(h, int(y2 * h))))
    if x2 - x1 < 2 or y2 - y1 < 2:
        return None
    return frame[y1:y2, x1:x2]


def _prep(img):
    """Upscale small banner crops — EasyOCR is much better above ~40px text."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    scale = max(1, int(80 / max(1, gray.shape[0])))
    if scale > 1:
        gray = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    return gray


def read_field(frame, field, box):
    """OCR one calibrated box → {raw, value, ok}. Moon boxes are not OCR'd.

    For the clock field, value is a LIST of candidate ISO datetimes: banner
    punctuation OCRs unreliably (colon→digit, slash→7), so we read several
    preprocessing variants and let the caller match any candidate against the
    file timestamp.
    """
    img = crop(frame, box)
    if img is None:
        return {"raw": "", "value": None, "ok": False}
    if field == "moon":
        return {"raw": "", "value": None, "ok": True}

    reader = _get_reader()
    # recognize() rather than readtext(): readtext runs CRAFT text-detection to
    # find where the words are, but calibration already told us that - the crop
    # IS the field. Skipping that pass reads the same characters ~6x faster,
    # which matters because the clock alone is three passes per item.
    if field == "clock":
        candidates, raws = set(), []
        for variant in _clock_variants(img):
            raw = " ".join(reader.recognize(variant, allowlist=_ALLOWLISTS["clock"], detail=0)).strip()
            raws.append(raw)
            candidates.update(_clock_candidates(raw))
        return {"raw": raws[0] if raws else "", "value": sorted(candidates), "ok": bool(candidates)}

    results = reader.recognize(_prep(img), allowlist=_ALLOWLISTS.get(field), detail=0)
    raw = " ".join(results).strip()
    value, ok = _parse(field, raw)
    return {"raw": raw, "value": value, "ok": ok}


def _clock_variants(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    big = cv2.resize(gray, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)
    _, otsu = cv2.threshold(big, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return (big, otsu, 255 - otsu)


def _clock_candidates(raw):
    """All plausible datetimes readable from one OCR string.

    Digits should be MMDDYYYYHHMMSS (14, zero-padded). A misread colon often
    *inserts* a digit (12:07:36 → 1207336), so for 15-16 digits we also try
    dropping one or two digits from the time portion and keep what validates.
    """
    text = raw.replace(" ", "")
    ap = "PM" if "PM" in text else ("AM" if "AM" in text else None)
    if ap is None:
        return []
    digits = re.sub(r"\D", "", text)

    def attempt(d):
        if len(d) != 14:
            return None
        try:
            mo, day, y, hh, mm, ss = (int(d[i:j]) for i, j in
                                      ((0, 2), (2, 4), (4, 8), (8, 10), (10, 12), (12, 14)))
            if not (2010 <= y <= 2099 and hh <= 12):
                return None
            return datetime.datetime(y, mo, day, hh % 12 + (12 if ap == "PM" else 0), mm, ss).isoformat()
        except ValueError:
            return None

    pool = {digits}
    for _ in range(min(2, max(0, len(digits) - 14))):
        pool = {d[:i] + d[i + 1:] for d in pool for i in range(8, len(d))}
    return [c for c in (attempt(d) for d in pool) if c]


def _parse(field, raw):
    text = raw.replace(" ", "")
    if field == "temperature":
        m = re.search(r"(-?\d{1,3})F?", text)
        return (int(m.group(1)), True) if m else (None, False)
    if field == "cameraId":
        m = re.fullmatch(r"[A-Z0-9]{2,15}", text)
        return (text, True) if m else (text or None, False)
    return raw or None, bool(raw)
