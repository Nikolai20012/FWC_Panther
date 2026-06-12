"""File-timestamp handling for trail-cam videos.

The camera writes its local clock time to the FAT card. The OS converts that
to an epoch using *today's* UTC offset, so clips recorded in the opposite DST
regime read back exactly one hour off. When the user calibrated a clock box,
the banner (the camera's own clock) is used to detect and undo that shift.
"""

import datetime
import os
import sys

FMT = "%Y-%m-%d-%H-%M-%S"

# How far the banner clock may drift from the file timestamp and still count
# as agreement (banner renders ~1s after file creation; OCR adds noise).
TOLERANCE_S = 90

DST_SHIFTS = (0, 3600, -3600)


def file_creation(path):
    st = os.stat(path)
    ts = getattr(st, "st_birthtime", None)
    if ts is None:
        ts = st.st_ctime if sys.platform == "win32" else st.st_mtime
    return datetime.datetime.fromtimestamp(ts)


def resolve(path, clock_candidates=None):
    """Best-effort recording time → (datetime, clock_match).

    clock_candidates: ISO strings OCR'd from the banner (may be noisy/empty).
    If any candidate agrees with file time under a DST shift, apply the shift.
    """
    ft = file_creation(path)
    for iso in clock_candidates or []:
        try:
            banner = datetime.datetime.fromisoformat(iso)
        except ValueError:
            continue
        for shift in DST_SHIFTS:
            shifted = ft + datetime.timedelta(seconds=shift)
            if abs((banner - shifted).total_seconds()) <= TOLERANCE_S:
                return shifted, True
    return ft, clock_candidates is None  # no clock box: nothing to disagree with
