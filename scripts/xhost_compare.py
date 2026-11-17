#!/usr/bin/env python3
"""Cross-host reproducibility of the Stage B tree runs (2026-10-26).

Compares a short rerun (noiseless_n50.py, same arguments, --tmax shorter, --tag _xhost) on the other host with the
first records of the original run: group labels (int16), logdev and spread (float32), bit for bit. The noise draws do
not depend on --tmax, so the rerun is the prefix of the original when the arithmetic is identical. Reports, per array,
whether the prefix is identical and the first record that differs (time = every * (record + 1); spread rows every
10 * every), and the largest |difference| of logdev and spread at and after it.
  -> printed; and <out>.json with --out

  python3 scripts/xhost_compare.py analysis/noiseless_n50/<orig>.npz analysis/noiseless_n50/<orig>_xhost.npz
"""
from pathlib import Path
import argparse
import json

import numpy as np


def compare(a_path, b_path):
    A, B = np.load(a_path), np.load(b_path)
    ev = int(A["every"])
    out = dict(original=Path(a_path).name, rerun=Path(b_path).name, every=ev)
    for key, step in (("group", ev), ("logdev", ev), ("spread", 10 * ev)):
        a, b = A[key], B[key]
        k = min(len(a), len(b))
        a, b = a[:k], b[:k]
        same = np.array([np.array_equal(x.view(np.uint8), y.view(np.uint8)) for x, y in zip(a, b)])
        r = dict(records=k, identical=bool(same.all()))
        if not same.all():
            f = int(np.argmin(same))
            r["first_diff_record"] = f
            r["first_diff_time"] = step * (f + 1)
            r["n_diff_records"] = int((~same).sum())
            if key != "group":
                with np.errstate(invalid="ignore"):  # -inf - -inf of absent deviations
                    d = np.abs(a[f:].astype(float) - b[f:].astype(float))
                r["max_abs_diff_after"] = float(np.nanmax(d)) if np.isfinite(d).any() else None
            else:
                r["n_elements_diff_at_first"] = int((a[f] != b[f]).sum())
        out[key] = r
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("original")
    ap.add_argument("rerun")
    ap.add_argument("--out")
    a = ap.parse_args()
    r = compare(a.original, a.rerun)
    print(json.dumps(r, indent=1))
    if a.out:
        Path(a.out).write_text(json.dumps(r, indent=1))


if __name__ == "__main__":
    main()
