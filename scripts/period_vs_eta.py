#!/usr/bin/env python3
"""Cycle length T against the noise strength eta in the Stage B tree runs of run 88 (descriptive, 2026-10-26).

Reads the outputs of scripts/stageB_classify.py (the fixed classification; nothing here changes it) and, per eta, pools
the cycles of each label (D, F) over the seeds. The first cycle of every run (the transient) and the c cycles are left
out. Per eta and label: number of cycles, median, quartiles and range of T, and the median depth of the 132 small group
(log10 of its spread at the closest approach). The noiseless reference (eta = 0) is listed separately. For each label,
a least-squares line T = a + b ln(eta) over the eta > 0 values that have cycles of that label, when there are at least
two such values (the slope is descriptive only; no model is assumed).
  -> printed; and <out>.json

  python3 scripts/period_vs_eta.py analysis/noiseless_n50/stageB_classify.json --states r1600 --out analysis/noiseless_n50/period_vs_eta.json
"""
from pathlib import Path
import argparse
import json

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("classify", nargs="+")
    ap.add_argument("--states", default="r1600", help="substring of the file names to keep (the starting state)")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    runs = {}
    for f in a.classify:
        for r in json.load(open(f))["runs"]:
            if a.states in r["file"] and not any(t in r["file"] for t in ("_xhost", "_xsame", "_xlong")):
                runs[r["file"]] = r  # later files replace earlier ones of the same name
    rows = {}
    for r in runs.values():
        for c in r["cycles"][1:]:
            if c["label"] in ("D", "F"):
                rows.setdefault((r["eta"], c["label"]), []).append((c["T"], c["depth132small"], r["seed"]))
    table = []
    for (eta, lab), v in sorted(rows.items()):
        T = np.array([x[0] for x in v])
        d = np.array([x[1] for x in v if x[1] is not None])
        table.append(dict(eta=eta, label=lab, n_cycles=len(T), n_runs=len({x[2] for x in v}),
                          T_median=float(np.median(T)), T_q25=float(np.percentile(T, 25)),
                          T_q75=float(np.percentile(T, 75)), T_min=float(T.min()), T_max=float(T.max()),
                          depth132_median=float(np.median(d)) if len(d) else None))
    fits = {}
    for lab in ("D", "F"):
        pts = [(np.log(t["eta"]), t["T_median"]) for t in table if t["label"] == lab and t["eta"] > 0]
        if len(pts) >= 2:
            x, y = np.array(pts).T
            b, a0 = np.polyfit(x, y, 1)
            fits[lab] = dict(n_eta=len(pts), slope_per_ln_eta=float(b), slope_per_decade=float(b * np.log(10)),
                             intercept=float(a0))
    out = dict(sources=[Path(f).name for f in a.classify], states=a.states, table=table, fits=fits)
    Path(a.out).write_text(json.dumps(out, indent=1))
    for t in table:
        print(f"eta {t['eta']:8.0e} {t['label']} cycles {t['n_cycles']:3d} runs {t['n_runs']} "
              f"T med {t['T_median']:7.0f} [{t['T_q25']:.0f}, {t['T_q75']:.0f}] range [{t['T_min']:.0f}, {t['T_max']:.0f}] "
              f"depth132 {t['depth132_median']}")
    print(json.dumps(fits, indent=1))


if __name__ == "__main__":
    main()
