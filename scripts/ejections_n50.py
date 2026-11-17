#!/usr/bin/env python3
"""Departures of elements from their clusters in the N = 50 traces of itinerancy_n50.py (exploratory, 2026-10-23).

Firing times (upward x = 0, linear interpolation between the saved time units) are assigned to the cycles of element
0; clusters per cycle are groups of elements whose firing times differ by < 1e-3. A departure is an element that
leaves a cluster of >= 10 elements and next stays in a cluster of >= 10 for 10 cycles that does not contain its
former companions. Recorded: sign of its firing offset from the parent just after leaving (advanced < 0), the
position of the destination among the big clusters in firing order counted from the parent (+1 = next), and the
number of cycles until it joined. Repeated detections of one departure (cluster momentarily below 10) are merged.

  python3 scripts/ejections_n50.py --index 394   -> analysis/heteroclinic_n50/ejections_0394.json
"""
from collections import Counter
from pathlib import Path
import argparse
import json
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from eps_three_cluster import ROOT  # noqa: E402

IT = ROOT / "analysis/itinerancy_n50"
OUT = ROOT / "analysis/heteroclinic_n50"


def firing_matrix(Z):
    n = Z.shape[1] // 2
    fires = []
    for i in range(n):
        x = Z[:, i]
        k = np.nonzero((x[:-1] < 0) & (x[1:] >= 0))[0]
        fires.append(k + (-x[k]) / (x[k + 1] - x[k]))
    ref = fires[0]
    T = float(np.median(np.diff(ref)))
    F = np.full((len(ref) - 2, n), np.nan)
    for i in range(n):
        for k in range(1, len(ref) - 2):
            d = fires[i][(fires[i] >= ref[k] - T / 2) & (fires[i] < ref[k] + T / 2)] - ref[k]
            if len(d) == 1:
                F[k, i] = d[0]
    return F, T


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", type=int, default=394)
    ap.add_argument("--noise", default="")
    a = ap.parse_args()
    F, T = firing_matrix(np.load(IT / f"trace_{a.index:04d}{a.noise}.npz")["z"])
    n = F.shape[1]

    def wrap(d):
        return (d + T / 2) % T - T / 2

    def clusters(row):
        left, out = set(range(n)), []
        while left:
            i = left.pop()
            c = {j for j in left if abs(wrap(row[j] - row[i])) < 1e-3}
            left -= c
            out.append(tuple(sorted(c | {i})))
        return out

    memb = {k: {i: c for c in clusters(F[k]) for i in c} for k in range(len(F)) if not np.isnan(F[k]).any()}
    ks = sorted(memb)
    seen, rows = set(), []
    for a_, b_ in zip(ks[:-1], ks[1:]):
        for i in range(n):
            pa = memb[a_][i]
            if len(pa) < 10 or len(memb[b_][i]) >= 10:
                continue
            later = [k for k in ks if k > b_]
            for j, k in enumerate(later[:-10]):
                c = memb[k][i]
                if len(c) >= 10 and all(len(memb[later[j + u]][i]) >= 10 for u in range(10)):
                    if (set(c) & set(pa)) - {i} or (i, k) in seen:
                        break
                    seen.add((i, k))
                    big = [c2 for c2 in set(memb[a_].values()) if len(c2) >= 10]
                    ph = {c2: np.median(F[a_][list(c2)]) for c2 in big}
                    order = sorted(big, key=lambda c2: wrap(ph[c2] - ph[pa]) % T)
                    dst = [c2 for c2 in big if (set(c2) & set(c)) - {i}]
                    off = float(wrap(F[b_][i] - np.median(F[b_][[x for x in pa if x != i]])))
                    rows.append(dict(cycle=a_, element=i, parent_size=len(pa), offset=off,
                                     direction="advanced" if off < 0 else "retarded",
                                     destination=f"+{order.index(dst[0])}/{len(big)}" if dst else None,
                                     cycles_to_join=k - a_))
                    break
    summary = Counter((r["direction"], r["destination"]) for r in rows)
    print(a.index, "T", round(T, 4), "departures", len(rows), dict(summary))
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"ejections_{a.index:04d}{a.noise}.json").write_text(json.dumps(dict(
        period=T, departures=rows, summary={f"{k[0]} {k[1]}": v for k, v in summary.items()})))


if __name__ == "__main__":
    main()
