#!/usr/bin/env python3
"""Are the switches of the large-N persistent-noise runs a threshold effect? (exploratory, 2026-10-24)

Input: analysis/switching_check/rn_<index>_states_z.npy (full state every 10 time units, written by
switching_renoise.py --states). For each record: single-linkage clustering of the N points (x, y) with the max norm
(the metric of switching_renoise.labels); every merge of two groups of >= 2 elements is kept with its height (the
distance between the two groups) and the two sizes. From these:
  - switches (renoise_summary.residences) when clusters are cut at thresholds 1e-6, 1e-5, 1e-4, 1e-3, 1e-2;
  - heights of all joins (single elements included) onto groups of >= N/20 elements above 1e-7 (log10 histogram);
  - for the merge that forms a group of the given --size (the parent cluster, e.g. 328 for run 46), the time series
    of its height: the distance between the sub-cluster and the rest of the parent.
  python3 scripts/renoise_threshold_check.py --index 46 --size 328  -> analysis/switching_check/threshold_<index>.json
"""
from pathlib import Path
import argparse
import json
import sys

import numpy as np
from scipy.cluster.hierarchy import linkage

sys.path.insert(0, str(Path(__file__).resolve().parent))
from renoise_summary import D, residences  # noqa: E402

THS = (1e-6, 1e-5, 1e-4, 1e-3, 1e-2)


def merges(z):
    n = z.size // 2
    Z = linkage(np.c_[z[:n], z[n:]], method="single", metric="chebyshev")
    size = np.r_[np.ones(n, int), Z[:, 3].astype(int)]
    big = [(float(h), int(size[int(a)]), int(size[int(b)])) for a, b, h, _ in Z
           if size[int(a)] >= 2 and size[int(b)] >= 2]
    # joins of anything (also single elements) onto a group of >= N/20 elements, above 1e-7
    joins = [float(h) for a, b, h, _ in Z if max(size[int(a)], size[int(b)]) >= n // 20 and h > 1e-7]
    return big, joins, Z


def cut(Z, n, th):
    # labels from the linkage at threshold th (union-find over merges below th)
    p = list(range(2 * n - 1))

    def f(i):
        while p[i] != i:
            p[i] = p[p[i]]
            i = p[i]
        return i
    for k, (a, b, h, _) in enumerate(Z):
        if h < th:
            p[f(int(a))] = n + k
            p[f(int(b))] = n + k
    return np.array([f(i) for i in range(n)])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", type=int, required=True)
    ap.add_argument("--size", type=int, required=True)
    a = ap.parse_args()
    zs = np.load(D / f"rn_{a.index:04d}_states_z.npy")
    n = zs.shape[1] // 2
    labs = {th: [] for th in THS}
    gap, joins = [], []
    for z in zs:
        m, j, Z = merges(z)
        joins += j
        for th in THS:
            labs[th].append(cut(Z, n, th))
        hs = [h for h, s1, s2 in m if s1 + s2 == a.size]
        gap.append(min(hs) if hs else np.nan)
    sw = {}
    for th in THS:
        res = residences(np.asarray(labs[th]))
        sw[f"{th:g}"] = sum(x[2] != y[2] for x, y in zip(res[:-1], res[1:]))
    g = np.asarray(gap)
    lg = np.log10(g[np.isfinite(g)])
    out = dict(index=a.index, N=n, size=a.size, switches_by_threshold=sw, gap=g.tolist(),
               gap_log10_quantiles={q: float(np.quantile(lg, q)) for q in (0, .05, .25, .5, .75, .95, 1)} if lg.size else None,
               gap_found_fraction=float(np.isfinite(g).mean()),
               joins_log10_hist=np.histogram(np.log10(joins), bins=np.arange(-7, 0.5, .5))[0].tolist() if joins else None)
    (D / f"threshold_{a.index:04d}.json").write_text(json.dumps(out))
    print(a.index, "switches", sw, "gap found", round(out["gap_found_fraction"], 3), "log10 gap quantiles",
          {k: round(v, 2) for k, v in (out["gap_log10_quantiles"] or {}).items()})
    print("  joins log10 histogram (-7 to 0 by 0.5):", out["joins_log10_hist"])


if __name__ == "__main__":
    main()
