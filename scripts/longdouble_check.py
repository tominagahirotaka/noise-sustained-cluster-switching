#!/usr/bin/env python3
"""Independent precision check of the tree representation (noiseless_n50.py --states) with plain RK4 in extended
precision (numpy longdouble, 64-bit mantissa on x86, relative 1e-19) (exploratory, 2026-10-24).

Same start (record --t0 of analysis/switching_check/<states>_z.npy), no noise, RK4 h = 0.005, --tmax time units.
Every 100 time units: groups at 1e-3 (absolute positions), and for each element of a group of >= 5 the log10 median
of its max-norm distances to the other members (the "spread" of noiseless_n50.py). Compared with the tree run
analysis/noiseless_n50/nl_<states>_r<t0>.npz (spread every 10 * every time units) at the same times, for elements whose
two values are both above --floor (default 1e-18, a decade above the longdouble resolution).
  -> analysis/noiseless_n50/longdouble_<states>_r<t0>.json (spreads cached in ..._spread.npy; --compare_only reuses them)

  python3 scripts/longdouble_check.py --states rn_0088_n1e-08_t80000_states --t0 1600 --tmax 20000
"""
from pathlib import Path
import argparse
import json
import sys

import numpy as np
from scipy.sparse.csgraph import connected_components

sys.path.insert(0, str(Path(__file__).resolve().parent))
from eps_three_cluster import ROOT  # noqa: E402

B, GAM = np.longdouble(.7), np.longdouble(.5)


def spread(x, y, th=1e-3):
    n = x.size
    d = np.maximum(abs(x[:, None] - x[None]), abs(y[:, None] - y[None]))
    g = connected_components(np.asarray(d, float) < th, directed=False)[1]
    out = np.full(n, np.nan)
    for c in np.unique(g):
        mem = np.nonzero(g == c)[0]
        if mem.size >= 5:
            sub = d[np.ix_(mem, mem)]
            for a, k in enumerate(mem):
                out[k] = float(np.log10(np.median(np.delete(sub[a], a))))
    return out, g


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--states", required=True)
    ap.add_argument("--t0", type=int, required=True)
    ap.add_argument("--tmax", type=int, default=20000)
    ap.add_argument("--floor", type=float, default=1e-18)
    ap.add_argument("--compare_only", action="store_true", help="reuse the saved longdouble spreads")
    a = ap.parse_args()
    SC = ROOT / "analysis/switching_check"
    js = json.loads((SC / f"{a.states}.json").read_text())
    e, K = np.longdouble(js["eps"]), np.longdouble(js["K"])
    z = np.load(SC / f"{a.states}_z.npy")[a.t0].astype(np.longdouble)
    n = z.size // 2
    h = np.longdouble(.005)

    def f(v):
        x, y = v[:n], v[n:]
        return np.concatenate([(x - x ** 3 / 3 - y) / e + K * (x.mean() - x), e * (x + B - GAM * y)])

    OUT = ROOT / "analysis/noiseless_n50"
    cache = OUT / f"longdouble_{a.states}_r{a.t0}_spread.npy"
    ld = [] if not (a.compare_only and cache.exists()) else None
    for t in range(1, (a.tmax if ld is not None else 0) + 1):
        for _ in range(200):
            k1 = f(z)
            k2 = f(z + h / 2 * k1)
            k3 = f(z + h / 2 * k2)
            k4 = f(z + h * k3)
            z = z + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
        if t % 100 == 0:
            ld.append(spread(z[:n], z[n:])[0])
            if t % 5000 == 0:
                print("t", t, flush=True)
    if ld is not None:
        ld = np.asarray(ld)
        np.save(cache, ld)
    else:
        ld = np.load(cache)
    if not (OUT / f"nl_{a.states}_r{a.t0}.npz").exists():
        print("tree run not finished; spreads saved to", cache)
        return
    nl = np.load(ROOT / f"analysis/noiseless_n50/nl_{a.states}_r{a.t0}.npz")
    ev = int(nl["every"])
    sp = nl["spread"]  # row k at time (k + 1) * 10 * ev
    rows = []
    lf = np.log10(a.floor)
    for k in range(len(ld)):
        t = 100 * (k + 1)
        j = t // (10 * ev) - 1
        if t % (10 * ev) or j >= len(sp):
            continue
        A, Bt = ld[k], sp[j].astype(float)
        ok = np.isfinite(A) & np.isfinite(Bt) & (A > lf) & (Bt > lf)
        if ok.any():
            d = A[ok] - Bt[ok]
            rows.append(dict(t=t, n=int(ok.sum()), min_log10=float(min(A[ok].min(), Bt[ok].min())),
                             max_abs_diff=float(abs(d).max()), median_abs_diff=float(np.median(abs(d)))))
    deep = [r for r in rows if r["min_log10"] < -16]
    out = dict(states=a.states, t0=a.t0, tmax=a.tmax, floor=a.floor, rows=rows,
               max_abs_diff_all=max(r["max_abs_diff"] for r in rows) if rows else None,
               max_abs_diff_below_1e16=max(r["max_abs_diff"] for r in deep) if deep else None)
    (ROOT / f"analysis/noiseless_n50/longdouble_{a.states}_r{a.t0}.json").write_text(json.dumps(out, indent=1))
    print("compared", len(rows), "times; max |diff| (decades)", out["max_abs_diff_all"], "; at times with values below 1e-16:",
          out["max_abs_diff_below_1e16"], len(deep), "times", flush=True)


if __name__ == "__main__":
    main()
