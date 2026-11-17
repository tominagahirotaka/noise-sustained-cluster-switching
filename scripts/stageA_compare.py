#!/usr/bin/env python3
"""Stage A of the tiny-noise experiment for run 88 (exploratory, 2026-10-25).

A2: residences of the big-size tuple (137,132,131) (clusters of >= 20 at 1e-3, single linkage; interruptions shorter
than 50 records bridged; residences of >= 200 records) and the spread of the 137 cluster at the start of each residence
(log10 of the median over its members of the median distance to the other members), with the same definitions
  - in the double-precision persistent-noise runs (switching_renoise.py --states; start t = 0) and
  - in the tree runs with kicks (noiseless_n50.py --eta, start record 1600 of the eta = 1e-8 run).
A1: the tree run with the new code and eta = 0 against the earlier noiseless run (first 20000 time units, bitwise).
A3: R1 events of the tracked pairs (5 vs 127: elements 8, 3; 12 vs 125: 5, 0): the jump of the tracked log10
distance across the event against the local slope, and the per-cycle minima, for join 1e-6, 1e-7 and an extra
absolute error of 1e-15 at every re-attachment.
  -> analysis/noiseless_n50/stageA.json

  python3 scripts/stageA_compare.py
"""
from pathlib import Path
import json
import sys

import numpy as np
from scipy.sparse.csgraph import connected_components

sys.path.insert(0, str(Path(__file__).resolve().parent))
from eps_three_cluster import ROOT  # noqa: E402

NL = ROOT / "analysis/noiseless_n50"
SC = ROOT / "analysis/switching_check"
TARGET = (137, 132, 131)


def residences(big):
    R = len(big)
    runs, k0 = [], 0
    for k in range(1, R + 1):
        if k == R or big[k] != big[k0]:
            runs.append((k0, k, big[k0]))
            k0 = k
    segs = []
    for r0, r1, b in runs:
        if b != TARGET:
            continue
        if segs and r0 - segs[-1][1] < 50:
            segs[-1][1] = r1
        else:
            segs.append([r0, r1])
    return [(s0, s1) for s0, s1 in segs if s1 - s0 >= 200]


def big_of(lab):
    return tuple(sorted((int(s) for s in np.bincount(lab) if s >= 20), reverse=True))


def double_run(eta):
    tag = "rn_0088" + ("" if eta == 1e-10 else f"_n{eta:.0e}") + "_t80000_states"
    zs = np.load(SC / f"{tag}_z.npy")
    n = zs.shape[1] // 2
    rows = []
    labs = []
    for z in zs:
        P = np.c_[z[:n], z[n:]]
        D = np.max(abs(P[:, None] - P[None]), axis=2)
        lab = connected_components(D < 1e-3, directed=False)[1]
        labs.append((lab, D))
    big = [big_of(l) for l, _ in labs]
    for s0, s1 in residences(big):
        lab, D = labs[s0]
        c = int(np.nonzero(np.bincount(lab) == 137)[0][0]) if 137 in np.bincount(lab) else None
        sp = None
        if c is not None:
            mem = np.nonzero(lab == c)[0]
            sub = D[np.ix_(mem, mem)]
            sp = float(np.log10(np.median([np.median(np.delete(sub[a], a)) for a in range(len(mem))])))
        rows.append(dict(start=10 * s0, L=10 * (s1 - s0), spread137_start=sp))
    return rows


def tree_run(path):
    d = np.load(path)
    g, sp, ev = d["group"], d["spread"].astype(float), int(d["every"])
    big = [big_of(l) for l in g]
    rows = []
    for s0, s1 in residences(big):
        lab = g[s0]
        cnt = np.bincount(lab)
        x = None
        if 137 in cnt:
            mem = np.nonzero(lab == int(np.nonzero(cnt == 137)[0][0]))[0]
            j = min(max((s0 + 1) // 10, 0), len(sp) - 1)  # spread every 10 records
            x = float(np.nanmedian(sp[j, mem]))
        rows.append(dict(start=ev * s0, L=ev * (s1 - s0), spread137_start=x))
    return rows


def a3(path):
    d = np.load(path)
    tr, ev = d["track"], d["r1_events"]
    out = dict(r1=len(ev), pairs=d["track_pairs"].tolist(), jumps=[])
    for q, (i, j) in enumerate(d["track_pairs"]):
        y = tr[:, q]
        for t, a, b, ld, size in ev:
            t = int(t)
            if t < 4 or t + 3 >= len(y):
                continue
            # the event happens in the bookkeeping of time unit t, before the track of unit t (index t - 1)
            before, after = y[t - 4:t - 1], y[t - 1:t + 2]
            if not (np.all(np.isfinite(before)) and np.all(np.isfinite(after))):
                continue
            slope = np.median(np.diff(np.r_[before, after]))
            jump = (after[0] - before[-1]) - slope
            out["jumps"].append(dict(pair=[int(i), int(j)], t=t, node=int(a), to=int(b), subtree=int(size),
                                     log10_d=float(ld), value=float(after[0]), jump_minus_slope=float(jump)))
        # per-cycle minima of the tracked distance (cycles ~ 20000; windows of 20000)
        out[f"minima_{i}_{j}"] = [float(np.nanmin(y[k:k + 20000])) for k in range(0, len(y) - 19999, 20000)]
    return out


def main():
    res = {}
    # A1
    new, old = np.load(NL / "nl_rn_0088_n1e-08_t80000_states_r1600_A1.npz"), \
        np.load(NL / "nl_rn_0088_n1e-08_t80000_states_r1600.npz")
    k = len(new["group"])
    res["A1"] = dict(records=k, group_equal=bool(np.array_equal(new["group"], old["group"][:k])),
                     logdev_equal=bool(np.array_equal(new["logdev"], old["logdev"][:k], equal_nan=True)),
                     spread_equal=bool(np.array_equal(new["spread"], old["spread"][:len(new["spread"])], equal_nan=True)))
    print("A1", res["A1"])
    # A2
    res["A2"] = {}
    for eta in (1e-10, 1e-12):
        dbl = double_run(eta)
        tre = {s: tree_run(NL / f"nl_rn_0088_n1e-08_t80000_states_r1600_e{eta:.0e}_s{s}.npz") for s in (1, 2, 3, 4)}
        res["A2"][f"{eta:.0e}"] = dict(double=dbl, tree=tre)
        print(f"A2 eta {eta:.0e}\n  double:", [(r['start'], r['L'], None if r['spread137_start'] is None else round(r['spread137_start'], 1)) for r in dbl])
        for s, rows in tre.items():
            print(f"  tree s{s}:", [(r['start'], r['L'], None if r['spread137_start'] is None else round(r['spread137_start'], 1)) for r in rows])
    # A3
    res["A3"] = {}
    for name in ("A3base", "j1e-07_A3", "jit1e-15_A3"):
        p = NL / f"nl_rn_0088_n1e-08_t80000_states_r1600_{name}.npz"
        r = a3(p)
        res["A3"][name] = r
        big = [x for x in r["jumps"] if abs(x["jump_minus_slope"]) > 0.05]
        print(f"A3 {name}: R1 events {r['r1']}, events on tracked pairs {len(r['jumps'])}, |jump - slope| > 0.05 decades: {len(big)}")
        for x in big[:10]:
            print("   ", x)
        for q in r["pairs"]:
            print("    minima", q, np.round(r[f"minima_{q[0]}_{q[1]}"], 2).tolist())
    # the decisive A3 test: the tracked distances of the variants against the base run, by depth of the base value
    # (the jump-against-local-slope test above is not informative: the tracked distance at integer times varies with the
    # fast phase, median |second difference| 0.015 decades, more during firing)
    base = np.load(NL / "nl_rn_0088_n1e-08_t80000_states_r1600_A3base.npz")["track"]
    res["A3"]["variant_vs_base"] = {}
    for name in ("j1e-07_A3", "jit1e-15_A3"):
        x = np.load(NL / f"nl_rn_0088_n1e-08_t80000_states_r1600_{name}.npz")["track"]
        rows = []
        for q in range(base.shape[1]):
            for lo, hi in ((-8, 0), (-12, -8), (-15, -12), (-17, -15)):
                sel = (base[:, q] >= lo) & (base[:, q] < hi)
                if sel.any():
                    dd = abs(x[sel, q] - base[sel, q])
                    rows.append(dict(pair=q, band=[lo, hi], n=int(sel.sum()), max=float(dd.max()), median=float(np.median(dd))))
        res["A3"]["variant_vs_base"][name] = rows
        print(f"A3 {name} vs base, max |d log10| by band:", [(r['pair'], r['band'], f"{r['max']:.1e}") for r in rows])
    (NL / "stageA.json").write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
