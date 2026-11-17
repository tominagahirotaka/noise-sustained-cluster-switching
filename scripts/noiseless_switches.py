#!/usr/bin/env python3
"""Residences and switches in the noiseless runs of noiseless_n50.py (exploratory, 2026-10-23).

Every 10 time units the big groups (>= 10 elements at threshold 1e-3) are taken as a set of member sets; a residence
is a run of identical sets lasting >= 200 time units. For each residence that ends with exactly one element leaving
the big groups: its length L, the group sizes, the leaving element, its rank by spread in its group at the start
(0 = largest; spread = median log10 tree distance to the other members, recorded by noiseless_n50.py), the largest
spread in that group at the start (log10 d0), the growth rate of the leaving element's spread over the residence,
and -log(d0)/rate, the residence predicted from linear growth.

  python3 scripts/noiseless_switches.py 379 386 393 394 397  -> analysis/noiseless_n50/residences.json
"""
from pathlib import Path
import json
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from eps_three_cluster import ROOT  # noqa: E402

D = ROOT / "analysis/noiseless_n50"


def residences(g, hold=200):
    st = []
    for t in range(9, len(g), 10):
        L = g[t]
        st.append((t + 1, frozenset(frozenset(np.nonzero(L == x)[0].tolist()) for x in np.unique(L)
                                    if np.sum(L == x) >= 10)))
    out, k = [], 0
    while k < len(st):
        j = k
        while j + 1 < len(st) and st[j + 1][1] == st[k][1]:
            j += 1
        if st[j][0] - st[k][0] >= hold:
            out.append((st[k][0], st[j][0], st[k][1]))
        k = j + 1
    return out


def analyze(index):
    f = np.load(D / f"nl_{index:04d}_t5000.npz")
    g, sp = f["group"], f["spread"]
    pers = residences(g)
    rows = []
    for (a0, a1, A), (_, _, B) in zip(pers[:-1], pers[1:]):
        ha = {i: c for c in A for i in c}
        hb = {i for c in B for i in c}
        out = sorted(set(ha) - hb)
        if len(out) != 1 or a1 - a0 < 1500:
            continue
        e = out[0]
        U = sorted(ha[e])
        r, r2 = (a0 + 200) // 10, (a1 - 300) // 10
        rate = float((sp[r2, e] - sp[r, e]) * np.log(10) / ((r2 - r) * 10))
        d0 = float(np.nanmax(sp[r, U]))
        rows.append(dict(start=int(a0), end=int(a1), L=int(a1 - a0), sizes=sorted(map(len, A), reverse=True),
                         leaver=e, rank=int(np.sum(sp[r, U] > sp[r, e])), log10_d0=d0, rate=rate,
                         L_pred=float(-d0 * np.log(10) / rate)))
    return dict(index=index, eK=float(f["eps"] * f["K"]), residences=rows)


def main():
    res = {}
    for a in sys.argv[1:]:
        r = analyze(int(a))
        res[a] = r
        print(a, "eK", round(r["eK"], 3))
        for x in r["residences"]:
            print(f"  {x['start']:7d}-{x['end']:7d} L {x['L']:6d} sizes {x['sizes']} leaver {x['leaver']:2d} rank"
                  f" {x['rank']} log10 d0 {x['log10_d0']:8.1f} rate {x['rate']:.4f} L_pred {x['L_pred']:8.0f}")
    (D / "residences.json").write_text(json.dumps(res))


if __name__ == "__main__":
    main()
