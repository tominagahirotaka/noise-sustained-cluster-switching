#!/usr/bin/env python3
"""Switches in the persistent-noise runs of switching_renoise.py (exploratory, 2026-10-24).

From analysis/switching_check/rn_<index>.npz (cluster labels every 10 time units, threshold 1e-4): the clusters of
>= 2 elements are taken as a set of member sets; a residence is a run of identical sets lasting >= 200 time units;
a switch is a change between consecutive residences. Counted over the whole run and over its second half.
Sustained switching: >= 3 switches in the second half. Only the default noise (rn_<index>.json without _n suffix).

  python3 scripts/renoise_summary.py  -> analysis/switching_check/renoise_summary.json
"""
from pathlib import Path
import collections
import json
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from noiseless_switches import ROOT  # noqa: E402

D = ROOT / "analysis/switching_check"


def residences(lab, hold=200):
    st = [(10 * (k + 1), frozenset(frozenset(np.nonzero(L == x)[0].tolist()) for x in np.unique(L)
                                   if np.sum(L == x) >= 2)) for k, L in enumerate(lab)]
    out, k = [], 0
    while k < len(st):
        j = k
        while j + 1 < len(st) and st[j + 1][1] == st[k][1]:
            j += 1
        if st[j][0] - st[k][0] >= hold:
            out.append((st[k][0], st[j][0], st[k][1]))
        k = j + 1
    return out


def main():
    sel = json.loads((D / "selected.json").read_text())
    rows = []
    for f in sorted(D.glob("rn_[0-9][0-9][0-9][0-9].json")):
        s = json.loads(f.read_text())
        lab = np.load(f.with_suffix(".npz"))["labels"]
        res = residences(lab)
        sw = [b[0] for a, b in zip(res[:-1], res[1:]) if a[2] != b[2]]
        half = s["tmax"] / 2
        late = sum(t > half for t in sw)
        cnt = collections.Counter(tuple(sorted(map(len, r[2]), reverse=True)) for r in res for _ in range((r[1] - r[0]) // 10 + 1))
        rows.append(dict(index=s["index"], N=s["N"], eps=s["eps"], eK=round(s["eps"] * s["K"], 3),
                         cls=sel[s["index"]].get("class"), switches=len(sw), switches_late=late,
                         lyap_late=s["lyap_late_half"], start=s["sw_final"], final=s["final_sizes"],
                         most=list(cnt.most_common(1)[0][0]) if cnt else None, sustained=late >= 3))
    (D / "renoise_summary.json").write_text(json.dumps(rows, indent=1))
    print(len(rows), "runs;", sum(r["sustained"] for r in rows), "sustained")
    tab = collections.defaultdict(lambda: [0, 0])
    for r in rows:
        k = (r["N"], r["eps"], r["eK"])
        tab[k][0] += 1
        tab[k][1] += r["sustained"]
    for k in sorted(tab):
        if tab[k][1]:
            print(k, "runs", tab[k][0], "sustained", tab[k][1])
    for r in rows:
        if r["sustained"]:
            print(f"  {r['index']:4d} N{r['N']} eps {r['eps']} eK {r['eK']} sw {r['switches']} late {r['switches_late']}"
                  f" lyap {r['lyap_late']:.4f} most {r['most'][:6]}")


if __name__ == "__main__":
    main()
