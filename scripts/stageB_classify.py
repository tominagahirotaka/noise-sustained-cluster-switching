#!/usr/bin/env python3
"""Per-cycle description and classification of the Stage B tree runs of run 88 (exploratory, 2026-10-25).

Input: tree runs nl_<states>_r<t0>[_e<eta>_s<seed>].npz of noiseless_n50.py (group labels every `every` time units,
spreads every 10 * every). Reference clusters: the groups of >= 20 elements at the first record (137, 132, 131).

  - Window partition: windows of --win records (default 50, i.e. 500 time units); two elements are linked when they are
    in the same 1e-3 group in at least half of the records of the window; the window partition is the connected
    components of the links. This removes the flicker of the 1e-3 groups with the fast phase.
  - Window code of a reference cluster R (by its member elements, fixed from the first record):
      I  intact (one group);  S  two groups of >= 2 elements, the smaller >= 3, no other group of >= 2;
      F  three or more groups of >= 2 elements;  x  otherwise (one or two groups of >= 2 plus single elements).
  - Cycle: starts at a window where the 137 cluster leaves I after >= --min_intact windows of I.
    Per cycle: start time, length, number of F windows of 137 and of 132, the largest number of groups (>= 2) of 137,
    the small group of 137 in its first S window (element ids), the small group of 132 in its last S window before it
    is intact again, and the depths: the minimum over the cycle of the median spread (log10) of these small groups'
    elements while they are inside their parent cluster.
  - Cycle length: from the split start at the record resolution (the first record at which 137 is in one 1e-3 group in
    fewer than half of the next --win records, interpolated linearly) to the next one.
  - Cycle level (author, 2026-10-25): a deterministic stretch is >= 3 consecutive cycles with the same small groups of
    137 and of 132, no F window of 137, every length within 5% and every 132 small-group depth within +-0.5 decades of
    the medians over the run of cycles with these sets; its cycles are labeled D. Other cycles: F (an F window of 137)
    or c (no F window, not in a deterministic stretch; e.g. clean two-way splits by changing sets).
  - Run level, cycles starting at t >= --judge_from (default 150000), at least 3 of them:
      Deterministic-cycle-like: more than half of the cycles are D;
      Fragmented-switching: at least half are F (and fewer than half D). F means a fragmented cycle, not a noise-
        induced one: the first (transient) cycle of the noiseless reference runs is F too;
      Other / unresolved: otherwise (fewer than 3 cycles, clean splits by changing sets, or a mixture).
  Depths use only the spread times at which the small group is, in that record, in the same 1e-3 group as the rest of
  its parent (else the spread is the small group's internal one; RESULTS_NOTES 2026-10-25, the 12-subset).
  Switch history over the whole run: the times at which the cycle label changes; the deterministic stretches.
  -> <out>.json

  python3 scripts/stageB_classify.py analysis/noiseless_n50/nl_*_e1e-1?_s?.npz --out analysis/noiseless_n50/stageB_classify.json
"""
from pathlib import Path
import argparse
import json

import numpy as np
from scipy.sparse.csgraph import connected_components


def window_partitions(g, win):
    n = g.shape[1]
    out = []
    for w in range(len(g) // win):
        M = np.zeros((n, n), np.int32)
        for r in g[w * win:(w + 1) * win]:
            M += r[:, None] == r[None, :]
        out.append(connected_components(M >= win / 2, directed=False)[1])
    return np.array(out)


def code(lab, mem):
    labs, cc = np.unique(lab[mem], return_counts=True)
    order = np.argsort(-cc)
    labs, cc = labs[order], cc[order]
    big = cc >= 2
    if len(cc) == 1:
        return "I", None
    if big.sum() >= 3:
        return "F", None
    if big.sum() == 2 and cc[1] >= 3:
        return "S", mem[lab[mem] == labs[1]]
    return "x", None


def run(path, win, min_intact, judge_from):
    d = np.load(path)
    g = d["group"].astype(int)
    ev = int(d["every"])
    sp = d["spread"].astype(float)  # row j at time 10 * ev * (j + 1)
    g0 = g[0]
    c = np.bincount(g0)
    refs = sorted(((int(c[l]), np.nonzero(g0 == l)[0]) for l in np.unique(g0) if c[l] >= 20), key=lambda x: -x[0])
    (n137, m137), (n132, m132) = refs[0], refs[1]
    P = window_partitions(g, win)
    W = len(P)
    tw = ev * win * np.arange(W)  # window start times
    cA, cB, sA, sB, ngA = [], [], [], [], []
    for lab in P:
        a, sa = code(lab, m137)
        b, sb = code(lab, m132)
        cA.append(a); cB.append(b); sA.append(sa); sB.append(sb)
        _, cc = np.unique(lab[m137], return_counts=True)
        ngA.append(int((cc >= 2).sum()))
    rpw = win // 10  # spread rows per window

    def depth(members, w0, w1, parent):
        """Min over [w0, w1) of the median spread (log10) of the small group's elements, at the spread times at which
        all of them are in the same 1e-3 group as the rest of the parent cluster in that record (otherwise the spread
        would be the small group's internal one)."""
        rest = np.setdiff1d(parent, members)
        best = np.inf
        for j in range(w0 * rpw, min(w1 * rpw, len(sp))):
            r = g[10 * (j + 1) - 1]
            lab = np.bincount(r[rest]).argmax()
            if np.all(r[members] == lab):
                x = sp[j, members]
                if np.isfinite(x).any():
                    best = min(best, float(np.nanmedian(x)))
        return None if not np.isfinite(best) else best

    starts = []
    k = 0
    for w in range(1, W):
        if cA[w - 1] == "I":
            k += 1
        else:
            k = 0
        if cA[w] != "I" and cA[w - 1] == "I" and k >= min_intact:
            starts.append(w)
    # split start at the record resolution: first record at which the fraction of records with 137 in one 1e-3 group,
    # over the next `win` records, falls below 1/2 (linear interpolation between records), near the window start
    intact_r = np.array([len(np.unique(r[m137])) == 1 for r in g], float)
    cs = np.r_[0., np.cumsum(intact_r)]
    frac = (cs[win:] - cs[:-win]) / win  # frac[k]: records k .. k + win - 1

    def split_time(w0):
        for k in range(max(1, win * (w0 - 2)), min(len(frac), win * (w0 + 1))):
            if frac[k - 1] >= .5 > frac[k]:
                x = (frac[k - 1] - .5) / (frac[k - 1] - frac[k])
                return ev * (k + x)  # record k is at time ev * (k + 1); the window starts at its first record
        return float(tw[w0])

    ts = [split_time(w) for w in starts]
    cycles = []
    for i, (w0, w1) in enumerate(zip(starts[:-1], starts[1:])):
        small137 = next((sA[w] for w in range(w0, w1) if cA[w] == "S"), None)
        # last S window of 132 before it is intact again (the regathering small group)
        small132 = None
        for w in range(w0, w1):
            if cB[w] == "S":
                small132 = sB[w]
        prev = starts[i - 1] if i > 0 else 0
        d12 = depth(small137, prev, w0, m137) if small137 is not None else None
        d5 = depth(small132, w0, w1, m132) if small132 is not None else None
        cycles.append(dict(
            t=int(tw[w0]), t_split=float(ts[i]), T=float(ts[i + 1] - ts[i]), T_win=int(tw[w1] - tw[w0]),
            F137=int(sum(cA[w] == "F" for w in range(w0, w1))), F132=int(sum(cB[w] == "F" for w in range(w0, w1))),
            maxgroups137=int(max(ngA[w0:w1])),
            small137=None if small137 is None else sorted(int(x) for x in small137),
            small132=None if small132 is None else sorted(int(x) for x in small132),
            depth137small=d12, depth132small=d5))
    # cycle level: deterministic stretches = >= 3 consecutive cycles with the same small groups of 137 and 132 (both
    # found), no F window of 137, each length within 5% and each 132 small-group depth within 0.5 decades of the
    # medians over the run of cycles with these sets
    for cy in cycles:
        cy["label"] = "F" if cy["F137"] > 0 else "c"  # c: no F window, not (yet) in a deterministic stretch
    stretches = []
    i = 0
    while i < len(cycles):
        key = (cycles[i]["small137"], cycles[i]["small132"])
        j = i
        while (j < len(cycles) and cycles[j]["F137"] == 0 and None not in key
               and (cycles[j]["small137"], cycles[j]["small132"]) == key and cycles[j]["depth132small"] is not None):
            j += 1
        if j - i >= 3:
            Ts = np.array([c["T"] for c in cycles[i:j]])
            ds = np.array([c["depth132small"] for c in cycles[i:j]])
            ok = (abs(Ts - np.median(Ts)) <= .05 * np.median(Ts)) & (abs(ds - np.median(ds)) <= .5)
            k = i
            while k < j:
                if not ok[k - i]:
                    k += 1
                    continue
                k1 = k
                while k1 < j and ok[k1 - i]:
                    k1 += 1
                if k1 - k >= 3:
                    stretches.append(dict(first=k, last=k1 - 1, t0=cycles[k]["t_split"], t1=cycles[k1 - 1]["t_split"]
                                          + cycles[k1 - 1]["T"], n=k1 - k))
                    for c in cycles[k:k1]:
                        c["label"] = "D"
                k = k1
        i = max(j, i + 1)
    # run level, cycles starting at t >= judge_from
    late = [cy for cy in cycles if cy["t"] >= judge_from]
    nD = sum(c["label"] == "D" for c in late)
    nF = sum(c["label"] == "F" for c in late)
    if len(late) < 3:
        cls = "Other/unresolved"
    elif nD > len(late) / 2:
        cls = "Deterministic-cycle-like"
    elif nF >= len(late) / 2 and nD < len(late) / 2:
        cls = "Fragmented-switching"
    else:
        cls = "Other/unresolved"
    why = dict(n_cycles=len(late), n_D=nD, n_F=nF, n_c=len(late) - nD - nF)
    # switches: times at which the cycle label changes
    switches = [[cycles[0]["t_split"], cycles[0]["label"]]] if cycles else []
    for c in cycles[1:]:
        if c["label"] != switches[-1][1]:
            switches.append([c["t_split"], c["label"]])
    codes = "".join(a + b for a, b in zip(cA, cB))
    return dict(file=Path(path).name, eta=float(d["eta"]) if "eta" in d.files else 0., seed=int(d["seed"]) if "seed" in d.files else None, refs=[n137, n132],
                window=int(ev * win), classification=cls, criteria=why, labels="".join(c["label"] for c in cycles),
                switches=switches, stretches=stretches, cycles=cycles, window_codes=codes)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="+")
    ap.add_argument("--win", type=int, default=50)
    ap.add_argument("--min_intact", type=int, default=4)
    ap.add_argument("--judge_from", type=int, default=150000)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    res = [run(f, a.win, a.min_intact, a.judge_from) for f in a.files]
    Path(a.out).write_text(json.dumps(dict(win=a.win, min_intact=a.min_intact, judge_from=a.judge_from, runs=res), indent=1))
    for r in res:
        late = [c for c in r["cycles"] if c["t"] >= a.judge_from]
        print(f"{r['file'][-28:]:>28} {r['classification']:<26} cycles(late) {len(late):2d} "
              f"{r['criteria']} T {[round(c['T']) for c in late]} labels {r['labels']}")


if __name__ == "__main__":
    main()
