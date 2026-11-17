#!/usr/bin/env python3
"""Switches in the runs of first_departure_n50.py: how many elements move, first switch and later ones (2026-10-27).

The distance of two elements at a saved sample depends strongly on the phase (during a spike a small offset gives a
large difference in x), so distances are taken as the minimum over each period of the node (period-min). A first
version measured instantaneous distances and found all 16 members of U "departing" at once; that was this artefact.
  - per period, the period-min pairwise distance max(|dx|, |dy|) of all elements; labels: connected components at
    --th (1e-4, as B2), clusters of >= 2 elements;
  - residences (renoise_summary.residences, constant partition for >= --hold time units) and switches with the
    element moves (n50_web_analysis.persistent, as B2); every switch is classified: "swap" = one element leaves a
    cluster and one singleton joins the same cluster (the reduced connection), otherwise by the numbers of elements
    out to singletons, in from singletons and cluster to cluster;
  - the spread of the original U: per period the largest period-min distance of its members from their median;
    growth rate of log(spread) fitted between 1e-5 and 1e-3 (float32 states: below ~1e-7 is not resolved), also
    on 1e-5..1e-4 (growth_U_lo) and 1e-4..1e-3 (growth_U_hi) separately (is the fit range linear?);
  - W: the same spread for the members of U that stay together through the first switch (log10 at every period: its
    rise and fall up to the second switch, t1 < t < t2; below ~1e-6 is not reliable).
  - b2style_kinds: the same run counted as B2 counted the long runs (instantaneous labels every 10 time units at
    --th, any phase), to see whether the phase dependence changes the kinds of switches. A finite-time change, not a transverse Lyapunov exponent.
Summary per group (node, perturbation): kinds of the first switch and of the later ones.
  -> analysis/heteroclinic_n50/firstdep/first_departure_summary.json

  python3 scripts/first_departure_analysis.py
"""
from collections import Counter, defaultdict
from pathlib import Path
import argparse
import json
import sys

import numpy as np
from scipy.sparse.csgraph import connected_components

sys.path.insert(0, str(Path(__file__).resolve().parent))
from renoise_summary import residences  # noqa: E402
from n50_web_analysis import persistent  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
FD = ROOT / "analysis/heteroclinic_n50/firstdep"


def kind(s):
    mv = s["moves"]
    if len(mv) >= 10:
        return "dissolve"
    out = [(e, a) for e, a, b in mv if b == "free" and a != "free"]
    inn = [(e, b) for e, a, b in mv if a == "free" and b != "free"]
    cc = [x for x in mv if "free" not in x]
    if len(out) == 1 and len(inn) == 1 and not cc and out[0][1] == inn[0][1]:
        return "swap"
    return f"out{len(out)}_in{len(inn)}_cc{len(cc)}"


def fit(t, y, lo, hi):
    k = (y > np.log(lo)) & (y < np.log(hi))
    return float(np.polyfit(t[k], y[k], 1)[0]) if k.sum() > 3 else None


def one(path, th, hold, stat="min"):
    d = np.load(path)
    Z, member, U, ev, T = d["z"].astype(float), d["member"], int(d["U"]), int(d["every"]), float(d["period"])
    n = member.size
    w = int(round(T / ev))
    nc = len(Z) // w
    P = Z[:nc * w].reshape(nc, w, 2 * n)
    Dp = np.maximum(abs(P[:, :, :n, None] - P[:, :, None, :n]), abs(P[:, :, n:, None] - P[:, :, None, n:]))
    Dm = Dp.min(axis=1) if stat == "min" else np.median(Dp, axis=1)
    lab = np.array([connected_components(Dm[c] < th, directed=False)[1] for c in range(nc)])
    # residences() takes samples 10 time units apart; one period is one sample here, so hold is given in periods x 10
    res = residences(lab, hold=10 * max(1, int(round(hold / T))))
    sw = persistent(res) if len(res) > 1 else []
    ui = np.nonzero(member == U)[0]
    mx, my = np.median(P[:, :, ui], axis=2), np.median(P[:, :, n + ui], axis=2)
    du = np.maximum(abs(P[:, :, ui] - mx[:, :, None]), abs(P[:, :, n + ui] - my[:, :, None])).min(axis=1)
    tp = np.arange(nc) * T
    spread = np.log(np.maximum(du.max(axis=1), 1e-300))
    pre = tp < (sw[0]["t"] / 10 * T if sw else tp[-1])
    r = dict(file=path.name, periods=nc, residences=len(res), switches=len(sw), kinds=[kind(s) for s in sw],
             moves=[len(s["moves"]) for s in sw], t_switch=[round(s["t"] / 10 * T, 1) for s in sw],
             growth_U=fit(tp[pre], spread[pre], 1e-5, 1e-3), growth_U_lo=fit(tp[pre], spread[pre], 1e-5, 1e-4),
             growth_U_hi=fit(tp[pre], spread[pre], 1e-4, 1e-3),
             log10_spread_U=np.round(spread / np.log(10), 2).tolist())
    # the same run counted as B2 did: instantaneous labels every 10 time units at --th, whatever the phase
    lab10 = np.array([connected_components(np.maximum(abs(z[:n, None] - z[None, :n]), abs(z[n:, None] - z[None, n:]))
                                           < th, directed=False)[1] for z in Z[10 // ev::10 // ev]])
    res10 = residences(lab10, hold=hold)
    sw10 = persistent(res10) if len(res10) > 1 else []
    r["b2style_kinds"] = [kind(s) for s in sw10]
    if sw:
        moved = {e for e, a, b in sw[0]["moves"]}
        stay = np.array([k for k, i in enumerate(ui) if i not in moved])
        r["first_moves"] = [[int(e), str(a), str(b)] for e, a, b in sw[0]["moves"]]
        r["first_U_elements_moved"] = int(len(ui) - len(stay))
        if len(stay) > 1:
            lw = np.log10(np.maximum(du[:, stay].max(axis=1), 1e-300))
            k2 = int(sw[1]["t"] / 10) if len(sw) > 1 else nc  # only up to the second switch (t1 < t < t2)
            kp = int(np.argmax(lw[:k2]))
            r["log10_W_stay"] = np.round(lw, 2).tolist()
            r["W_peak"] = dict(t=round(kp * T, 1), log10=float(lw[kp]))
            below = np.nonzero((np.arange(nc) > kp) & (np.arange(nc) < k2) & (lw < -6))[0]
            r["W_back_below_1e-6_after"] = round((below[0] - kp) * T, 1) if len(below) else None
    return r


def med(x):
    x = [v for v in x if v is not None]
    return float(np.median(x)) if x else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--th", type=float, default=1e-4)
    ap.add_argument("--hold", type=float, default=200.)
    ap.add_argument("--stat", choices=["min", "median"], default="min",
                    help="period statistic of the pairwise distance for the labels (median: a single close pass in a "
                         "period does not count as being together)")
    ap.add_argument("--glob", default="fd_*.npz")
    ap.add_argument("--out", default="first_departure_summary.json")
    a = ap.parse_args()
    rows = [one(p, a.th, a.hold, a.stat) for p in sorted(FD.glob(a.glob))]
    groups = defaultdict(list)
    for r in rows:
        f = r["file"][3:-4]
        groups[f.rsplit("_r", 1)[0] if ("_noise_" in f or "_init" in f) else f].append(r)
    summ = {}
    for k, g in sorted(groups.items()):
        s = dict(runs=len(g), first_switch=dict(Counter(r["kinds"][0] if r["kinds"] else "none" for r in g)),
                 later_switches=dict(Counter(x for r in g for x in r["kinds"][1:])),
                 all_switches=dict(Counter(x for r in g for x in r["kinds"])),
                 b2style_all_switches=dict(Counter(x for r in g for x in r["b2style_kinds"])),
                 switches_per_run=dict(sorted(Counter(r["switches"] for r in g).items())),
                 median_growth_U=med(r["growth_U"] for r in g),
                 median_growth_U_halves=[med(r["growth_U_lo"] for r in g), med(r["growth_U_hi"] for r in g)],
                 median_W_peak_log10=med(r.get("W_peak", {}).get("log10") for r in g),
                 median_W_back_below_1e6=med(r.get("W_back_below_1e-6_after") for r in g),
                 W_not_back=sum(1 for r in g if "W_peak" in r and r.get("W_back_below_1e-6_after") is None))
        summ[k] = s
        print(k, s, flush=True)
    (FD / a.out).write_text(json.dumps(dict(args=vars(a), summary=summ, runs=rows)))


if __name__ == "__main__":
    main()
