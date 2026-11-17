#!/usr/bin/env python3
"""Paper 3, the N = 50 switching network from existing data (2026-10-27; DECISIONS 2026-10-27, design (B), B1-B3).

Data: the persistent-noise reruns of the 418 excluded runs (switching_renoise.py, eta = 1e-10, 20000 time units;
analysis/switching_check/rn_<index>.npz: cluster labels of the 50 elements every 10 time units, threshold 1e-4) and
their summary (renoise_summary.json, residences of >= 200 time units as in renoise_summary.py), and the transverse
stability map of the balanced partitions (analysis/heteroclinic_n50/n50_grid.json).

B1 map: for every (eps, eps K) of the N = 50 excluded runs, the number of runs and of runs with sustained switching
   (>= 3 switches in the second half), their most frequent partitions, and the grid prediction: the number of
   transversely unstable clusters of size >= 2 on (17,17,16) and (17,16,16,1) (over the seeds of the grid), with the
   largest internal exponent.
B2 order of the switches (sustained N = 50 runs): consecutive residences are compared; clusters (>= 2 elements) are
   matched across a switch one to one by the assignment of largest total overlap (a greedy match mislabelled clusters
   when the partition type changed; corrected 2026-10-27) and given persistent identities; an element that changes its cluster
   gives a directed transition (donor -> receiver, a singleton counts as "free"). Per run: elements moved per switch,
   the fraction of elements that changed cluster at least once and how many distinct clusters each visited, the
   directed transition counts between persistent clusters (a cyclic order shows as one direction dominating), and
   the "relay": the fraction of switches whose donor is the receiver of the previous switch. A switch with >= 10
   moves is a large cluster that loses coherence at the label threshold (all members become singletons) and re-forms
   later with nearly the same members (overlap >= 80%; it keeps its identity); these "dissolutions" are counted
   separately and left out of the transitions. Element transitions: the cluster an element leaves and the next
   cluster it joins, skipping its time as a singleton; with three clusters, the counts in the two cyclic directions.
B3 residence times: per run the residence durations (time units), their median, and median x lambda_u with
   lambda_u the largest internal exponent of the grid at that point (for the partition type of the run when it is
   on the grid), compared with ln(1/eta) = 23.0 (eta = 1e-10; the escape from a deviation ~ eta to O(1) at the rate
   lambda_u takes ~ ln(1/eta)/lambda_u; the residence also contains the approach, so only the order is compared).
  -> analysis/heteroclinic_n50/n50_web_analysis.json
"""
from pathlib import Path
import collections
import json
import sys

import numpy as np
from scipy.optimize import linear_sum_assignment

sys.path.insert(0, str(Path(__file__).resolve().parent))
from renoise_summary import residences, D  # noqa: E402
from noiseless_switches import ROOT  # noqa: E402

OUT = ROOT / "analysis/heteroclinic_n50"


def b1(summary, grid):
    rows = []
    pts = sorted({(r["eps"], r["eK"]) for r in summary if r["N"] == 50})
    for e, q in pts:
        runs = [r for r in summary if r["N"] == 50 and r["eps"] == e and r["eK"] == q]
        g = {}
        for typ in ((17, 17, 16), (17, 16, 16, 1)):
            gg = [x for x in grid if tuple(x["sizes"]) == typ and abs(x["eps"] - e) < 1e-9 and abs(x["q"] - q) < 1e-9]
            if gg:
                g[str(typ)] = dict(n_unstable=sorted({x["n_unstable_big"] for x in gg}),
                                   max_internal=[round(min(x["max_big"] for x in gg), 4),
                                                 round(max(x["max_big"] for x in gg), 4)])
        rows.append(dict(eps=e, eK=q, runs=len(runs), sustained=sum(r["sustained"] for r in runs),
                         most=collections.Counter(str(r["most"]) for r in runs if r["sustained"]).most_common(),
                         grid=g))
    return rows


def persistent(res):
    """Persistent cluster identities across consecutive residences; per switch the moves (element, donor, receiver)."""
    ids = {frozenset(c): i for i, c in enumerate(res[0][2])}
    nxt = len(ids)
    gone = {}
    switches = []
    for (t0, t1, A), (u0, u1, B) in zip(res[:-1], res[1:]):
        if A == B:
            continue
        newids = {}
        old, new = list(ids), list(B)
        ov = np.array([[len(a & c) for c in new] for a in old])
        ri, ci = linear_sum_assignment(-ov)  # largest total overlap, one to one
        for i, j in zip(ri, ci):
            if ov[i, j] > 0:
                newids[new[j]] = ids[old[i]]
        for c in new:
            if c not in newids:
                back = [i for i, m in gone.items() if len(m & c) >= .8 * max(len(m), len(c))]
                if back and back[0] not in newids.values():
                    newids[c] = back[0]
                    del gone[back[0]]
                else:
                    newids[c] = nxt
                    nxt += 1
        for a in old:
            if ids[a] not in newids.values() and len(a) >= 10:
                gone[ids[a]] = a  # a large cluster that lost coherence (all members free at the threshold)
        where0 = {el: ids[c] for c in ids for el in c}
        where1 = {el: newids[c] for c in newids for el in c}
        moves = [(el, where0.get(el, "free"), where1.get(el, "free")) for el in range(50)
                 if where0.get(el, "free") != where1.get(el, "free")]
        switches.append(dict(t=u0, t_prev_end=t1, moves=moves, sizes_before=sorted(map(len, A), reverse=True),
                             sizes_after=sorted(map(len, B), reverse=True),
                             members_before={ids[c]: sorted(c) for c in ids}))
        ids = newids
    return switches


def b2(index):
    lab = np.load(D / f"rn_{index:04d}.npz")["labels"]
    res = residences(lab)
    sw = persistent(res)
    nmov = [len(s["moves"]) for s in sw]
    visited = collections.defaultdict(set)
    for el in range(50):
        visited[el]
    for s in sw:
        for el, a, b in s["moves"]:
            visited[el] |= {a, b}
    moved = [el for el in range(50) if visited[el]]
    dissolve = [s["t"] for s in sw if len(s["moves"]) >= 10]
    small = [s for s in sw if len(s["moves"]) < 10]
    trans = collections.Counter((str(a), str(b)) for s in small for _, a, b in s["moves"])
    last = {}
    el_trans = collections.Counter()
    for s in small:  # cluster -> cluster of an element, skipping its time as a singleton
        for el, a, b in s["moves"]:
            if a != "free":
                last[el] = a
            if b != "free" and el in last:
                if last[el] != b:
                    el_trans[(last[el], b)] += 1
                del last[el]
    cyc = None
    big = sorted({x for k in el_trans for x in k})
    if len(big) == 3:
        a, b, c = big
        fwd = el_trans[(a, b)] + el_trans[(b, c)] + el_trans[(c, a)]
        bwd = el_trans[(b, a)] + el_trans[(c, b)] + el_trans[(a, c)]
        cyc = [fwd, bwd]
    relay, n_relay = 0, 0
    for s0, s1 in zip(small[:-1], small[1:]):
        rec0 = {b for _, a, b in s0["moves"] if b != "free"}
        don1 = {a for _, a, b in s1["moves"] if a != "free"}
        if rec0 and don1:
            n_relay += 1
            relay += bool(rec0 & don1)
    pairs = collections.Counter()
    for (a, b), c in trans.items():
        if a != "free" and b != "free":
            pairs[tuple(sorted((a, b)))] += 0
    asym = []
    for a, b in {tuple(sorted(k)) for k in trans if "free" not in k}:
        f, r = trans[(a, b)], trans[(b, a)]
        asym.append((a, b, f, r))
    dur = [r[1] - r[0] for r in res]
    return dict(index=index, residences=len(res), switches=len(sw), dissolutions=dissolve,
                element_transitions=[[str(a), str(b), c] for (a, b), c in el_trans.most_common()], cyclic=cyc,
                moves_per_switch=collections.Counter(nmov),
                elements_moved=len(moved), clusters_visited=collections.Counter(len(visited[el] - {"free"})
                                                                               for el in moved),
                transitions=[[a, b, c] for (a, b), c in trans.most_common()],
                pair_asymmetry=asym, relay=[relay, n_relay], residence_durations=dur,
                sizes=collections.Counter(str(s["sizes_before"]) for s in sw).most_common(3))


def firing_order(index):
    """With the full state (analysis/itinerancy_n50/trace_<index>.npz, every time unit, the same orbit as rn_<index>):
    before every switch, the upward x = 0 crossings of the mean x of each big cluster over the last 60 time units of
    the previous residence give the cyclic firing order; each element move (donor -> next cluster joined, skipping
    the singleton time) and each change of donor between consecutive switches is classified as +1 (the cluster that
    fires next after) or -1 (the one that fires just before) in that order."""
    tr = np.load(ROOT / f"analysis/itinerancy_n50/trace_{index:04d}.npz")["z"]
    lab = np.load(D / f"rn_{index:04d}.npz")["labels"]
    sw = [s for s in persistent(residences(lab)) if len(s["moves"]) < 10]
    n = tr.shape[1] // 2

    def order(s):
        t1 = int(s["t_prev_end"])
        cyc = {}
        for i, mem in s["members_before"].items():
            if len(mem) < 5:
                continue
            x = tr[max(t1 - 60, 1) - 1:t1, mem].mean(1)
            up = np.flatnonzero((x[:-1] < 0) & (x[1:] >= 0))
            if len(up):
                cyc[i] = up[-1] + x[up[-1]] / (x[up[-1]] - x[up[-1] + 1])
        if len(cyc) != 3:
            return None
        last = max(cyc.values())
        return [i for i, _ in sorted(cyc.items(), key=lambda kv: (kv[1] - last) % 1e9)]

    def rel(o, a, b):
        if o is None or a not in o or b not in o:
            return None
        d = (o.index(b) - o.index(a)) % 3
        return {1: +1, 2: -1}.get(d)

    moves, donors = collections.Counter(), collections.Counter()
    pending = {}
    prev = None
    for s in sw:
        o = order(s)
        for el, a, b in s["moves"]:
            if a != "free":
                pending[el] = (a, o)
            elif el in pending and b != "free":
                a0, o0 = pending.pop(el)
                moves[rel(o0, a0, b)] += 1
        don = [a for _, a, b in s["moves"] if a != "free"]
        if prev is not None and don and prev[0] != don[0]:
            donors[rel(o, prev[0], don[0])] += 1
        if don:
            prev = (don[0], o)
    return dict(index=index, element_move_to=dict(moves), next_donor=dict(donors))


def main():
    summary = json.loads((D / "renoise_summary.json").read_text())
    grid = json.loads((OUT / "n50_grid.json").read_text())
    out = dict(B1=b1(summary, grid), B2={}, B3=[])
    print("B1: eps, eK, runs, sustained, most frequent partitions of sustained runs, grid (n unstable, max internal)")
    for r in out["B1"]:
        print(f"  {r['eps']:.2f} {r['eK']:+.1f} runs {r['runs']:2d} sustained {r['sustained']:2d} {r['most'][:3]} "
              f"{ {k: (v['n_unstable'], v['max_internal']) for k, v in r['grid'].items()} }")
    sus = [r for r in summary if r["N"] == 50 and r["sustained"]]
    print("B2/B3 per sustained N = 50 run:")
    for r in sus:
        b = b2(r["index"])
        out["B2"][r["index"]] = b
        g = [x for x in grid if abs(x["eps"] - r["eps"]) < 1e-9 and abs(x["q"] - r["eK"]) < 1e-9
             and x["max_big"] > 0]
        lam = max((x["max_big"] for x in g), default=None)
        med = float(np.median(b["residence_durations"])) if b["residence_durations"] else None
        out["B3"].append(dict(index=r["index"], eps=r["eps"], eK=r["eK"], median_residence=med, lambda_u=lam,
                              median_x_lambda=med * lam if (med and lam) else None))
        top = b["transitions"][:4]
        print(f"  {r['index']} eps {r['eps']} eK {r['eK']}: switches {b['switches']}, moves/switch "
              f"{dict(b['moves_per_switch'])}, dissolutions {len(b['dissolutions'])}, relay {b['relay'][0]}/"
              f"{b['relay'][1]}, element cluster->cluster {b['element_transitions'][:6]}, cyclic (one way, other) "
              f"{b['cyclic']}; "
              f"residence median {med} x lambda_u {lam} = {out['B3'][-1]['median_x_lambda']}", flush=True)
    print("firing order (trace runs): +1 = the cluster firing next after, -1 = the one firing just before")
    out["firing_order"] = []
    for idx in (379, 386, 393, 394, 397):
        if (ROOT / f"analysis/itinerancy_n50/trace_{idx:04d}.npz").exists():
            f = firing_order(idx)
            out["firing_order"].append(f)
            print(f"  {idx}: element joins {f['element_move_to']}, next donor {f['next_donor']}", flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "n50_web_analysis.json").write_text(json.dumps(out, indent=1, default=lambda o: dict(o) if isinstance(
        o, collections.Counter) else str(o)))


if __name__ == "__main__":
    main()
