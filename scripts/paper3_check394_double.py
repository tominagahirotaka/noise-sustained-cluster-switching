#!/usr/bin/env python3
"""Paper 3, Sec. II C: plain double precision against the tree representation for run 394 (2026-10-31).

The comparison of RESULTS_NOTES 2026-10-23 (noiseless_n50.py --check 4000) was printed to the console only. This
script redoes it with saved output. Start: analysis/itinerancy_n50/trace_0394.npz at t0 = 5000 (eta = 1e-10 up to
then), no noise afterwards, RK4 h = 0.005, both integrations from the same state.
  plain : noiseless_n50.plain (the --check integrator, unchanged), state every time unit.
  tree  : the run of `noiseless_n50.py --index 394 --t0 5000 --tmax T --check T --tag _check<T>`
          -> analysis/noiseless_n50/nl_0394_t5000_check<T>.npz (groups at 1e-3 every time unit).
Residences: the same configuration for >= 200 time units (instantaneous configurations flicker with the phase).
Recorded every time unit, for the plain run: cluster sizes at 1e-3 (single linkage, max norm), and the sizes of the
classes of bitwise-identical elements (identical x and y). For both: the configuration (clusters of >= 10 elements at
1e-3, as element sets) and its changes (switches).
  -> analysis/paper3_figs/check394_double.json

  python3 scripts/paper3_check394_double.py --tmax 8000
"""
from pathlib import Path
import argparse
import hashlib
import json
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from noiseless_n50 import plain, groups, IT, OUT  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def config(g):
    labs, cc = np.unique(g, return_counts=True)
    return frozenset(frozenset(np.flatnonzero(g == l).tolist()) for l, c in zip(labs, cc) if c >= 10)


def switches(cfgs):
    out = []
    for t in range(1, len(cfgs)):
        if cfgs[t] != cfgs[t - 1]:
            out.append(t + 1)
    return out


def residences(cfgs, minlen=200):
    """Maximal runs of one configuration lasting >= minlen time units: [start, end, sorted sizes]."""
    out, s0 = [], 0
    for t in range(1, len(cfgs) + 1):
        if t == len(cfgs) or cfgs[t] != cfgs[s0]:
            if t - s0 >= minlen:
                out.append([s0 + 1, t, sorted((len(c) for c in cfgs[s0]), reverse=True)])
            s0 = t
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tmax", type=int, default=8000)
    a = ap.parse_args()
    trp = IT / "trace_0394.npz"
    tr = np.load(trp)
    e, K = float(tr["eps"]), float(tr["K"])
    z0 = tr["z"][5000].astype(float)
    n = z0.size // 2
    rec = plain(z0.copy(), e, K, a.tmax)
    tree_p = OUT / f"nl_0394_t5000_check{a.tmax}.npz"
    tg = np.load(tree_p)["group"]
    rows, pc, tc, all_ident = [], [], [], None
    for t in range(a.tmax):
        P = np.c_[rec[t][:n], rec[t][n:]]
        g = groups(P, 1e-3)
        _, cnt = np.unique(P, axis=0, return_counts=True)
        if all_ident is None and len(cnt) == 3:
            all_ident = t + 1
        pc.append(config(g))
        tc.append(config(tg[t]))
        if (t + 1) % 100 == 0:
            rows.append(dict(t=t + 1, plain_sizes=sorted(np.bincount(g).tolist(), reverse=True),
                             plain_identical=sorted(cnt[cnt >= 2].tolist(), reverse=True),
                             tree_sizes=sorted(np.bincount(tg[t]).tolist(), reverse=True)))
    first_ident = next((r["t"] for r in rows if r["plain_identical"] and r["plain_identical"][0] >= 10), None)
    same = [r["t"] for r in rows if tc[r["t"] - 1] == pc[r["t"] - 1]]
    sw_p, sw_t = switches(pc), switches(tc)
    out = dict(
        script="scripts/paper3_check394_double.py", script_sha256=sha(__file__),
        noiseless_n50_sha256=sha(ROOT / "scripts/noiseless_n50.py"),
        inputs={str(trp.relative_to(ROOT)): sha(trp), str(tree_p.relative_to(ROOT)): sha(tree_p)},
        command=" ".join(sys.argv), eps=e, K=K, t0=5000, tmax=a.tmax,
        rules="clusters: single linkage, max norm over (x, y), 1e-3; configuration: clusters >= 10 as element sets; "
              "identical: bitwise equal (x, y); time = units after t0",
        first_time_identical_class_ge10=first_ident,
        first_time_all_elements_in_3_identical_classes=all_ident,
        residences_plain=residences(pc), residences_tree=residences(tc),
        switches_plain_instantaneous=sw_p, switches_tree_instantaneous=sw_t,
        last_time_same_configuration=max(same) if same else None,
        first_time_configuration_differs=next((r["t"] for r in rows if tc[r["t"] - 1] != pc[r["t"] - 1]), None),
        every100=rows)
    p = ROOT / "analysis/paper3_figs/check394_double.json"
    p.write_text(json.dumps(out, indent=1))
    print("first identical class >= 10 at", first_ident)
    print("all elements in 3 identical classes from", all_ident)
    print("plain residences", out["residences_plain"])
    print("tree residences", out["residences_tree"])
    print("first differing config (100 grid)", out["first_time_configuration_differs"])


if __name__ == "__main__":
    main()
