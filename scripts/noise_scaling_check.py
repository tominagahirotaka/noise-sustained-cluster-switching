#!/usr/bin/env python3
"""Noise dependence of the switches of one persistent-noise run (exploratory, 2026-10-24).

Input: analysis/switching_check/rn_<index>[_n<eta>]_t<tmax>_states_z.npy (switching_renoise.py --states) for each
noise sd eta. Clusters cut at --th (single linkage, max norm; renoise_threshold_check). Residences of >= 200 time
units (renoise_summary.residences) and switches between them. For noise-sustained switching on a heteroclinic
network the residences grow like ln(1/eta) / rate; for a deterministic oscillation they do not depend on eta.
Also, for each residence, the smallest spread (log10, max norm) of the largest cluster: it should sit near eta when
the cluster is held only by the noise.

  python3 scripts/noise_scaling_check.py --index 88 --tmax 80000  -> analysis/switching_check/noise_scaling_<index>.json
"""
from pathlib import Path
import argparse
import json
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from renoise_summary import D, residences  # noqa: E402
from renoise_threshold_check import cut, merges  # noqa: E402

ETAS = (1e-6, 1e-8, 1e-10, 1e-12, 1e-14)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", type=int, required=True)
    ap.add_argument("--tmax", type=int, default=80000)
    ap.add_argument("--th", type=float, default=1e-3)
    a = ap.parse_args()
    out = []
    for eta in ETAS:
        tag = f"rn_{a.index:04d}" + ("" if eta == 1e-10 else f"_n{eta:.0e}") + f"_t{a.tmax}_states"
        f = D / f"{tag}_z.npy"
        if not f.exists():
            continue
        zs = np.load(f)
        n = zs.shape[1] // 2
        labs, spr = [], []
        for z in zs:
            _, _, Z = merges(z)
            L = cut(Z, n, a.th)
            labs.append(L)
            u, inv = np.unique(L, return_inverse=True)
            c = np.argmax(np.bincount(inv))
            sel = inv == c
            spr.append(max(np.ptp(z[:n][sel]), np.ptp(z[n:][sel])))
        res = residences(np.asarray(labs))
        lg = np.log10(np.maximum(spr, 1e-300))
        R = [dict(start=int(r0), end=int(r1), L=int(r1 - r0), min_log10_spread=float(lg[r0 // 10 - 1:r1 // 10].min()))
             for r0, r1, _ in res]
        sw = sum(x[2] != y[2] for x, y in zip(res[:-1], res[1:]))
        Ls = np.array([r["L"] for r in R[1:-1]]) if len(R) > 2 else np.array([])  # first and last are cut by the run
        js = json.loads((D / f"{tag}.json").read_text())
        row = dict(eta=eta, switches=int(sw), residences=R, L_median=float(np.median(Ls)) if Ls.size else None,
                   L_mean=float(Ls.mean()) if Ls.size else None, L_max=int(Ls.max()) if Ls.size else None,
                   spread_min_median=float(np.median([r["min_log10_spread"] for r in R])) if R else None,
                   lyap_late=js["lyap_late_half"])
        out.append(row)
        print(f"eta {eta:.0e} switches {sw:3d} residences {len(R):3d} L median {row['L_median']} mean {row['L_mean']}"
              f" max {row['L_max']} min log10 spread (median) {row['spread_min_median']} lyap_late {row['lyap_late']:.5f}",
              flush=True)
    (D / f"noise_scaling_{a.index:04d}.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
