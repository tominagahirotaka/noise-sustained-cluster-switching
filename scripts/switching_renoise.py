#!/usr/bin/env python3
"""Are the positive exponents of scripts/switching_check.py real, or held by exact synchrony? (exploratory, 2026-10-23)

In the long runs, elements of a contracting cluster can become identical to the last bit; identical elements stay
identical, so a cluster that later becomes transversely unstable cannot split. The second copy used for the exponent
is not identical and picks up the instability. Here the final state of sw_<index> gets Gaussian noise of sd 1e-6 and,
every 10 time units, noise of sd 1e-10 on all 2N coordinates (the same noise on both copies), so that exact synchrony
cannot persist. RK4, h = 0.005, to --tmax. Recorded every 10 time units: cluster labels at threshold 1e-4, largest
within-cluster spread; largest exponent (two copies, 1e-8, renormalized every 5 time units); firing times of element 0.
  -> analysis/switching_check/rn_<index>.npz and rn_<index>.json

  python3 scripts/switching_renoise.py --index 397 --tmax 20000      (remote host: numpy, scipy)
  --noise s : sd of the repeated noise (default 1e-10); other values write rn_<index>_n<s>.npz/.json
  --tmax other than 20000 adds _t<tmax> to the names
  --states : also saves the full state (before the noise) every 10 time units, rn_<...>_states_z.npy, and adds _states
             to the names (same trajectory as without the option)
"""
from pathlib import Path
import argparse
import json

import numpy as np
from scipy.sparse.csgraph import connected_components

ROOT = Path(__file__).resolve().parents[1]
D = ROOT / "analysis/switching_check"
B, G = .7, .5
D0 = 1e-8


def labels(z, n, th):
    d = np.maximum(abs(z[:n, None] - z[None, :n]), abs(z[n:, None] - z[None, n:]))
    return connected_components(d < th, directed=False)


def run(index, tmax, noise=1e-10, states=False):
    r = json.loads((D / "selected.json").read_text())[index]
    e, K = r["eps"], r["K"]
    z0 = np.load(D / f"sw_{index:04d}.npz")["final"]
    n = z0.size // 2
    rng = np.random.default_rng(10_000 + index)
    zA = z0 + rng.normal(0, 1e-6, z0.size)
    g = rng.standard_normal(z0.size)
    zB = zA + D0 * g / np.linalg.norm(g)

    def f(v):
        x, y = v[:n], v[n:]
        return np.r_[(x - x ** 3 / 3 - y) / e + K * (x.mean() - x), e * (x + B - G * y)]

    def rk(z, h):
        k1 = f(z)
        k2 = f(z + h / 2 * k1)
        k3 = f(z + h / 2 * k2)
        k4 = f(z + h * k3)
        return z + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)

    h = .005
    nunits = int(tmax)
    lab_rec = np.zeros((nunits // 10, n), np.int16)
    spr = np.zeros(nunits // 10)
    rec = np.zeros((nunits // 10, z0.size)) if states else None
    fire0, lsum, lyap = [], 0., []
    xprev = zA[0]
    t = 0.
    for u in range(nunits):
        for _ in range(200):
            zA, zB = rk(zA, h), rk(zB, h)
            t += h
            if xprev < 0 <= zA[0]:
                fire0.append(t - h * zA[0] / (zA[0] - xprev))
            xprev = zA[0]
        if (u + 1) % 5 == 0:
            dd = zB - zA
            nd = np.linalg.norm(dd)
            lsum += np.log(nd / D0)
            zB = zA + D0 * dd / nd
        if (u + 1) % 10 == 0:
            k = (u + 1) // 10 - 1
            nc, lab_rec[k] = labels(zA, n, 1e-4)
            if states:
                rec[k] = zA
            spr[k] = max(max(np.ptp(zA[:n][lab_rec[k] == c]), np.ptp(zA[n:][lab_rec[k] == c])) for c in range(nc))
            w = rng.normal(0, noise, z0.size)
            zA, zB = zA + w, zB + w
        if (u + 1) % 1000 == 0:
            lyap.append([u + 1, lsum / (u + 1)])
    tag = f"rn_{index:04d}" + ("" if noise == 1e-10 else f"_n{noise:.0e}") + ("" if tmax == 20000 else f"_t{int(tmax)}")
    if states:
        tag += "_states"
        np.save(D / f"{tag}_z.npy", rec)
    np.savez_compressed(D / f"{tag}.npz", labels=lab_rec, spread=spr, fire0=np.asarray(fire0),
                        lyap=np.asarray(lyap), final=zA)
    ly = np.asarray(lyap)
    late = (ly[-1, 0] * ly[-1, 1] - ly[len(ly) // 2 - 1, 0] * ly[len(ly) // 2 - 1, 1]) / (ly[-1, 0] - ly[len(ly) // 2 - 1, 0])
    fin = sorted(np.bincount(lab_rec[-1]).tolist(), reverse=True)
    summ = dict(index=index, noise=noise, N=n, eps=e, K=K, tmax=tmax, lyap_full=lsum / nunits, lyap_late_half=late, final_sizes=fin,
                sw_final=sorted(np.bincount(labels(z0, n, 1e-4)[1]).tolist(), reverse=True))
    (D / f"{tag}.json").write_text(json.dumps(summ))
    print(index, noise, n, e, round(e * K, 2), "lyap_late", round(late, 5), "final", fin[:8], "sw_final", summ["sw_final"][:8],
          flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", type=int, required=True)
    ap.add_argument("--tmax", type=float, default=20000.)
    ap.add_argument("--noise", type=float, default=1e-10)
    ap.add_argument("--states", action="store_true")
    a = ap.parse_args()
    run(a.index, a.tmax, a.noise, a.states)


if __name__ == "__main__":
    main()
