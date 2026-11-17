#!/usr/bin/env python3
"""Itinerancy of the N = 50 runs of scripts/switching_renoise.py (379, 386, 393, 394, 397; eps = 0.25) (exploratory,
2026-10-23).

  trace   : the run of switching_renoise.py (same start, same noise realization; --noise, default 1e-10) to --tmax,
            saving the full state every time unit (float64).
            -> analysis/itinerancy_n50/trace_<index>[_n<noise>].npz
  reduced : the reduced m-cluster system (eps_three_cluster.Partition) of the partition --sizes, started from the
            cluster centroids of the trace at the first time unit after --t0 whose partition (threshold 1e-4) has
            these sizes, or, with --seed, from random centroids (x in [-2, 2], y in [-0.6, 1.4]); transient 5000,
            then 15000 time units with the two-trajectory largest exponent, the transverse (internal 2x2 block)
            exponent of every cluster, and upward x_1 = 0 returns (smallest lag <= 60 with return distance < 1e-6
            over the last 50 returns). Same tolerances as chaos_check.scan (DOP853, rtol 1e-10).
            -> analysis/itinerancy_n50/reduced_<sizes>_<source>.json

  python3 scripts/itinerancy_n50.py trace --index 394 --tmax 20000              (remote host: numpy, scipy)
  python3 scripts/itinerancy_n50.py reduced --index 394 --sizes 17,16,16,1 --t0 1000
  python3 scripts/itinerancy_n50.py reduced --sizes 17,16,16,1 --eps 0.25 --q -1.0 --seed 1
"""
from pathlib import Path
import argparse
import json
import sys

import numpy as np
from scipy.integrate import solve_ivp
from scipy.sparse.csgraph import connected_components

sys.path.insert(0, str(Path(__file__).resolve().parent))
from eps_three_cluster import Partition, ROOT  # noqa: E402
import chaos_check  # noqa: E402

SW = ROOT / "analysis/switching_check"
OUT = ROOT / "analysis/itinerancy_n50"
B, G = .7, .5
D0 = 1e-8


def labels(z, n, th):
    d = np.maximum(abs(z[:n, None] - z[None, :n]), abs(z[n:, None] - z[None, n:]))
    return connected_components(d < th, directed=False)


def ntag(noise):
    return "" if noise == 1e-10 else f"_n{noise:.0e}"


def trace(a):
    """Same arithmetic as switching_renoise.run, without the exponent copy (its random draws are kept)."""
    r = json.loads((SW / "selected.json").read_text())[a.index]
    e, K = r["eps"], r["K"]
    z0 = np.load(SW / f"sw_{a.index:04d}.npz")["final"]
    n = z0.size // 2
    rng = np.random.default_rng(10_000 + a.index)
    z = z0 + rng.normal(0, 1e-6, z0.size)
    rng.standard_normal(z0.size)

    def f(v):
        x, y = v[:n], v[n:]
        return np.r_[(x - x ** 3 / 3 - y) / e + K * (x.mean() - x), e * (x + B - G * y)]

    h = .005
    nunits = int(a.tmax)
    rec = np.zeros((nunits, 2 * n))
    for u in range(nunits):
        for _ in range(200):
            k1 = f(z)
            k2 = f(z + h / 2 * k1)
            k3 = f(z + h / 2 * k2)
            k4 = f(z + h * k3)
            z = z + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
        rec[u] = z
        if (u + 1) % 10 == 0:
            z = z + rng.normal(0, a.noise, z0.size)
    OUT.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(OUT / f"trace_{a.index:04d}{ntag(a.noise)}.npz", z=rec, eps=e, K=K)
    print(a.index, "trace done", flush=True)


def reduced(a):
    sizes = [int(s) for s in a.sizes.split(",")]
    if a.seed is None:
        tr = np.load(OUT / f"trace_{a.index:04d}{ntag(a.noise)}.npz")
        e, K = float(tr["eps"]), float(tr["K"])
        Z = tr["z"]
        n = Z.shape[1] // 2
        for t in range(int(a.t0), len(Z)):
            m, lab = labels(Z[t], n, 1e-4)
            if sorted(np.bincount(lab).tolist(), reverse=True) == sorted(sizes, reverse=True):
                break
        else:
            raise SystemExit("partition not found")
        order = sorted(range(m), key=lambda c: -np.sum(lab == c))
        sizes = [int(np.sum(lab == c)) for c in order]
        v = np.r_[[Z[t, :n][lab == c].mean() for c in order], [Z[t, n:][lab == c].mean() for c in order]]
        src = f"{a.index}_t{t}"
    else:
        e, K = a.eps, a.q / a.eps
        rng = np.random.default_rng(a.seed)
        v = np.r_[rng.uniform(-2, 2, len(sizes)), rng.uniform(-.6, 1.4, len(sizes))]
        src = f"q{a.q:+.3f}_s{a.seed}"
    part = Partition(sizes)
    opt = dict(method="DOP853", rtol=1e-10, atol=1e-12, max_step=.05)
    v = solve_ivp(lambda _t, u: part.rhs(u, K, e), (0, 5000), v, **opt).y[:, -1]
    _, lam, intl, tr_, X = chaos_check.two_traj(part, v, K, e, 15000., opt, internal=True, returns=True)
    lag = None
    for L in range(1, 61):
        if len(X) > L + 50 and max(np.max(abs(X[k] - X[k - L])) for k in range(len(X) - 50, len(X))) < 1e-6:
            lag = L
            break
    ret = np.diff(tr_)
    big = part.sizes > 1
    res = dict(source=src, eps=e, K=K, q=e * K, sizes=sizes, lyap_max=float(lam), internal=intl.tolist(),
               internal_max_big=float(intl[big].max()), lag=lag, return_mean=float(ret.mean()),
               return_sd=float(ret.std()), return_times=ret[-300:].tolist())
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"reduced_{'-'.join(map(str, sizes))}_{src}.json").write_text(json.dumps(res))
    print(src, sizes, "lyap", round(lam, 5), "internal", np.round(intl, 4).tolist(), "lag", lag,
          "T", round(ret.mean(), 3), round(ret.std(), 4), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["trace", "reduced"])
    ap.add_argument("--index", type=int, default=394)
    ap.add_argument("--tmax", type=float, default=20000.)
    ap.add_argument("--noise", type=float, default=1e-10)
    ap.add_argument("--sizes", default="17,16,16,1")
    ap.add_argument("--t0", type=float, default=1000.)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--eps", type=float, default=.25)
    ap.add_argument("--q", type=float, default=-1.)
    a = ap.parse_args()
    {"trace": trace, "reduced": reduced}[a.mode](a)


if __name__ == "__main__":
    main()
