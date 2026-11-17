#!/usr/bin/env python3
"""Slow approach and departure of a sub-cluster in the large-N persistent-noise runs: heteroclinic or a slow
oscillation of the relative phase? Reduced-system test (exploratory, 2026-10-24).

For run --index at record --rec of analysis/switching_check/rn_<index>_states_z.npy (switching_renoise.py --states):
clusters cut at --cut (default 1e-3; single linkage, max norm), centroids -> reduced m-cluster system (Partition) at the run's
eps, K. The pair (--pair, two cluster sizes) is the sub-cluster and the cluster it approaches.
  1. split system: transient 5000, then 20000 time units without noise; section returns (cluster 0, x = 0 upward);
     the pair distance (max norm) at the returns; smallest lag <= 3000 returns with return distance < 1e-6
     (periodic orbit locked to the fast oscillation); largest exponent from two trajectories and the internal
     (transverse) exponents of all clusters (chaos_check.two_traj);
  2. split periodic orbit near the attractor: shooting from the mean of the section returns; collective Floquet
     multipliers (largest four, modulus and angle) and internal exponents (Partition.audit);
  3. merged system (the pair joined at the population-weighted centroid): periodic orbit after a transient of 2000;
     internal (transverse) Floquet exponent of the joined cluster, log|mu| / T, and the largest collective multiplier.
  -> analysis/switching_check/reduced_floquet_<index><--tag>.json

  python3 scripts/large_n_reduced_floquet.py --index 90 --rec 1576 --pair 109 22
"""
from pathlib import Path
import argparse
import json
import sys

import numpy as np
from scipy.integrate import solve_ivp

sys.path.insert(0, str(Path(__file__).resolve().parent))
from eps_three_cluster import Partition  # noqa: E402
import chaos_check  # noqa: E402
from renoise_summary import D  # noqa: E402
from renoise_threshold_check import cut, merges  # noqa: E402

OPT = dict(method="DOP853", rtol=1e-11, atol=1e-13, max_step=.05)
TOL = 2e-12


def reduce(z, n, th=1e-3):
    _, _, Z = merges(z)
    _, inv = np.unique(cut(Z, n, th), return_inverse=True)
    sz = np.bincount(inv)
    o = np.argsort(-sz, kind="stable")
    return sz[o], np.r_[[z[:n][inv == c].mean() for c in o], [z[n:][inv == c].mean() for c in o]]


def periodic(part, z, K, e, T0):
    w = part.shoot(np.r_[z, T0], K, e, TOL)
    if w is None:
        return None
    a = part.audit(w, K, e, TOL)
    m, n, T = part.m, part.n, w[-1]
    _, A = part.flow(w, K, e, TOL)
    mu = np.linalg.eigvals(A[:n, :n] + np.eye(n))
    mu = mu[np.argsort(-abs(mu))]
    mu = mu[abs(mu - 1) > 1e-6][:4]
    return w, a, [[float(abs(x)), float(np.angle(x))] for x in mu]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", type=int, required=True)
    ap.add_argument("--rec", type=int, required=True)
    ap.add_argument("--pair", type=int, nargs=2, required=True)
    ap.add_argument("--length", type=float, default=20000.)
    ap.add_argument("--cut", type=float, default=1e-3)
    ap.add_argument("--tag", default="")
    a = ap.parse_args()
    s = json.loads((D / f"rn_{a.index:04d}_states.json").read_text())
    e, K = s["eps"], s["K"]
    zs = np.load(D / f"rn_{a.index:04d}_states_z.npy")
    n = zs.shape[1] // 2
    sizes, v = reduce(zs[a.rec], n, a.cut)
    part = Partition(sizes.tolist())
    m = part.m
    i, j = (int(np.nonzero(sizes == p)[0][0]) for p in a.pair)
    print(a.index, "sizes", sizes.tolist(), "pair", a.pair, "eps", e, "K", K, flush=True)
    # 1. split system
    v = solve_ivp(lambda _t, u: part.rhs(u, K, e), (0, 5000), v, **OPT).y[:, -1]
    sol = solve_ivp(lambda _t, u: part.rhs(u, K, e), (0, a.length), v, events=chaos_check.sec, **OPT)
    X, tr = sol.y_events[0], sol.t_events[0]
    dist = np.maximum(abs(X[:, i] - X[:, j]), abs(X[:, m + i] - X[:, m + j]))
    lag = None
    for L in range(1, min(3001, len(X) - 51)):
        if max(np.max(abs(X[k] - X[k - L])) for k in range(len(X) - 50, len(X))) < 1e-6:
            lag = L
            break
    _, lam, intl, _, _ = chaos_check.two_traj(part, v, K, e, a.length, OPT, internal=True, returns=True)
    lg = np.log10(dist)
    # slow period in returns: mean spacing of the minima of the pair distance below its median
    lo = np.nonzero((lg[1:-1] < lg[:-2]) & (lg[1:-1] <= lg[2:]) & (lg[1:-1] < np.median(lg)))[0] + 1
    mins = [lo[0]] if lo.size else []
    for k in lo[1:]:
        if k - mins[-1] > 20:
            mins.append(k)
    slow = float(np.mean(np.diff(tr[mins]))) if len(mins) > 2 else None
    split = dict(lyap=float(lam), internal=intl.tolist(), lag_returns=lag, fast_period=float(np.diff(tr).mean()),
                 slow_period=slow, pair_log10_dist=[float(lg.min()), float(np.median(lg)), float(lg.max())])
    # 2. split periodic orbit near the attractor
    po = periodic(part, X.mean(axis=0), K, e, float(np.diff(tr).mean()))
    if po is not None:
        w, au, mu = po
        split["orbit"] = dict(period=float(w[-1]), multipliers=mu, internal_max=au["internal_max"],
                              valid=au["valid"], pair_dist_at_section=float(max(abs(w[i] - w[j]), abs(w[m + i] - w[m + j]))))
    # 3. merged system
    pm = np.r_[sizes[[k for k in range(m) if k != j]]]
    pm[i if i < j else i - 1] += sizes[j]
    keep = [k for k in range(m) if k != j]
    xm, ym = X[-1, :m].copy(), X[-1, m:].copy()
    xm[i] = (sizes[i] * xm[i] + sizes[j] * xm[j]) / (sizes[i] + sizes[j])
    ym[i] = (sizes[i] * ym[i] + sizes[j] * ym[j]) / (sizes[i] + sizes[j])
    mpart = Partition(pm.tolist())
    vm = solve_ivp(lambda _t, u: mpart.rhs(u, K, e), (0, 2000), np.r_[xm[keep], ym[keep]], **OPT).y[:, -1]
    ii = i if i < j else i - 1
    merged = dict(sizes=pm.tolist())
    zz, T0 = section_and_period(mpart, vm, K, e)
    pr = periodic(mpart, zz, K, e, T0)
    if pr is not None:
        w, au, mu = pr
        T = w[-1]
        merged.update(period=float(T), multipliers=mu, valid=au["valid"],
                      joined_internal_multiplier=au["internal_max"][ii],
                      joined_internal_exponent=float(np.log(au["internal_max"][ii]) / T),
                      internal_max_others=max(x for k, x in enumerate(au["internal_max"]) if k != ii and pm[k] > 1))
    out = dict(index=a.index, rec=a.rec, eps=e, K=K, sizes=sizes.tolist(), pair=a.pair, split=split, merged=merged)
    (D / f"reduced_floquet_{a.index:04d}{a.tag}.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))


def section_and_period(part, z, K, e):
    z = solve_ivp(lambda _t, u: part.rhs(u, K, e), (0, 1), z, **OPT).y[:, -1]
    s = solve_ivp(lambda _t, u: part.rhs(u, K, e), (0, 300), z, events=chaos_check.sec, **OPT)
    z1 = s.y_events[0][0]
    s2 = solve_ivp(lambda _t, u: part.rhs(u, K, e), (0, 300), z1, events=chaos_check.sec, **OPT)
    return z1, float(s2.t_events[0][s2.t_events[0] > 1][0])


if __name__ == "__main__":
    main()
