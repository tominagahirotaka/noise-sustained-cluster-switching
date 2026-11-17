#!/usr/bin/env python3
"""Contraction rate along the actual return direction of the slow approach-and-departure (exploratory, 2026-10-24).

Reduced system from run --index at record --rec (clusters cut at --cut, centroids; as large_n_reduced_floquet.py).
  - O: the periodic orbit of the reduced system with the pair (--pair, two cluster sizes) joined (or, without
    --pair, of the reduced system itself). Full Floquet spectrum: collective multipliers (non-neutral) and, for every
    cluster, both multipliers of its internal (transverse) 2x2 block; exponents log|mu| / T.
  - With --pair: the split system without noise, transient 5000, then --length; at the section returns (cluster 0,
    x = 0 upward): the pair distance p (max norm) and the distance c of the joined projection (population-weighted
    mean of the pair) to the section point of O. The log p series is smoothed (running median of 21 returns); every
    descent from a maximum to the next minimum (extrema with a hysteresis of one decade) and every ascent is fitted, between 1 decade from its ends, by a
    straight line in time: return rates and departure rates. Also c at each minimum of p.
  -> analysis/switching_check/return_rate_<index><--tag>.json

  python3 scripts/return_rate.py --index 90 --rec 1576 --pair 109 22
  python3 scripts/return_rate.py --index 88 --rec 300 --tag _O        (spectrum of the reduced partition only)
"""
from pathlib import Path
import argparse
import json
import sys

import numpy as np
from scipy.integrate import solve_ivp

sys.path.insert(0, str(Path(__file__).resolve().parent))
from eps_three_cluster import GAMMA, Partition  # noqa: E402
import chaos_check  # noqa: E402
from large_n_reduced_floquet import OPT, TOL, reduce, section_and_period  # noqa: E402
from renoise_summary import D  # noqa: E402


def spectrum(part, w, K, e):
    m, n, T = part.m, part.n, w[-1]

    def f(_t, v):
        z = v[:n]
        A = np.zeros((m, 2, 2))
        A[:, 0, 0] = (1 - z[:m] ** 2) / e - K
        A[:, 0, 1] = -1 / e
        A[:, 1, 0] = e
        A[:, 1, 1] = -e * GAMMA
        return np.r_[part.rhs(z, K, e), (part.jac(z, K, e) @ v[n:n + n * n].reshape(n, n)).ravel(),
                     (A @ v[n + n * n:].reshape(m, 2, 2)).ravel()]

    s = solve_ivp(f, (0, T), np.r_[w[:n], np.eye(n).ravel(), np.tile(np.eye(2), (m, 1, 1)).ravel()],
                  method="DOP853", rtol=1e-12, atol=1e-14, max_step=.04)
    v = s.y[:, -1]
    M, B = v[n:n + n * n].reshape(n, n), v[n + n * n:].reshape(m, 2, 2)
    mu = np.linalg.eigvals(M)
    mu = np.delete(mu, int(np.argmin(abs(mu - 1))))
    mu = mu[np.argsort(-abs(mu))]
    internal = [sorted(np.abs(np.linalg.eigvals(B[k])).tolist(), reverse=True) for k in range(m)]
    ex = lambda x: float(np.log(max(x, 1e-300)) / T)  # noqa: E731
    return dict(period=float(T), return_error=float(np.max(abs(v[:n] - w[:n]))),
                collective=[[float(abs(x)), float(np.angle(x)), ex(abs(x))] for x in mu],
                internal=[dict(size=int(part.sizes[k]), multipliers=internal[k], exponents=[ex(x) for x in internal[k]])
                          for k in range(m)])


def orbit(part, z, K, e):
    z = solve_ivp(lambda _t, u: part.rhs(u, K, e), (0, 2000), z, **OPT).y[:, -1]
    zz, T0 = section_and_period(part, z, K, e)
    return part.shoot(np.r_[zz, T0], K, e, TOL)


def fits(t, lg, up):
    """Straight-line rates (per time unit, natural log) of the descents (up = False) or ascents of lg."""
    k = 21
    sm = np.array([np.median(lg[max(0, i - k // 2):i + k // 2 + 1]) for i in range(len(lg))])
    # alternating maxima and minima with a hysteresis of one decade
    ext, up_now, cand = [], None, 0
    for i in range(1, len(sm)):
        if up_now is None:
            if abs(sm[i] - sm[0]) > 1:
                up_now = sm[i] > sm[0]
                ext.append(int(np.argmin(sm[:i + 1]) if up_now else np.argmax(sm[:i + 1])))
                cand = i
            continue
        if (sm[i] > sm[cand]) == up_now:
            cand = i
        elif abs(sm[i] - sm[cand]) > 1:
            ext.append(cand)
            up_now, cand = not up_now, i
    rates = []
    for a, b in zip(ext[:-1], ext[1:]):
        if (sm[b] > sm[a]) != up:
            continue
        lo, hi = min(sm[a], sm[b]) + 1, max(sm[a], sm[b]) - 1
        sel = np.arange(a, b + 1)
        sel = sel[(lg[sel] > lo) & (lg[sel] < hi)]
        if sel.size > 10:
            rates.append(float(np.polyfit(t[sel], lg[sel] * np.log(10), 1)[0]))
    return rates, ext, sm


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", type=int, required=True)
    ap.add_argument("--rec", type=int, required=True)
    ap.add_argument("--cut", type=float, default=1e-3)
    ap.add_argument("--pair", type=int, nargs=2)
    ap.add_argument("--length", type=float, default=40000.)
    ap.add_argument("--tag", default="")
    a = ap.parse_args()
    s = json.loads((D / f"rn_{a.index:04d}_states.json").read_text())
    e, K = s["eps"], s["K"]
    zs = np.load(D / f"rn_{a.index:04d}_states_z.npy")
    sizes, v = reduce(zs[a.rec], zs.shape[1] // 2, a.cut)
    part = Partition(sizes.tolist())
    m = part.m
    out = dict(index=a.index, rec=a.rec, eps=e, K=K, sizes=sizes.tolist(), pair=a.pair)
    if a.pair is None:
        w = orbit(part, v, K, e)
        out["O"] = None if w is None else dict(sizes=sizes.tolist(), **spectrum(part, w, K, e))
    else:
        i, j = (int(np.nonzero(sizes == p)[0][0]) for p in a.pair)
        keep = [k for k in range(m) if k != j]
        pm = sizes.copy()
        pm[i] += sizes[j]
        pm = pm[keep]
        mpart = Partition(pm.tolist())
        ii = keep.index(i)
        vs = solve_ivp(lambda _t, u: part.rhs(u, K, e), (0, 5000), v, **OPT).y[:, -1]

        def join(X):
            x, y = X[:m].copy(), X[m:].copy()
            x[i] = (sizes[i] * x[i] + sizes[j] * x[j]) / (sizes[i] + sizes[j])
            y[i] = (sizes[i] * y[i] + sizes[j] * y[j]) / (sizes[i] + sizes[j])
            return np.r_[x[keep], y[keep]]

        w = orbit(mpart, join(vs), K, e)
        if w is not None:
            out["O"] = dict(sizes=pm.tolist(), joined=int(pm[ii]), **spectrum(mpart, w, K, e))
            o = w[:mpart.n]
        sol = solve_ivp(lambda _t, u: part.rhs(u, K, e), (0, a.length), vs, events=chaos_check.sec, **OPT)
        X, t = sol.y_events[0], sol.t_events[0]
        p = np.maximum(abs(X[:, i] - X[:, j]), abs(X[:, m + i] - X[:, m + j]))
        lg = np.log10(p)
        c = np.array([np.max(abs(join(x) - o)) for x in X]) if w is not None else np.full(len(X), np.nan)
        down, ext, sm = fits(t, lg, up=False)
        up, _, _ = fits(t, lg, up=True)
        mins = [k for k, k2 in zip(ext[:-1], ext[1:]) if sm[k] < sm[k2]]
        out["split"] = dict(return_rates=down, departure_rates=up,
                            minima=[dict(t=float(t[k]), log10_pair=float(lg[k]), log10_dist_to_O=float(np.log10(c[k])))
                                    for k in mins],
                            log10_dist_to_O_range=[float(np.log10(c.min())), float(np.log10(np.median(c))),
                                                   float(np.log10(c.max()))])
    (D / f"return_rate_{a.index:04d}{a.tag}.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
