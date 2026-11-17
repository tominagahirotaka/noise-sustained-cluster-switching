#!/usr/bin/env python3
"""Is the non-periodic seven-cluster state (many_cluster_check index 95) chaotic, and where does the chaos come from?
(exploratory, 2026-10-22)

Run 95 (N = 100, eps = 0.25, eps K = -1.0, seed 5042; partition 28:20:17:12:11:7:5) has no periodic orbit; with
scripts/no_recurrence_check.py (rtol 1e-10) the reduced seven-cluster system has largest Lyapunov exponent
+0.0155 and every cluster is transversely stable. Modes, all on the reduced m-cluster system (eps_three_cluster.Partition)
started from the cluster centroids of the saved final state of --index:

  verify : (a) the variational Lyapunov spectrum (QR every 5 time units) with rtol --rtol, atol rtol/100, max step
           --max-step, over --length after a transient of 2000; (b) the largest exponent from two trajectories
           (separation 1e-8, renormalized every 5 time units), same tolerances.
           -> analysis/chaos_check/verify_<index>_r<rtol>.json
  long   : integration to --length (no variational equations), upward x_1 = 0 returns; return-time statistics in
           windows of 10000 (an ending chaotic transient would show as a window with constant return time).
           -> analysis/chaos_check/long_<index>.json
  scan   : the same partition at eps K = --q (eps fixed): transient 5000 from the centroids, then 15000 time units with
           the two-trajectory largest exponent, the largest transverse exponent of every cluster of size >= 2
           (internal 2x2 blocks), and upward x_1 = 0 returns; the smallest lag <= 60 with return distance < 1e-6
           over the last 50 returns (periodic with that lag), and the last 300 return times and section states.
           -> analysis/chaos_check/scan_<index>_q<q>.json (q with 4 decimals since 2026-10-24; older files 3)

  python3 scripts/chaos_check.py verify --index 95 --rtol 1e-12 --max-step 0.02      (remote host: scipy)
  python3 scripts/chaos_check.py long --index 95 --length 200000
  python3 scripts/chaos_check.py scan --index 95 --q -0.95
  python3 scripts/chaos_check.py attractor --index 95 --q -1.0 --length 200000
  attractor: section states over --length (after a transient of 5000) and 400 time units of trajectory
           -> analysis/chaos_check/attractor_<index>_q<q>.json
"""
from pathlib import Path
import argparse
import json
import sys

import numpy as np
from scipy.integrate import solve_ivp
from scipy.sparse.csgraph import connected_components

sys.path.insert(0, str(Path(__file__).resolve().parent))
from eps_three_cluster import Partition, ROOT, GAMMA  # noqa: E402

OUT = ROOT / "analysis/chaos_check"
SRC = ROOT / "analysis/many_cluster_check/selected.json"
D0, DT = 1e-8, 5.


def load(index):
    r = json.loads(SRC.read_text())[index]
    z = np.asarray(r["final_state"], float)
    N = z.size // 2
    d = np.maximum(abs(z[:N, None] - z[None, :N]), abs(z[N:, None] - z[None, N:]))
    m, lab = connected_components(d < 1e-4, directed=False)
    v = np.r_[[z[:N][lab == c].mean() for c in range(m)], [z[N:][lab == c].mean() for c in range(m)]]
    return r, Partition(np.bincount(lab)), v


def sec(_t, u):
    return u[0]


sec.direction = 1


def two_traj(part, v, K, e, length, opt, internal=False, returns=False):
    """Largest exponent from two trajectories; optionally internal exponents and section returns of trajectory 1."""
    n, m = part.n, part.m

    def f(_t, u):
        a, b = u[:n], u[n:2 * n]
        out = [part.rhs(a, K, e), part.rhs(b, K, e)]
        if internal:
            W = u[2 * n:].reshape(m, 2, 2)
            Ji = np.zeros((m, 2, 2))
            Ji[:, 0, 0] = (1 - a[:m] ** 2) / e - K
            Ji[:, 0, 1] = -1 / e
            Ji[:, 1, 0] = e
            Ji[:, 1, 1] = -e * GAMMA
            out.append(np.einsum("aij,ajk->aik", Ji, W).ravel())
        return np.concatenate(out)
    g = np.random.default_rng(0).standard_normal(n)
    b = v + D0 * g / np.linalg.norm(g)
    W = np.tile(np.eye(2), (m, 1, 1))
    lsum, isum, t, tr, xs = 0., np.zeros(m), 0., [], []
    while t < length - 1e-9:
        s = solve_ivp(f, (t, t + DT), np.r_[v, b, W.ravel()] if internal else np.r_[v, b],
                      events=sec if returns else None, **opt)
        u = s.y[:, -1]
        v, dd = u[:n], u[n:2 * n] - u[:n]
        nd = np.linalg.norm(dd)
        lsum += np.log(nd / D0)
        b = v + D0 * dd / nd
        if internal:
            W = u[2 * n:].reshape(m, 2, 2)
            for c in range(m):
                q, rr = np.linalg.qr(W[c])
                isum[c] += np.log(abs(rr[0, 0]))
                W[c] = q
        if returns:
            tr += s.t_events[0].tolist()
            xs += [y[:n].tolist() for y in s.y_events[0]]
        t += DT
    return v, lsum / t, isum / t, np.asarray(tr), np.asarray(xs)


def tag(a):
    return "low" if a.src and "low_cluster" in a.src else ""


def verify(a):
    r, part, v = load(a.index)
    e, K = r["eps"], r["K"]
    opt = dict(method="DOP853", rtol=a.rtol, atol=a.rtol / 100, max_step=a.max_step)
    v = solve_ivp(lambda _t, u: part.rhs(u, K, e), (0, 2000), v, **opt).y[:, -1]
    n = part.n

    def f(_t, u):
        return np.r_[part.rhs(u[:n], K, e), (part.jac(u[:n], K, e) @ u[n:].reshape(n, n)).ravel()]
    Q, lsum, t, est = np.eye(n), np.zeros(n), 0., []
    w = v.copy()
    while t < a.length - 1e-9:
        u = solve_ivp(f, (t, t + DT), np.r_[w, Q.ravel()], **opt).y[:, -1]
        w = u[:n]
        Q, R = np.linalg.qr(u[n:].reshape(n, n))
        lsum += np.log(abs(np.diag(R)))
        t += DT
        if abs(t % (a.length / 4)) < 1e-6:
            est.append(dict(t=t, lyap=sorted((lsum / t).tolist(), reverse=True)))
    _, l2, _, _, _ = two_traj(part, v, K, e, a.length, opt)
    res = dict(index=a.index, rtol=a.rtol, max_step=a.max_step, length=a.length, qr=est, two_trajectory=float(l2))
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"verify_{tag(a)}{a.index}_r{a.rtol:g}.json").write_text(json.dumps(res))
    print(a.index, a.rtol, a.max_step, "QR", np.round(est[-1]["lyap"][:4], 5).tolist(), "two-traj", round(l2, 5),
          flush=True)


def long(a):
    r, part, v = load(a.index)
    e, K = r["eps"], r["K"]
    s = solve_ivp(lambda _t, u: part.rhs(u, K, e), (0, a.length), v, method="DOP853", rtol=1e-11, atol=1e-13,
                  max_step=.05, events=sec)
    tr = s.t_events[0]
    ret = np.diff(tr)
    win = [dict(t0=w0, mean=float(ret[(tr[1:] >= w0) & (tr[1:] < w0 + 10000)].mean()),
                std=float(ret[(tr[1:] >= w0) & (tr[1:] < w0 + 10000)].std()))
           for w0 in np.arange(0, a.length, 10000)]
    res = dict(index=a.index, length=a.length, returns=len(tr), windows=win, return_times=ret[-2000:].tolist())
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"long_{tag(a)}{a.index}.json").write_text(json.dumps(res))
    print(a.index, "returns", len(tr), "window std", [round(w["std"], 4) for w in win], flush=True)


def scan(a):
    r, part, v = load(a.index)
    e = r["eps"]
    K = a.q / e
    opt = dict(method="DOP853", rtol=1e-10, atol=1e-12, max_step=.05)
    v = solve_ivp(lambda _t, u: part.rhs(u, K, e), (0, 5000), v, **opt).y[:, -1]
    _, lam, intl, tr, X = two_traj(part, v, K, e, 15000., opt, internal=True, returns=True)
    big = part.sizes > 1
    lag = None
    for L in range(1, 61):
        if len(X) > L + 50 and max(np.max(abs(X[k] - X[k - L])) for k in range(len(X) - 50, len(X))) < 1e-6:
            lag = L
            break
    ret = np.diff(tr)
    res = dict(index=a.index, eps=e, q=a.q, K=K, sizes=part.sizes.tolist(), lyap_max=float(lam),
               internal=intl.tolist(), internal_max=float(intl[big].max()), lag=lag, returns=len(tr),
               return_times=ret[-300:].tolist(), section_states=np.asarray(X[-300:]).tolist())
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"scan_{tag(a)}{a.index}_q{a.q:+.4f}.json").write_text(json.dumps(res))
    print(a.index, f"q={a.q:+.3f}", "lyap", round(lam, 5), "internal_max", round(res["internal_max"], 4), "lag", lag,
          "T", round(ret.mean(), 3), round(ret.std(), 4), flush=True)


def attractor(a):
    """Section states (upward x_1 = 0) over --length after a transient of 5000, and 400 time units of trajectory."""
    r, part, v = load(a.index)
    e, K = r["eps"], a.q / r["eps"]
    opt = dict(method="DOP853", rtol=1e-11, atol=1e-13, max_step=.05)
    v = solve_ivp(lambda _t, u: part.rhs(u, K, e), (0, 5000), v, **opt).y[:, -1]
    s = solve_ivp(lambda _t, u: part.rhs(u, K, e), (0, a.length), v, events=sec, **opt)
    seg = solve_ivp(lambda _t, u: part.rhs(u, K, e), (0, 400), s.y[:, -1], t_eval=np.arange(0, 400, .02), **opt)
    res = dict(index=a.index, eps=e, q=a.q, K=K, sizes=part.sizes.tolist(), p=part.p.tolist(),
               t_sec=s.t_events[0].tolist(), z_sec=s.y_events[0].tolist(), seg_t=seg.t.tolist(),
               seg_z=seg.y.tolist())
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"attractor_{tag(a)}{a.index}_q{a.q:+.3f}.json").write_text(json.dumps(res))
    print(a.index, f"q={a.q:+.3f}", "section points", len(s.t_events[0]), flush=True)


def main():
    global SRC
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=None, help="JSON list with final_state entries (default: many_cluster_check)")
    ap.add_argument("mode", choices=["verify", "long", "scan", "attractor"])
    ap.add_argument("--index", type=int, default=95)
    ap.add_argument("--rtol", type=float, default=1e-12)
    ap.add_argument("--max-step", type=float, default=.02)
    ap.add_argument("--length", type=float, default=40000.)
    ap.add_argument("--q", type=float, default=-1.)
    a = ap.parse_args()
    if a.src:
        SRC = ROOT / a.src
    {"verify": verify, "long": long, "scan": scan, "attractor": attractor}[a.mode](a)


if __name__ == "__main__":
    main()
