#!/usr/bin/env python3
"""Heteroclinic connections between the saddle cluster orbits of the N = 50 slow switching (exploratory, 2026-10-23).

For a node (partition --sizes at eps = 0.25, eps*K = --q, started from the same random centroids as
itinerancy_n50.py reduced --seed): transient 5000, shoot the periodic orbit (section x_1 = 0 upward), compute the
internal (2x2 block) monodromy of every cluster and take the transversely unstable cluster U. The unstable manifold
of the orbit is followed inside the subspace where one element of U separates: partition with U reduced by one plus
a singleton, started on the orbit with the singleton displaced by +-delta along the unstable internal Floquet vector.
The integration (DOP853, rtol 1e-12) runs for --tend; the distance of the singleton from every cluster (max over
x and y, maximum over each period) is recorded. If the singleton merges into a cluster (distance < 1e-7 at the end),
the merged partition is shot again from the final state and audited: period, internal exponents, which cluster is
unstable. Growth and contraction rates are fitted to the log-distance between 1e-7 and 1e-3. Every pair of components
is also checked for a merge (pair_merges, added 2026-10-27: in (16,16,16,1,1) a singleton joins the rest of U).

  python3 scripts/heteroclinic_n50.py --sizes 17,17,16 --q -1.0 --seed 1
-> analysis/heteroclinic_n50/node_<sizes>_q<q>_s<seed>.json
"""
from pathlib import Path
import argparse
import json
import sys

import numpy as np
from scipy.integrate import solve_ivp

sys.path.insert(0, str(Path(__file__).resolve().parent))
from eps_three_cluster import Partition, ROOT, GAMMA  # noqa: E402

OUT = ROOT / "analysis/heteroclinic_n50"
OPT = dict(method="DOP853", rtol=1e-12, atol=1e-15, max_step=.05)


def sec(_t, u):
    return u[0]


sec.direction = 1
sec.terminal = True


def to_section(part, z, K, e):
    """Advance past t = 1 to the next upward x_1 = 0 crossing; return the point and the elapsed time."""
    z = solve_ivp(lambda _t, u: part.rhs(u, K, e), (0, 1), z, **OPT).y[:, -1]
    s = solve_ivp(lambda _t, u: part.rhs(u, K, e), (0, 300), z, events=sec, **OPT)
    return s.y_events[0][0], 1 + s.t_events[0][0]


def orbit(part, z, K, e):
    """Shoot from a point near the section (period guess: the next return); return the orbit point, period,
    internal monodromies and firing phases."""
    z, _ = to_section(part, z, K, e)
    _, T0 = to_section(part, z, K, e)
    w = part.shoot(np.r_[z, T0], K, e, 2e-12)
    if w is None:
        return None
    m, n, T = part.m, part.n, w[-1]

    def f(_t, v):
        zz = v[:n]
        A = np.zeros((m, 2, 2))
        A[:, 0, 0] = (1 - zz[:m] ** 2) / e - K
        A[:, 0, 1] = -1 / e
        A[:, 1, 0] = e
        A[:, 1, 1] = -e * GAMMA
        return np.r_[part.rhs(zz, K, e), (A @ v[n:].reshape(m, 2, 2)).ravel()]

    s = solve_ivp(f, (0, T), np.r_[w[:n], np.tile(np.eye(2), (m, 1, 1)).ravel()], dense_output=True, **OPT)
    B = s.y[n:, -1].reshape(m, 2, 2)
    t = np.linspace(0, T, 20001)
    x = s.sol(t)[:m]
    fire = [float(t[np.argmax((x[c, :-1] < 0) & (x[c, 1:] >= 0))]) for c in range(m)]  # upward x = 0
    return w[:n], float(T), B, fire


def rate(t, ld, lo, hi):
    """Slope of log(distance) over the part of the record with lo < distance < hi."""
    k = (ld > np.log(lo)) & (ld < np.log(hi))
    return float(np.polyfit(t[k], ld[k], 1)[0]) if k.sum() > 3 else None


def branch(part, z, K, e, T, U, v, delta, tend):
    m = part.m
    sizes = part.sizes.copy()
    sizes[U] -= 1
    sp = Partition(np.r_[sizes, 1])
    x, y = z[:m], z[m:]
    z0 = np.r_[x, x[U] + delta * v[0], y, y[U] + delta * v[1]]
    s = solve_ivp(lambda _t, u: sp.rhs(u, K, e), (0, tend), z0, dense_output=True, **OPT)
    nper = int(tend / T)
    tt = np.linspace(0, nper * T, nper * 400 + 1)
    Z = s.sol(tt)
    d = np.maximum(abs(Z[:m] - Z[m]), abs(Z[m + 1:2 * m + 1] - Z[2 * m + 1]))  # singleton vs each cluster
    D = np.maximum(d[:, :-1].reshape(m, nper, 400).max(axis=2), 1e-300)
    tp = (np.arange(nper) + .5) * T
    L = np.log(D)
    tgt = int(np.argmin(D[:, -1]))
    res = dict(delta=delta, log10_dist_end=np.log10(D[:, -1]).tolist(),
               growth_from_parent=rate(tp, L[U], 1e-7, 1e-3), target=tgt, merged=bool(D[tgt, -1] < 1e-7))
    k = np.nonzero(D[U] > 1e-2)[0]
    res["t_leave"] = float(tp[k[0]]) if len(k) else None
    if len(k):  # firing time of the singleton minus that of its parent cluster, just after leaving
        t = np.linspace(tp[k[0]], tp[k[0]] + 2 * T, 40001)
        X = s.sol(t)
        f = [t[np.nonzero((X[c, :-1] < 0) & (X[c, 1:] >= 0))[0]] for c in (m, U)]
        off = float((f[0][0] - f[1][np.argmin(abs(f[1] - f[0][0]))] + T / 2) % T - T / 2)
        res["offset_from_parent"] = off
        res["direction"] = "advanced" if off < 0 else "retarded"
    if res["merged"]:
        j = np.nonzero(D[tgt] > 1e-3)[0]
        tail = slice(j[-1], None) if len(j) else slice(None)
        res["contraction_into_target"] = rate(tp[tail], L[tgt][tail], 1e-9, 1e-3)
        res["t_merge_1e-3"] = float(tp[j[-1]]) if len(j) else None
    # every pair of components (not only the singleton): which pairs merge, when, and how fast (e.g. another
    # singleton that joins the rest of U while the separated element stays single)
    P = []
    for i in range(m + 1):
        for j in range(i + 1, m + 1):
            dp = np.maximum(abs(Z[i] - Z[j]), abs(Z[m + 1 + i] - Z[m + 1 + j]))
            Dp = np.maximum(dp[:-1].reshape(nper, 400).max(axis=1), 1e-300)
            if Dp[0] > 1e-3 and Dp[-1] < 1e-7:
                jj = np.nonzero(Dp > 1e-3)[0][-1]
                Lp = np.log(Dp)
                P.append({"pair": [i, j], "t_merge_1e-3": float(tp[jj]),
                          "contraction": rate(tp[jj:], Lp[jj:], 1e-9, 1e-3)})
    res["pair_merges"] = P
    res["final"] = s.y[:, -1].tolist()
    res["log10_dist_record"] = np.round(np.log10(D[:, ::5]), 2).tolist()
    return res, sizes


def describe(part, z, K, e, ref=None):
    """Shoot and audit the orbit through z: period, internal exponents, unstable cluster, firing order from it."""
    o = orbit(part, z, K, e)
    if o is None:
        return None
    z, T, B, fire = o
    sizes = part.sizes.tolist()
    lam = [float(np.log(max(abs(np.linalg.eigvals(b)))) / T) for b in B]
    U = int(np.argmax([lam[c] if sizes[c] > 1 else -np.inf for c in range(len(sizes))]))
    order = sorted(range(len(sizes)), key=lambda c: (fire[c] - fire[U]) % T)
    m = len(sizes)
    zz = z.reshape(2, m)
    rep = [min(c for c in range(m) if np.max(abs(zz[:, c] - zz[:, j])) < 1e-6) for j in range(m)]
    eff = {}
    for j in range(m):  # clusters on the same trajectory count as one (e.g. a singleton that rejoined)
        eff[rep[j]] = eff.get(rep[j], 0) + sizes[j]
    eff_set = sorted((eff[c], round(lam[c], 4)) for c in eff)
    d = dict(sizes=sizes, period=T, internal=lam, unstable=U, firing_order_from_U=order, orbit_point=z.tolist(),
             coincide=rep, effective=eff_set)
    if ref is not None:
        d["same_as_node"] = bool(abs(T - ref["period"]) < 1e-6 and eff_set == [tuple(x) for x in ref["effective"]])
    return d, B


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sizes", default="17,17,16")
    ap.add_argument("--q", type=float, default=-1.)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--eps", type=float, default=.25)
    ap.add_argument("--init", default=None, help="json with sizes and orbit_point (e.g. an end orbit) instead of --seed")
    ap.add_argument("--tag", default=None)
    ap.add_argument("--tend", type=float, default=6000.)
    ap.add_argument("--deltas", default="1e-8,-1e-8")
    a = ap.parse_args()
    e, K = a.eps, a.q / a.eps
    if a.init:
        d0 = json.loads(Path(a.init).read_text())
        sizes, v = d0["sizes"], np.asarray(d0["orbit_point"])
        part = Partition(sizes)
    else:
        sizes = [int(s) for s in a.sizes.split(",")]
        part = Partition(sizes)
        rng = np.random.default_rng(a.seed)
        v = np.r_[rng.uniform(-2, 2, len(sizes)), rng.uniform(-.6, 1.4, len(sizes))]
        v = solve_ivp(lambda _t, u: part.rhs(u, K, e), (0, 5000), v, method="DOP853", rtol=1e-10, atol=1e-12,
                      max_step=.05).y[:, -1]
    r0 = describe(part, v, K, e)
    if r0 is None:
        raise SystemExit("node shooting failed")
    node, B = r0
    node.update(eps=e, K=K, q=a.q, seed=a.seed, init=a.init)
    z, T, U, m = np.asarray(node["orbit_point"]), node["period"], node["unstable"], len(sizes)
    w, V = np.linalg.eig(B[U])
    k = int(np.argmax(abs(w)))
    vu = np.real(V[:, k])
    vu /= np.max(abs(vu))
    node["unstable_multiplier"] = [float(w[k].real), float(w[k].imag)]
    print("node", sizes, "q", a.q, "T", round(T, 6), "internal", np.round(node["internal"], 5).tolist(), "U", U,
          "order", node["firing_order_from_U"], flush=True)
    branches = []
    for delta in [float(s) for s in a.deltas.split(",")]:
        r, ss = branch(part, z, K, e, T, U, vu, delta, a.tend)
        fz = np.asarray(r["final"])
        if r["merged"]:
            merged = ss.copy()
            merged[r["target"]] += 1
            end = describe(Partition(merged), np.r_[fz[:m], fz[m + 1:2 * m + 1]], K, e, node)
        else:
            end = describe(Partition(np.r_[ss, 1]), fz, K, e, node)
        r["end"] = None if end is None else end[0]
        msg = f"  {r.get('direction')} ({r.get('offset_from_parent', 0):+.3f}) leave {r['t_leave']}" \
              f" growth {r['growth_from_parent']} merged {r['merged']} target {r['target']}" \
              f" contraction {r.get('contraction_into_target')} pair_merges {r['pair_merges']}"
        if end is not None:
            d = end[0]
            msg += f" -> {d['sizes']} T {d['period']:.6f} U {d['unstable']} order {d['firing_order_from_U']}" \
                   f" internal {np.round(d['internal'], 5).tolist()} effective {d['effective']}" \
                   f" same_as_node {d.get('same_as_node')}"
        print(msg, flush=True)
        branches.append(r)
    OUT.mkdir(parents=True, exist_ok=True)
    tag = a.tag or f"{'-'.join(map(str, sizes))}_q{a.q:+.3f}_s{a.seed}"
    (OUT / f"node_{tag}.json").write_text(json.dumps(dict(node=node, branches=branches)))


if __name__ == "__main__":
    main()
