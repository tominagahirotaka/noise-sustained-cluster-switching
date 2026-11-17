#!/usr/bin/env python3
"""Noiseless continuation of the N = 50 slow switching without the loss of small deviations to rounding
(exploratory, 2026-10-23; version 2, tree representation).

Representation: a forest of nodes. A node is an element or a "ghost" (an uncoupled copy of the element equation: it
feels the mean field but does not contribute to it). A root carries its absolute state; every other node carries its
deviation from its parent as exp(s) u with |u|_max = 1 and a separate log-scale s (s = -inf: on the parent). The
deviation obeys the exact factored equations

  d(dx)/dt = dx [(1 - (x^2 + x p + p^2)/3)/eps - K] - dy/eps,   d(dy)/dt = eps (dx - gamma dy),

with x the node's and p the parent's x, so a deviation keeps its relative precision at any size (no underflow, no
rounding into exact synchrony). Relative positions are only ever formed along the tree. RK4, h = 0.005 as in the
full-system runs; u is renormalized after every step.

Bookkeeping every time unit (absolute positions X are used only to find nearby nodes):
  R1  a root, or a node farther than 1e-3 from its parent, that comes within --join (1e-6) of a node outside its own
      subtree and ancestors is re-attached to that node, with its deviation X_a - X_b (precise at that size).
  R2  two children of one parent whose deviations agree to a relative 1e-6 are put under a new ghost (a copy of the
      first child's trajectory) before their difference is lost to rounding.
  R3  ghosts without element descendants are removed; a ghost with one child is replaced by that child.
Initial tree: one root ghost per connected group (distance < 1e-3) of the start state, at its centroid.
Version 1 (one ghost level, archived in analysis/noiseless_n50/v1_ghost_artifact/) let ghosts of one cluster
converge in absolute coordinates, which reintroduced exact synchrony at the ghost level.

Start: the state of analysis/itinerancy_n50/trace_<index>[_n<noise>].npz at time --t0 (noise eta = 1e-10 up to then);
afterwards no noise. --check T compares with a plain double-precision RK4 run (partition at 1e-2 every 100 units).
Records every time unit: the label of each element's group (elements closer than 1e-3 in absolute coordinates) and
log10 of each element's deviation from its parent; every 10 time units ("spread"): for each element of a group of >= 5,
log10 of the median of its distances to the other members, computed along the tree.
-> analysis/noiseless_n50/nl_<index>[_n<noise>]_t<t0>.npz

  python3 scripts/noiseless_n50.py --index 394 --t0 5000 --tmax 200000
  --states <tag>: start instead from record --t0 of analysis/switching_check/<tag>_z.npy (switching_renoise.py
  --states; eps, K from <tag>.json) -> analysis/noiseless_n50/nl_<tag>_r<t0>.npz (2026-10-24).
  --every k: records every k time units (group, logdev) and every 10 k (spread); default 1 as before.
  --eta s (2026-10-25): every --kick_every (10) time units, after the record, independent Gaussian kicks of sd s on both
  coordinates of every element (the noise of switching_renoise.py), applied exactly to the deviations (Forest.kick).
  --seed, --h (time step), --track i j ... (log10 distance of element pairs every time unit, along the tree),
  --r1_jitter s (extra absolute error at every R1 re-attachment; sensitivity test); R1 events are saved (r1_events).
  Names get _e<eta>_s<seed>, _k, _h, _j, _jit and --tag; eta = 0 with the defaults gives the earlier run unchanged.
"""
from pathlib import Path
import argparse
import json
import sys

import numpy as np
from numba import njit
from scipy.sparse.csgraph import connected_components

sys.path.insert(0, str(Path(__file__).resolve().parent))
from eps_three_cluster import ROOT  # noqa: E402

IT = ROOT / "analysis/itinerancy_n50"
OUT = ROOT / "analysis/noiseless_n50"
B, GAM = .7, .5
H, SUB = .005, 200


@njit(cache=True)
def positions(order, par, G, s, u):
    X = np.empty_like(G)
    for k in order:
        if par[k] < 0:
            X[k] = G[k]
        else:
            X[k] = X[par[k]] + np.exp(s[k]) * u[k]
    return X


@njit(cache=True)
def deriv(order, par, elem, G, s, u, e, K):
    X = positions(order, par, G, s, u)
    xm = 0.
    ne = 0
    for k in range(G.shape[0]):
        if elem[k]:
            xm += X[k, 0]
            ne += 1
    xm /= ne
    dG = np.zeros_like(G)
    du = np.zeros_like(u)
    for k in order:
        if par[k] < 0:
            x, y = G[k, 0], G[k, 1]
            dG[k, 0] = (x - x ** 3 / 3 - y) / e + K * (xm - x)
            dG[k, 1] = e * (x + B - GAM * y)
        else:
            x, p = X[k, 0], X[par[k], 0]
            a = (1 - (x * x + x * p + p * p) / 3) / e - K
            du[k, 0] = a * u[k, 0] - u[k, 1] / e
            du[k, 1] = e * (u[k, 0] - GAM * u[k, 1])
    return dG, du


@njit(cache=True)
def advance(order, par, elem, G, s, u, e, K, nsteps, h):
    for _ in range(nsteps):
        k1G, k1u = deriv(order, par, elem, G, s, u, e, K)
        k2G, k2u = deriv(order, par, elem, G + h / 2 * k1G, s, u + h / 2 * k1u, e, K)
        k3G, k3u = deriv(order, par, elem, G + h / 2 * k2G, s, u + h / 2 * k2u, e, K)
        k4G, k4u = deriv(order, par, elem, G + h * k3G, s, u + h * k3u, e, K)
        G = G + h / 6 * (k1G + 2 * k2G + 2 * k3G + k4G)
        u = u + h / 6 * (k1u + 2 * k2u + 2 * k3u + k4u)
        for k in order:
            if par[k] >= 0:
                m = max(abs(u[k, 0]), abs(u[k, 1]))
                if m > 0:
                    u[k] /= m
                    s[k] += np.log(m)
    return G, s, u


def scaled_sum(s1, u1, s2, u2, sign=1.):
    """exp(s1) u1 + sign exp(s2) u2 as (s, u)."""
    sm = max(s1, s2)
    if not np.isfinite(sm):
        return -np.inf, np.array([1., 0.])
    d = np.exp(s1 - sm) * u1 + sign * np.exp(s2 - sm) * u2
    m = np.max(abs(d))
    return (sm + np.log(m), d / m) if m > 0 else (-np.inf, np.array([1., 0.]))


class Forest:
    """Node arrays: par (-1 = root), elem (nodes 0..n-1 are the elements), G (absolute state of roots), s, u."""

    def __init__(self, Z):
        n = len(Z)
        lab = connected_components(np.max(abs(Z[:, None] - Z[None]), axis=2) < 1e-3, directed=False)[1]
        ng = lab.max() + 1
        self.n = n
        self.par = np.r_[n + lab, -np.ones(ng, dtype=np.int64)].astype(np.int64)
        self.elem = np.r_[np.ones(n, dtype=bool), np.zeros(ng, dtype=bool)]
        self.G = np.zeros((n + ng, 2))
        self.s = np.full(n + ng, -np.inf)
        self.u = np.tile([1., 0.], (n + ng, 1))
        for g in range(ng):
            self.G[n + g] = Z[lab == g].mean(axis=0)
        for i in range(n):
            self.set_dev(i, Z[i] - self.G[n + lab[i]])
        self.order = self.topo()

    def set_dev(self, k, d):
        m = np.max(abs(d))
        if m > 0:
            self.s[k], self.u[k] = np.log(m), d / m
        else:
            self.s[k], self.u[k] = -np.inf, (1., 0.)

    def add_node(self, parent, s, u):
        self.par = np.r_[self.par, parent]
        self.elem = np.r_[self.elem, False]
        self.G = np.r_[self.G, [[0., 0.]]]
        self.s = np.r_[self.s, s]
        self.u = np.r_[self.u, [u]]
        return len(self.par) - 1

    def topo(self):
        depth = np.zeros(len(self.par), dtype=np.int64)
        for k in range(len(self.par)):
            j = k
            while self.par[j] >= 0:
                j = self.par[j]
                depth[k] += 1
        return np.argsort(depth, kind="stable").astype(np.int64)

    def ancestors(self, k):
        out = set()
        while self.par[k] >= 0:
            k = self.par[k]
            out.add(k)
        return out

    def subtree(self, k):
        out, todo = {k}, [k]
        while todo:
            ch = np.nonzero(self.par == todo.pop())[0].tolist()
            out.update(ch)
            todo += ch
        return out

    def X(self):
        return positions(self.order, self.par, self.G, self.s, self.u)

    def log_dist(self, i, j):
        """log10 max-norm of X_i - X_j, summed along the tree (no rounding to zero for tiny distances)."""
        pi, pj = [i], [j]
        while self.par[pi[-1]] >= 0:
            pi.append(self.par[pi[-1]])
        while self.par[pj[-1]] >= 0:
            pj.append(self.par[pj[-1]])
        common = set(pi) & set(pj)
        terms = [(self.s[k], self.u[k], 1.) for k in pi if k not in common]
        terms += [(self.s[k], self.u[k], -1.) for k in pj if k not in common]
        if not common:  # different trees: absolute difference
            return float(np.log10(max(np.max(abs(self.X()[i] - self.X()[j])), 1e-300)))
        terms = [t for t in terms if np.isfinite(t[0])]
        if not terms:
            return -np.inf
        sm = max(t[0] for t in terms)
        d = sum(sg * np.exp(sk - sm) * uk for sk, uk, sg in terms)
        m = np.max(abs(d))
        return float((sm + np.log(m)) / np.log(10)) if m > 0 else -np.inf

    def spread(self, lab):
        """For every element: log10 of the median distance to the other members of its group (groups >= 5)."""
        out = np.full(self.n, np.nan)
        for g in np.unique(lab):
            mem = np.nonzero(lab == g)[0]
            if len(mem) < 5:
                continue
            D = np.full((len(mem), len(mem)), np.nan)
            for x1 in range(len(mem)):
                for x2 in range(x1 + 1, len(mem)):
                    D[x1, x2] = D[x2, x1] = self.log_dist(mem[x1], mem[x2])
            out[mem] = np.nanmedian(D, axis=1)
        return out

    def kick(self, w):
        """Independent kicks w (n x 2) on the absolute positions of the elements; ghosts are not kicked. All kicks are
        drawn first; every node's deviation from its parent changes by W_k - W_parent (W = 0 for ghosts), roots by W."""
        W = np.zeros((len(self.par), 2))
        W[:self.n] = w
        for k in range(len(self.par)):
            p = self.par[k]
            if p < 0:
                self.G[k] = self.G[k] + W[k]
                continue
            dW = W[k] - W[p]
            m = np.max(abs(dW))
            if m > 0:
                self.s[k], self.u[k] = scaled_sum(self.s[k], self.u[k], np.log(m), dW / m)

    def remove(self, k):
        keep = np.ones(len(self.par), dtype=bool)
        keep[k] = False
        remap = np.cumsum(keep) - 1
        par = self.par[keep]
        self.par = np.where(par >= 0, remap[np.maximum(par, 0)], -1).astype(np.int64)
        self.elem, self.G, self.s, self.u = self.elem[keep], self.G[keep], self.s[keep], self.u[keep]

    def bookkeeping(self, join, jitter=0., rng=None, log=None, t=0):
        """log: list for the R1 events [t, a, b, log10 |X_a - X_b| (absolute coordinates, used as the new deviation),
        size of the subtree of a]; jitter: sd of an extra absolute error added to that deviation (sensitivity test)."""
        X = self.X()
        log_far = np.log(1e-3)
        n_r1 = n_r2 = 0
        # R1: re-attach a root or a far node to a node it has come close to
        for a in range(len(self.par)):
            if not (self.par[a] < 0 or self.s[a] > log_far):
                continue
            d = np.max(abs(X - X[a]), axis=1)
            d[a] = np.inf
            if d.min() >= join:
                continue
            d[list(self.subtree(a) | self.ancestors(a))] = np.inf
            b = int(np.argmin(d))
            if d[b] < join:
                if self.par[a] < 0:
                    self.G[a] = 0.
                self.par[a] = b
                dev = X[a] - X[b]
                if jitter > 0:
                    dev = dev + rng.normal(0, jitter, 2)
                self.set_dev(a, dev)
                if log is not None:
                    log.append([t, a, b, float(np.log10(max(np.max(abs(X[a] - X[b])), 1e-300))), len(self.subtree(a))])
                self.order = self.topo()
                X = self.X()
                n_r1 += 1
        # R2: separate nearly equal siblings under a new ghost before rounding merges them
        for p in np.unique(self.par[self.par >= 0]):
            ch = [k for k in np.nonzero(self.par == p)[0] if np.isfinite(self.s[k])]
            done = set()
            for x1 in range(len(ch)):
                for x2 in range(x1 + 1, len(ch)):
                    j, k = ch[x1], ch[x2]
                    if j in done or k in done:
                        continue
                    sd, ud = scaled_sum(self.s[k], self.u[k], self.s[j], self.u[j], -1.)
                    if sd < max(self.s[j], self.s[k]) + np.log(1e-6):
                        g = self.add_node(p, self.s[j], self.u[j].copy())
                        self.par[j], self.s[j], self.u[j] = g, -np.inf, (1., 0.)
                        self.par[k], self.s[k], self.u[k] = g, sd, ud
                        done.update((j, k))
                        n_r2 += 1
        # R3: drop empty ghosts, splice out ghosts with a single child
        changed = True
        while changed:
            changed = False
            for k in range(self.n, len(self.par)):
                ch = np.nonzero(self.par == k)[0]
                if len(ch) > 1:
                    continue
                if len(ch) == 1:
                    c = ch[0]
                    if self.par[k] < 0:
                        self.order = self.topo()
                        self.G[c], self.par[c] = self.X()[c], -1
                        self.s[c], self.u[c] = -np.inf, (1., 0.)
                    else:
                        self.s[c], self.u[c] = scaled_sum(self.s[k], self.u[k], self.s[c], self.u[c])
                        self.par[c] = self.par[k]
                self.remove(k)
                changed = True
                break
        self.order = self.topo()
        return n_r1, n_r2


def plain(z, e, K, nunits, H=H):
    n = z.size // 2
    rec = np.empty((nunits, 2 * n))

    def f(v):
        x, y = v[:n], v[n:]
        return np.r_[(x - x ** 3 / 3 - y) / e + K * (x.mean() - x), e * (x + B - GAM * y)]
    for t in range(nunits):
        for _ in range(int(round(1 / H))):
            k1 = f(z)
            k2 = f(z + H / 2 * k1)
            k3 = f(z + H / 2 * k2)
            k4 = f(z + H * k3)
            z = z + H / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
        rec[t] = z
    return rec


def groups(P, th):
    return connected_components(np.max(abs(P[:, None] - P[None]), axis=2) < th, directed=False)[1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", type=int, default=394)
    ap.add_argument("--noise", default="")
    ap.add_argument("--t0", type=int, default=5000)
    ap.add_argument("--tmax", type=int, default=200000)
    ap.add_argument("--join", type=float, default=1e-6)
    ap.add_argument("--check", type=int, default=0)
    ap.add_argument("--states", default="", help="tag of analysis/switching_check/<tag>_z.npy; --t0 is then a record")
    ap.add_argument("--every", type=int, default=1)
    ap.add_argument("--eta", type=float, default=0., help="sd of the kicks on every element coordinate (2026-10-25)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--kick_every", type=int, default=10, help="time units between kicks (after the record)")
    ap.add_argument("--h", type=float, default=H)
    ap.add_argument("--track", type=int, nargs="*", default=[], help="element pairs i j ...: log10 distance every unit")
    ap.add_argument("--r1_jitter", type=float, default=0., help="extra absolute error at every R1 re-attachment")
    ap.add_argument("--tag", default="")
    a = ap.parse_args()
    sub = int(round(1 / a.h))
    rng = np.random.default_rng(a.seed)
    pairs = list(zip(a.track[::2], a.track[1::2]))
    if a.states:
        SC = ROOT / "analysis/switching_check"
        js = json.loads((SC / f"{a.states}.json").read_text())
        e, K = float(js["eps"]), float(js["K"])
        z0 = np.load(SC / f"{a.states}_z.npy")[a.t0].astype(float)
    else:
        tr = np.load(IT / f"trace_{a.index:04d}{a.noise}.npz")
        e, K = float(tr["eps"]), float(tr["K"])
        z0 = tr["z"][a.t0].astype(float)
    n = z0.size // 2
    F = Forest(np.c_[z0[:n], z0[n:]])
    print(a.index, "e", e, "K", K, "start groups", np.bincount(F.par[:n] - n).tolist(), flush=True)
    ev = a.every
    rec_g = np.empty((a.tmax // ev, n), dtype=np.int16)
    rec_l = np.empty((a.tmax // ev, n), dtype=np.float32)
    rec_sp = np.empty((a.tmax // (10 * ev), n), dtype=np.float32)
    nr1 = nr2 = 0
    rec_tr = np.empty((a.tmax, len(pairs)), dtype=np.float64)
    r1log = []
    chk = plain(z0.copy(), e, K, a.check, a.h) if a.check else None
    for t in range(a.tmax):
        F.G, F.s, F.u = advance(F.order, F.par, F.elem, F.G, F.s, F.u, e, K, sub, a.h)
        r1, r2 = F.bookkeeping(a.join, a.r1_jitter, rng, r1log, t + 1)
        for q, (i, j) in enumerate(pairs):
            rec_tr[t, q] = F.log_dist(i, j)
        nr1, nr2 = nr1 + r1, nr2 + r2
        if (t + 1) % ev == 0 or (chk is not None and t < a.check):
            X = F.X()[:n]
            g = groups(X, 1e-3)
        if (t + 1) % ev == 0:
            r = t // ev
            rec_g[r] = g
            rec_l[r] = np.where(np.isfinite(F.s[:n]), (F.s[:n] + np.log(np.max(abs(F.u[:n]), axis=1))) / np.log(10),
                                -np.inf)
            if (r + 1) % 10 == 0:
                rec_sp[r // 10] = F.spread(g)
        if chk is not None and t < a.check and (t + 1) % 100 == 0:
            P = np.c_[chk[t][:n], chk[t][n:]]
            l1, l2 = groups(X, 1e-2), groups(P, 1e-2)
            same = all(len(set(l2[l1 == g])) == 1 for g in set(l1)) and len(set(l1)) == len(set(l2))
            print(f"  check t={t + 1} same partition {same} sizes {sorted(np.bincount(l1).tolist(), reverse=True)}"
                  f" plain {sorted(np.bincount(l2).tolist(), reverse=True)}", flush=True)
        if a.eta > 0 and (t + 1) % a.kick_every == 0:
            F.kick(rng.normal(0, a.eta, (n, 2)))
        if (t + 1) % 5000 == 0:
            r = t // ev
            fin = rec_l[r][np.isfinite(rec_l[r])]
            print(f"  t={t + 1} nodes {len(F.par)} groups {sorted(np.bincount(rec_g[r]).tolist(), reverse=True)[:8]}"
                  f" R1 {nr1} R2 {nr2} min log10 dev {fin.min() if len(fin) else np.nan:.1f}", flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    name = f"nl_{a.states}_r{a.t0}" if a.states else f"nl_{a.index:04d}{a.noise}_t{a.t0}"
    if a.eta > 0:
        name += f"_e{a.eta:g}_s{a.seed}"  # :g, not :.0e (2026-10-27: 1.5e-20 and 2e-20 both became e2e-20) + ("" if a.kick_every == 10 else f"_k{a.kick_every}")
    if a.h != H:
        name += f"_h{a.h:g}"
    if a.join != 1e-6:
        name += f"_j{a.join:.0e}"
    if a.r1_jitter > 0:
        name += f"_jit{a.r1_jitter:.0e}"
    name += a.tag + ".npz"
    np.savez_compressed(OUT / name, group=rec_g, logdev=rec_l, spread=rec_sp, eps=e, K=K,
                        t0=a.t0, join=a.join, r1=nr1, r2=nr2, every=ev, eta=a.eta, seed=a.seed, kick_every=a.kick_every,
                        h=a.h, r1_jitter=a.r1_jitter, track=rec_tr, track_pairs=np.asarray(pairs, dtype=np.int64).reshape(-1, 2),
                        r1_events=np.asarray(r1log, dtype=float).reshape(-1, 5))
    print(a.index, "done", flush=True)


if __name__ == "__main__":
    main()
