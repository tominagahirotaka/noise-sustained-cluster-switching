#!/usr/bin/env python3
"""A0 of the tiny-noise experiment (2026-10-25): noiseless_n50.Forest.kick on a fresh tree (start record 1600 of
rn_0088_n1e-08_t80000_states, every third element hung under the previous one at 1e-25 or 1e-35) for each eta, against
exact positions in 60-digit decimal arithmetic. Run from the project root: python3 scripts/test_noise_kick.py
2026-10-31: the printed lines are also saved, with the input and script hashes, to
analysis/noiseless_n50/test_noise_kick.json (the earlier run had console output only).
"""
import hashlib
import json
import sys
import numpy as np
from decimal import Decimal as D, getcontext
sys.path.insert(0, "scripts")
import noiseless_n50 as M
getcontext().prec = 60
z0 = np.load(M.ROOT / "analysis/switching_check/rn_0088_n1e-08_t80000_states_z.npy")[1600]
n = z0.size // 2

def fresh(seed):
    F = M.Forest(np.c_[z0[:n], z0[n:]])
    r = np.random.default_rng(seed)
    for k in range(3, n, 3):  # element parents with tiny deviations (pairs at 1e-25 and 1e-35)
        if F.par[k] == F.par[k - 1]:
            F.par[k] = k - 1
            F.set_dev(k, r.normal(0, 1e-25 if k % 2 else 1e-35, 2))
    F.order = F.topo()
    return F

def exact(F):
    P = {}
    for k in F.order:
        if F.par[k] < 0:
            P[k] = [D(float(F.G[k, c])) for c in range(2)]
        else:
            e = D(float(F.s[k])).exp() if np.isfinite(F.s[k]) else D(0)
            P[k] = [P[F.par[k]][c] + e * D(float(F.u[k, c])) for c in range(2)]
    return P

rows = []
for eta in (1e-6, 1e-12, 1e-20, 1e-30, 1e-40):
    F = fresh(0)
    P0 = exact(F)
    w = np.random.default_rng(1).normal(0, eta, (n, 2))
    F.kick(w)
    P1 = exact(F)
    W = lambda k, c: D(float(w[k, c])) if k < n else D(0)
    def err(i, j):
        return max(abs((P1[i][c] - P1[j][c]) - (P0[i][c] - P0[j][c]) - (W(i, c) - W(j, c))) for c in range(2))
    def sep(i, j):
        return max(abs(P0[i][c] - P0[j][c]) for c in range(2))
    ghost_moved = max(abs(P1[k][c] - P0[k][c]) for k in range(n, len(F.par)) for c in range(2))
    hung = [(k - 1, k) for k in range(3, n, 3) if F.par[k] == k - 1]
    e_h = max(err(i, j) for i, j in hung) / D(eta)
    el_root = [(k, F.par[k]) for k in range(n) if F.par[k] >= n]  # element vs its ghost
    e_g = max(err(i, j) / max(sep(i, j), D(eta)) for i, j in el_root)
    rows.append(dict(eta=eta, pairs_err_over_eta=float(e_h), element_ghost_err_over_max_sep_eta=float(e_g),
                     ghosts_moved=float(ghost_moved), n_hung_pairs=len(hung), n_element_ghost=len(el_root)))
    print(f"eta {eta:.0e}: pairs at 1e-25/1e-35 err/eta {float(e_h):.1e}; element-ghost (sep up to 1e-3) err/max(sep,eta) {float(e_g):.1e}; ghosts moved {float(ghost_moved):.1e}")

def sha(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()

src = M.ROOT / "analysis/switching_check/rn_0088_n1e-08_t80000_states_z.npy"
(M.ROOT / "analysis/noiseless_n50/test_noise_kick.json").write_text(json.dumps(dict(
    script="scripts/test_noise_kick.py", script_sha256=sha(__file__), noiseless_n50_sha256=sha(M.__file__),
    input={str(src.relative_to(M.ROOT)): sha(src)}, record=1600, precision_digits=60, kick_seed=1, tree_seed=0,
    numpy=np.__version__, python=sys.version.split()[0], rows=rows), indent=1))
