#!/usr/bin/env python3
"""Exactly balanced three-cluster partition (fractions 1/3 each) at eps = 0.25 (2026-11-03, author's approval).

The N of the ensembles (50, 100, 200, 400, 1000) are not divisible by 3, so the exactly balanced three-cluster partition
never occurs there; balanced_partitions.json has only nearly balanced ones. The reduced system depends on the populations
only through the fractions, so (17,17,17) stands for every N divisible by 3; (33,33,33) is computed as a check that it
gives the same orbits. (17,17,16) and (34,33,33) are recomputed with the same procedure to compare with
balanced_partitions.json. Procedure of balanced_partitions_n.py: random centroids (seeds 1-3), transient 5000 (DOP853,
rtol 1e-10), shooting and internal exponents (heteroclinic_n50.describe), distinct orbits by period; in addition the
collective multipliers of every orbit (Partition.audit, tol 2e-12) and the phase offsets of the firings (fraction of
the period from the first cluster), which show whether the orbit is the symmetric splay orbit (offsets 1/3, 2/3).
  -> analysis/heteroclinic_n50/equal_partition.json

  python3 scripts/paper3_equal_partition.py
"""
from pathlib import Path
import hashlib
import json
import sys

import numpy as np
from scipy.integrate import solve_ivp

sys.path.insert(0, str(Path(__file__).resolve().parent))
from eps_three_cluster import Partition, ROOT  # noqa: E402
import heteroclinic_n50 as hc  # noqa: E402

PARTS = [(17, 17, 17), (33, 33, 33), (17, 17, 16), (34, 33, 33)]
QS = (-.8, -.9, -1., -1.2)
EPS = .25


def main():
    out = dict(script="scripts/paper3_equal_partition.py",
               sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), eps=EPS, rows=[])
    for sizes in PARTS:
        for q in QS:
            K = q / EPS
            part = Partition(list(sizes))
            seen = []
            for seed in (1, 2, 3):
                rng = np.random.default_rng(seed)
                v = np.r_[rng.uniform(-2, 2, len(sizes)), rng.uniform(-.6, 1.4, len(sizes))]
                v = solve_ivp(lambda _t, u: part.rhs(u, K, EPS), (0, 5000), v, method="DOP853", rtol=1e-10,
                              atol=1e-12, max_step=.05).y[:, -1]
                r = hc.describe(part, v, K, EPS)
                if r is None:
                    out["rows"].append(dict(sizes=sizes, q=q, seed=seed, status="no_orbit"))
                    print(sizes, q, seed, "no orbit", flush=True)
                    continue
                d = r[0]
                key = round(d["period"], 4)
                if key in seen:
                    continue
                seen.append(key)
                a = part.audit(np.r_[d["orbit_point"], d["period"]], K, EPS, 2e-12)
                o = hc.orbit(part, np.asarray(d["orbit_point"]), K, EPS)
                fire = np.asarray(o[3]) if o is not None else None
                offs = sorted(((fire - fire[0]) % d["period"] / d["period"]).round(4).tolist()) if fire is not None else None
                lam = np.asarray(d["internal"])
                row = dict(sizes=sizes, q=q, seed=seed, period=d["period"], internal=d["internal"],
                           max_internal=float(lam.max()), n_unstable=int(np.sum(lam > 0)),
                           collective_max=a["collective_max"], valid=a["valid"], return_error=a["return_error"],
                           firing_offsets=offs)
                out["rows"].append(row)
                print(sizes, q, seed, f"T {d['period']:.4f}", "internal", np.round(lam, 4).tolist(),
                      f"coll {a['collective_max']:.4f}", "valid", a["valid"], "offsets", offs, flush=True)
    (ROOT / "analysis/heteroclinic_n50/equal_partition.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
