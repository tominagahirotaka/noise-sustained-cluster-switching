#!/usr/bin/env python3
"""Transverse stability of nearly balanced three-cluster orbits (and three clusters plus a singleton) for several N
at eps = 0.25 (exploratory, 2026-10-23): why sustained switching was seen only for N = 50.

For each partition and eps K: reduced system from random centroids (seeds 1-3, as itinerancy_n50.py reduced),
transient 5000, shooting and audit (heteroclinic_n50.describe): period and internal (transverse) exponents.
-> analysis/heteroclinic_n50/balanced_partitions.json

  --grid : instead, (17,17,16) and (17,16,16,1) over eps in {0.2, ..., 0.4} and eps K in {-0.4, ..., -1.2}
           (seed 1 only) -> analysis/heteroclinic_n50/n50_grid.json
"""
from pathlib import Path
import json
import sys

import numpy as np
from scipy.integrate import solve_ivp

sys.path.insert(0, str(Path(__file__).resolve().parent))
from eps_three_cluster import Partition, ROOT  # noqa: E402
import heteroclinic_n50 as hc  # noqa: E402

PARTS = [(17, 17, 16), (17, 16, 16, 1), (34, 33, 33), (34, 34, 32), (33, 33, 33, 1), (67, 67, 66), (67, 66, 66, 1),
         (134, 133, 133), (134, 134, 132), (334, 333, 333), (334, 334, 332)]
QS = (-.9, -1., -1.2)


def main():
    grid = "--grid" in sys.argv
    out = []
    cases = ([(sz, ee, qq) for sz in [(17, 17, 16), (17, 16, 16, 1)] for ee in (.2, .25, .3, .35, .4)
              for qq in (-.4, -.5, -.6, -.7, -.8, -.9, -1., -1.2)] if grid
             else [(sz, .25, qq) for sz in PARTS for qq in QS])
    for sizes, e, q in cases:
        if True:
            K = q / e
            part = Partition(list(sizes))
            seen = []
            for seed in ((1,) if grid else (1, 2, 3)):
                rng = np.random.default_rng(seed)
                v = np.r_[rng.uniform(-2, 2, len(sizes)), rng.uniform(-.6, 1.4, len(sizes))]
                v = solve_ivp(lambda _t, u: part.rhs(u, K, e), (0, 5000), v, method="DOP853", rtol=1e-10,
                              atol=1e-12, max_step=.05).y[:, -1]
                d = hc.describe(part, v, K, e)
                if d is None:
                    out.append(dict(sizes=sizes, eps=e, q=q, seed=seed, status="no_orbit"))
                    print(sizes, q, seed, "no orbit", flush=True)
                    continue
                d = d[0]
                key = round(d["period"], 4)
                if key in seen:
                    continue
                seen.append(key)
                lam = np.asarray(d["internal"])
                big = np.asarray(sizes) > 1
                out.append(dict(sizes=sizes, eps=e, q=q, seed=seed, period=d["period"], internal=d["internal"],
                                n_unstable_big=int(np.sum(lam[big] > 0)), max_big=float(lam[big].max())))
                print(sizes, e, q, seed, f"T {d['period']:.4f}", "internal", np.round(lam, 4).tolist(),
                      "unstable big", int(np.sum(lam[big] > 0)), flush=True)
    name = "n50_grid.json" if grid else "balanced_partitions.json"
    (ROOT / "analysis/heteroclinic_n50" / name).write_text(json.dumps(out))


if __name__ == "__main__":
    main()
