#!/usr/bin/env python3
"""How many elements leave the unstable cluster in the first switch of the full N = 50 system? (exploratory, 2026-10-27)

The reduced (16,16,16,1,1) node connects to a permuted copy of itself when one element leaves the transversely
unstable cluster U and the singleton firing next joins the rest of U (heteroclinic_n50.py, pair_merges). In the
full system at eps = 0.25 most switches move several elements at once (RESULTS_NOTES 2026-10-27). Here the full system
is started on the node (node_<--node>.json, orbit_point, every element on its cluster) and perturbed:
  --mode noise  : Gaussian sd --sigma0 on all 2N coordinates at t = 0, then sd --noise every 10 time units
                  (as switching_renoise.py), random generator seeded by --rep; --noise 0 is the control with
                  the initial perturbation only (name fd_<node>_init<sigma0>_n0_r<rep>);
  --mode single : one element of U displaced in x by +-1e-8 (sign --sign; the unstable internal direction grows out of it),
                  no noise (control: should reproduce the reduced swap).
RK4, h = 0.005 (as switching_renoise.py), for --tmax; the state is saved every --every time units (float32).
  -> analysis/heteroclinic_n50/firstdep/fd_<node>_<mode>_n<noise>_r<rep>.npz

  python3 scripts/first_departure_n50.py --node e0.25_16-16-16-1-1_q-0.800_s2_pm --mode noise --rep 1
"""
from pathlib import Path
import argparse
import json

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
H = ROOT / "analysis/heteroclinic_n50"
B, G = .7, .5


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--node", required=True)
    ap.add_argument("--mode", choices=["noise", "single"], default="noise")
    ap.add_argument("--rep", type=int, default=1)
    ap.add_argument("--sigma0", type=float, default=1e-10)
    ap.add_argument("--noise", type=float, default=1e-10)
    ap.add_argument("--sign", type=int, choices=[-1, 1], default=-1)
    ap.add_argument("--tmax", type=float, default=3000.)
    ap.add_argument("--every", type=int, default=2)
    a = ap.parse_args()
    d = json.loads((H / f"node_{a.node}.json").read_text())["node"]
    e, K, sizes, U = d["eps"], d["K"], d["sizes"], d["unstable"]
    m = len(sizes)
    zc = np.asarray(d["orbit_point"])
    member = np.repeat(np.arange(m), sizes)
    n = member.size
    z = np.r_[zc[:m][member], zc[m:][member]]
    if a.mode == "noise":
        rng = np.random.default_rng(a.rep)
        z = z + rng.normal(0, a.sigma0, z.size)
    else:
        rng = None
        i0 = int(np.nonzero(member == U)[0][0])
        z[i0] += a.sign * 1e-8  # x only; the unstable internal direction grows out of it

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
    nunits = int(a.tmax)
    rec = np.zeros((nunits // a.every + 1, 2 * n), np.float32)
    rec[0] = z
    for u in range(nunits):
        for _ in range(200):
            z = rk(z, h)
        if a.mode == "noise" and (u + 1) % 10 == 0:
            z = z + rng.normal(0, a.noise, z.size)
        if (u + 1) % a.every == 0:
            rec[(u + 1) // a.every] = z
    out = H / "firstdep"
    out.mkdir(parents=True, exist_ok=True)
    tag = f"fd_{a.node}_{a.mode}" + (f"_n{a.noise:g}_r{a.rep}" if a.mode == "noise" else f"_s{a.sign:+d}")
    if a.mode == "noise" and a.noise == 0:  # initial perturbation only: sigma0 in the name
        tag = f"fd_{a.node}_init{a.sigma0:g}_n0_r{a.rep}"
    np.savez_compressed(out / f"{tag}.npz", z=rec, member=member, U=U, sizes=sizes, eps=e, K=K, every=a.every,
                        period=d["period"], firing_order_from_U=d["firing_order_from_U"], args=json.dumps(vars(a)))
    print(tag, "done", flush=True)


if __name__ == "__main__":
    main()
