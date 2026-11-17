#!/usr/bin/env python3
"""What do the runs excluded from Supplementary Material S10.2 do over long times? (exploratory, 2026-10-23)

Question (author): the flagged runs (count depends on the threshold or changes over the final checkpoints), the runs
held on internally unstable periodic orbits, and the runs whose reduced dynamics has a transversely unstable cluster
may be chaotic, quasi-periodic, transient, or switch elements between clusters.

  select : S10.2 grid (N = 50, 100, 200, 400, 1000; eps in {0.2, ..., 0.4}; eps K in {-0.4, ..., -1.0, -1.2};
           seeds 5001-5050) ->  analysis/switching_check/selected.json, one entry per run with its saved final state:
             kind "flagged"   : thresholds_agree or tail_constant false (analysis/eps_cluster_ensemble/runs_*.json)
             kind "unstable"  : unflagged, periodic orbit internally or collectively unstable
                                (analysis/low_cluster_check/chunk_*.json, analysis/many_cluster_check/summary.json)
             kind "transverse": unflagged, no periodic orbit, excluded in S10.2 (transversely unstable cluster or
                                count changed under the perturbation)
  run    : entry --index: Gaussian noise of sd 1e-6 on all 2N coordinates (PCG64 seed = index), full system RK4
           (h = 0.005) to --tmax. A second copy started 1e-8 away gives the largest Lyapunov exponent of the full
           system (renormalized every 5 time units; running estimate every 1000). Recorded:
             every 10 time units: cluster labels at threshold 1e-3 (max-norm, connected components), and counts at
                                  thresholds 1e-2, 1e-4, 1e-6;
             every time unit    : largest within-cluster spread of each cluster of the saved partition (threshold 1e-4);
             upward x = 0 crossings of every element (firing times, linear interpolation).
           -> analysis/switching_check/sw_<index>.npz and sw_<index>.json (summary)

  python3 scripts/switching_check.py select
  python3 scripts/switching_check.py run --index 0 --tmax 50000        (remote host: numpy, scipy)
"""
from pathlib import Path
import argparse
import glob
import json

import numpy as np
from scipy.sparse.csgraph import connected_components

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "analysis/switching_check"
TAGS = {50: "N50", 100: "N100final", 200: "N200", 400: "N400", 1000: "N1000"}
EPS = {.2, .25, .3, .35, .4}
QS = {-.4, -.5, -.6, -.7, -.8, -.9, -1., -1.2}
B, G = .7, .5
D0, DT = 1e-8, 5.


def labels(z, n, th):
    d = np.maximum(abs(z[:n, None] - z[None, :n]), abs(z[n:, None] - z[None, n:]))
    return connected_components(d < th, directed=False)


def select():
    key = lambda n, e, q, s: (n, round(e, 3), round(q, 3), s)  # noqa: E731
    status = {}
    for f in glob.glob(str(ROOT / "analysis/low_cluster_check/chunk_*.json")):
        for x in json.loads(Path(f).read_text()):
            if x["unflagged"]:
                status[key(x["N"], x["eps"], x["eps"] * x["K"], x["seed"])] = x
    pert = {json.loads(Path(f).read_text())["low_index"]: json.loads(Path(f).read_text())
            for f in glob.glob(str(ROOT / "analysis/low_cluster_check/perturb_0*.json"))}
    many = {key(x["N"], x["eps"], x["epsK"], x["seed"]): x
            for x in json.loads((ROOT / "analysis/many_cluster_check/summary.json").read_text()) if x["unflagged"]}
    sel = []
    for n, tag in TAGS.items():
        for r in json.loads((ROOT / f"analysis/eps_cluster_ensemble/runs_{tag}.json").read_text()):
            if not (round(r["eps"], 3) in EPS and round(r["eps"] * r["K"], 3) in QS and 5001 <= r["seed"] < 5051):
                continue
            k = key(n, r["eps"], r["eps"] * r["K"], r["seed"])
            kind = None
            if not (r["thresholds_agree"] and r["tail_constant"]):
                kind = "flagged"
            elif r["clusters"] < 6:
                x = status[k]
                if x["status"] == "ok" and not x["stable"]:
                    kind = "unstable"
                elif x["status"] == "no_recurrence":
                    p = pert[x["index"]]
                    if not (all(c["clusters"] == x["clusters"] for c in p["perturbed"])
                            and (x["internal_exp_max"] is None or x["internal_exp_max"] < 0)):
                        kind = "transverse"
            elif k in many:
                x = many[k]
                if x["audit"] == "unstable":
                    kind = "unstable"
                elif x["audit"] == "no_recurrence" and not x["kept"]:
                    kind = "transverse"
            if kind:
                sel.append(dict(kind=kind, N=n, eps=r["eps"], K=r["K"], seed=r["seed"], clusters=r["clusters"],
                                sizes=r["sizes"], checkpoint_tail=r["checkpoint_tail"],
                                thresholds_agree=r["thresholds_agree"], final_state=r["final_state"]))
    sel.sort(key=lambda s: -s["N"])  # largest first, so that the long jobs start early
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "selected.json").write_text(json.dumps(sel))
    import collections
    print(len(sel), collections.Counter((s["kind"], s["N"]) for s in sel))


def run(index, tmax):
    r = json.loads((OUT / "selected.json").read_text())[index]
    e, K = r["eps"], r["K"]
    z0 = np.asarray(r["final_state"], float)
    n = z0.size // 2
    _, lab0 = labels(z0, n, 1e-4)
    m0 = lab0.max() + 1
    rng = np.random.default_rng(index)
    zA = z0 + rng.normal(0, 1e-6, z0.size)
    g = rng.standard_normal(z0.size)
    zB = zA + D0 * g / np.linalg.norm(g)

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
    per = int(round(1 / h))
    nunits = int(tmax)
    lab_rec = np.zeros((nunits // 10, n), np.int16)
    cnt_rec = np.zeros((nunits // 10, 4), np.int16)
    spread = np.zeros((nunits, m0))
    fire_i, fire_t = [], []
    lsum, lyap = 0., []
    xprev = zA[:n].copy()
    t = 0.
    for u in range(nunits):
        for _ in range(per):
            zA, zB = rk(zA, h), rk(zB, h)
            t += h
            up = np.flatnonzero((xprev < 0) & (zA[:n] >= 0))
            if up.size:
                fire_i.append(up)
                fire_t.append(t - h * zA[up] / (zA[up] - xprev[up]))
            xprev = zA[:n].copy()
        for c in range(m0):
            s = lab0 == c
            spread[u, c] = max(np.ptp(zA[:n][s]), np.ptp(zA[n:][s]))
        if (u + 1) % int(DT) == 0:
            dd = zB - zA
            nd = np.linalg.norm(dd)
            lsum += np.log(nd / D0)
            zB = zA + D0 * dd / nd
        if (u + 1) % 1000 == 0:
            lyap.append([u + 1, lsum / (u + 1)])
        if (u + 1) % 10 == 0:
            k = (u + 1) // 10 - 1
            _, lab_rec[k] = labels(zA, n, 1e-3)
            cnt_rec[k] = [labels(zA, n, th)[0] for th in (1e-2, 1e-3, 1e-4, 1e-6)]
    fi = np.concatenate(fire_i) if fire_i else np.zeros(0, int)
    ft = np.concatenate(fire_t) if fire_t else np.zeros(0)
    OUT.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(OUT / f"sw_{index:04d}.npz", labels=lab_rec, counts=cnt_rec, spread=spread.astype(np.float32),
                        fire_i=fi.astype(np.int16), fire_t=ft, lab0=lab0, lyap=np.asarray(lyap), final=zA)
    summ = {k: r[k] for k in r if k != "final_state"}
    summ.update(index=index, tmax=tmax, lyap_full=lsum / nunits,
                counts_tail=cnt_rec[-5:, 2].tolist(), final_sizes=sorted(np.bincount(labels(zA, n, 1e-4)[1]).tolist(),
                                                                         reverse=True))
    (OUT / f"sw_{index:04d}.json").write_text(json.dumps(summ))
    print(index, r["kind"], r["N"], e, round(e * K, 2), r["seed"], r["clusters"], "lyap", round(summ["lyap_full"], 5),
          "final", summ["final_sizes"][:8], flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["select", "run"])
    ap.add_argument("--index", type=int, default=0)
    ap.add_argument("--tmax", type=float, default=50000.)
    a = ap.parse_args()
    if a.mode == "select":
        select()
    else:
        run(a.index, a.tmax)


if __name__ == "__main__":
    main()
