#!/usr/bin/env python3
"""Element moves in the switches of a long N = 50 run, and the two-element departures (exploratory, 2026-10-27).

The trajectory is rebuilt as in rerun_periodmin.py (from rn_<index>_states_z.npy, with the noise drawn again in the
same order; the end of every 10-unit interval is compared with the next saved state). Per period of element 0 the
period-min and period-max pairwise distances (max(|dx|, |dy|), min / max over the period) are saved (float32,
Dminmax_<index>.npz) and the
switches (labels at --th, residences of >= --hold time units, n50_web_analysis.persistent) are listed with their moves.
For every switch in which two elements leave the same cluster ("two out"), the two are followed from the switch on:
per period their mutual period-min and period-max distances, and the period-min distance of each from its former
cluster (the rest of it) and from every other cluster, for --follow time units. Both mutual distances below --th: the
two travel together as a 2-element cluster through the whole period; only the min below: they come close once per
period; both above: separate. (Clusters built from period-min distances need not be transitive: i-j and j-k close at
different times of a period does not make i, j, k one cluster.)
  -> analysis/switching_check/rerun_states/moves_<index>.json, Dminmax_<index>.npz

  python3 scripts/rerun_moves.py --index 396
"""
from collections import Counter
from pathlib import Path
import argparse
import hashlib
import json
import sys

import numpy as np
from scipy.sparse.csgraph import connected_components

sys.path.insert(0, str(Path(__file__).resolve().parent))
from renoise_summary import residences  # noqa: E402
from n50_web_analysis import persistent  # noqa: E402
from first_departure_analysis import kind  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
SC = ROOT / "analysis/switching_check"
RS = SC / "rerun_states"
B, G = .7, .5


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", type=int, required=True)
    ap.add_argument("--th", type=float, nargs="+", default=[1e-3, 1e-4])
    ap.add_argument("--hold", type=float, default=200.)
    ap.add_argument("--follow", type=float, default=1500.)
    a = ap.parse_args()
    r = json.loads((SC / "selected.json").read_text())[a.index]
    e, K = r["eps"], r["K"]
    S = np.load(RS / f"rn_{a.index:04d}_states_z.npy")
    d = np.load(RS / f"rn_{a.index:04d}_states.npz")
    n = S.shape[1] // 2

    def f(v):
        x, y = v[:n], v[n:]
        return np.r_[(x - x ** 3 / 3 - y) / e + K * (x.mean() - x), e * (x + B - G * y)]

    def rk(z, h):
        k1 = f(z)
        k2 = f(z + h / 2 * k1)
        k3 = f(z + h / 2 * k2)
        k4 = f(z + h * k3)
        return z + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)

    rng = np.random.default_rng(10_000 + a.index)  # same order as switching_renoise.run
    rng.normal(0, 1e-6, 2 * n)
    rng.standard_normal(2 * n)
    T = float(np.median(np.diff(d["fire0"])))
    w = int(round(T))
    nc = len(S) * 10 // w
    Dm = np.full((nc, n, n), np.inf, np.float32)
    Dx = np.zeros((nc, n, n), np.float32)  # period-max, same periods and same trajectory
    mismatch = []
    u_all = 0
    for k, z in enumerate(S):
        z = z + rng.normal(0, 1e-10, 2 * n)
        for _ in range(10):
            for _ in range(200):
                z = rk(z, .005)
            c = u_all // w
            if c < nc:
                dd = np.maximum(abs(z[:n, None] - z[None, :n]), abs(z[n:, None] - z[None, n:]))
                Dm[c] = np.minimum(Dm[c], dd)
                Dx[c] = np.maximum(Dx[c], dd)
            u_all += 1
        if k + 1 < len(S):
            mismatch.append(float(np.max(abs(z - S[k + 1]))))
    mismatch = np.asarray(mismatch)
    np.savez_compressed(RS / f"Dminmax_{a.index:04d}.npz", Dmin=Dm, Dmax=Dx, period_units=w)
    out = dict(index=a.index, eps=e, q=e * K, period=T, period_units=w,
               script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
               input=f"rn_{a.index:04d}_states_z.npy", noise_seed=10_000 + a.index,
               distance="max(|dx|, |dy|) of two elements at every time unit; Dmin/Dmax: min/max over each period",
               reconstruction_mismatch=dict(max=float(mismatch.max()), exact=int((mismatch == 0).sum()),
                                            intervals=len(mismatch)), results=[])
    print(a.index, "mismatch", out["reconstruction_mismatch"], flush=True)
    for th in a.th:
        lab = np.array([connected_components(Dm[c] < th, directed=False)[1] for c in range(nc)])
        res = residences(lab, hold=10 * max(1, int(round(a.hold / w))))
        sw = persistent(res) if len(res) > 1 else []
        rows = []
        for s in sw:
            row = dict(t=round(s["t"] / 10 * w), t_prev_end=round(s["t_prev_end"] / 10 * w), kind=kind(s),
                       moves=[[int(el), str(x), str(y)] for el, x, y in s["moves"]],
                       sizes_before=s["sizes_before"], sizes_after=s["sizes_after"])
            outs = Counter(x for el, x, y in s["moves"] if y == "free" and x != "free")
            for cl, cnt in outs.items():
                if cnt < 2:
                    continue
                pair = [el for el, x, y in s["moves"] if x == cl and y == "free"][:2]
                rest = [m for m in s["members_before"][cl] if m not in pair]
                c0 = int(s["t_prev_end"] / 10)
                c1 = min(nc, c0 + int(a.follow / w))
                mutual = [float(Dm[c, pair[0], pair[1]]) for c in range(c0, c1)]
                mutual_max = [float(Dx[c, pair[0], pair[1]]) for c in range(c0, c1)]
                others = {str(cl2): sorted(m) for cl2, m in s["members_before"].items() if cl2 != cl and len(m) >= 2}
                to_others = {k_: [[float(Dm[c, p, m].min()) for p in pair] for c in range(c0, c1)]
                             for k_, m in others.items()}
                to_rest = [[float(Dm[c, p, rest].min()) for p in pair] for c in range(c0, c1)]
                back = [next((round((c0 + j) * w) for j, v in enumerate(to_rest) if v[i] < th
                              and j > 0 and to_rest[j - 1][i] >= th), None) for i in range(2)]
                row.setdefault("two_out", []).append(dict(
                    cluster=str(cl), pair=pair, t_from=round(c0 * w),
                    log10_mutual=np.round(np.log10(np.maximum(mutual, 1e-30)), 2).tolist(),
                    log10_mutual_max=np.round(np.log10(np.maximum(mutual_max, 1e-30)), 2).tolist(),
                    log10_to_other_clusters={k_: np.round(np.log10(np.maximum(v, 1e-30)), 2).tolist()
                                             for k_, v in to_others.items()},
                    log10_to_rest=np.round(np.log10(np.maximum(to_rest, 1e-30)), 2).tolist(),
                    back_to_former_cluster=back,
                    together_fraction_min=float(np.mean(np.asarray(mutual) < th)),
                    together_fraction_max=float(np.mean(np.asarray(mutual_max) < th))))
            rows.append(row)
        out["results"].append(dict(th=th, hold=a.hold, kinds=dict(Counter(x["kind"] for x in rows)), switches=rows))
        print(a.index, th, dict(Counter(x["kind"] for x in rows)), flush=True)
    (RS / f"moves_{a.index:04d}.json").write_text(json.dumps(out))


if __name__ == "__main__":
    main()
