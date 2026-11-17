#!/usr/bin/env python3
"""Count the switches of a long N = 50 run in two ways on the same trajectory (exploratory, 2026-10-27).

Input: a run of switching_renoise.py --states (rn_<index>_states.npz: the labels as B2 used them; rn_<index>_states_z.npy:
the full state before the noise every 10 time units, float64). The trajectory between two saved states is rebuilt by
integrating 10 time units from each saved state (RK4, h = 0.005, as switching_renoise.py) and keeping every time unit;
the noise added after each saved state is drawn again from the same generator in the same order, so the rebuilt
trajectory is the original one up to rounding; the end of every interval is compared with the next saved state
(reconstruction_mismatch).
  - B2: the saved labels (instantaneous, every 10 time units, threshold 1e-4), residences (hold), persistent;
  - period-min: per period of element 0 (median of its firing intervals) the minimum over the period of the pairwise
    distance max(|dx|, |dy|), labels at --th, residences, persistent (as first_departure_analysis.py).
Every switch is classified with first_departure_analysis.kind. Event correspondence: for every B2 switch (hold 200),
the period-min switches between the end of the previous B2 residence and the B2 switch (+- --slack) are listed.
  -> analysis/switching_check/rerun_states/periodmin_<index>.json

  python3 scripts/rerun_periodmin.py --index 358
"""
from collections import Counter
from pathlib import Path
import argparse
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
    ap.add_argument("--th", type=float, nargs="+", default=[1e-4, 1e-3])
    ap.add_argument("--hold", type=float, nargs="+", default=[100., 200.])
    ap.add_argument("--slack", type=float, default=50.)
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

    # the noise of switching_renoise.run drawn again in the same order (initial noise, direction of the second copy,
    # then one draw every 10 time units added right after the state is saved); these runs used the default 1e-10
    rng = np.random.default_rng(10_000 + a.index)
    rng.normal(0, 1e-6, 2 * n)
    rng.standard_normal(2 * n)
    dense = np.zeros((len(S) * 10, 2 * n), np.float32)
    mismatch = []
    for k, z in enumerate(S):
        z = z + rng.normal(0, 1e-10, 2 * n)
        for u in range(10):
            for _ in range(200):
                z = rk(z, .005)
            dense[10 * k + u] = z  # time 10 (k + 1) + u + 1
        if k + 1 < len(S):
            mismatch.append(float(np.max(abs(z - S[k + 1]))))
    mismatch = np.asarray(mismatch)
    T = float(np.median(np.diff(d["fire0"])))
    w = int(round(T))
    nc = len(dense) // w
    P = dense[:nc * w].reshape(nc, w, 2 * n)
    Dm = np.empty((nc, n, n), np.float32)
    for c in range(nc):  # one period at a time (all at once does not fit in 1G)
        Q = P[c].astype(float)
        Dm[c] = np.maximum(abs(Q[:, :n, None] - Q[:, None, :n]), abs(Q[:, n:, None] - Q[:, None, n:])).min(axis=0)
    out = dict(index=a.index, eps=e, q=e * K, period=T, results=[],
               reconstruction_mismatch=dict(max=float(mismatch.max()), median=float(np.median(mismatch)),
                                            n_above_1e_12=int((mismatch > 1e-12).sum()), intervals=len(mismatch)))
    print(a.index, "reconstruction mismatch", out["reconstruction_mismatch"], flush=True)
    for th in a.th:
        lab = np.array([connected_components(Dm[c] < th, directed=False)[1] for c in range(nc)])
        for hold in a.hold:
            res = residences(lab, hold=10 * max(1, int(round(hold / (w))))) if len(lab) else []
            sw = persistent(res) if len(res) > 1 else []
            row = dict(method="period-min", th=th, hold=hold, kinds=dict(Counter(kind(s) for s in sw)),
                       t_switch=[round(s["t"] / 10 * w) for s in sw], kinds_list=[kind(s) for s in sw])
            out["results"].append(row)
            print(a.index, row["method"], th, hold, row["kinds"], flush=True)
    for hold in a.hold:
        res = residences(d["labels"], hold=hold)
        sw = persistent(res) if len(res) > 1 else []
        row = dict(method="B2", th=1e-4, hold=hold, kinds=dict(Counter(kind(s) for s in sw)),
                   t_switch=[s["t"] for s in sw])
        out["results"].append(row)
        print(a.index, "B2", 1e-4, hold, row["kinds"], flush=True)
    # which period-min switches fall inside each B2 switch (from the end of the previous B2 residence to the start of
    # the next, with --slack on both sides): how many does one B2 event split into
    res = residences(d["labels"], hold=200.)
    swb = persistent(res) if len(res) > 1 else []
    for x in out["results"]:
        if x["method"] != "period-min":
            continue
        x["per_B2_event"] = []
        for s_ in swb:
            inside = [kd for t_, kd in zip(x["t_switch"], x["kinds_list"])
                      if s_["t_prev_end"] - a.slack <= t_ <= s_["t"] + a.slack]
            x["per_B2_event"].append(dict(t=s_["t"], b2_kind=kind(s_), periodmin=inside))
    (RS / f"periodmin_{a.index:04d}.json").write_text(json.dumps(out))


if __name__ == "__main__":
    main()
