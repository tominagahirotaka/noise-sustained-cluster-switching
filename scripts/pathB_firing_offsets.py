#!/usr/bin/env python3
"""Sign of the firing offsets of the elements that leave together in path B (runs 396, 398; exploratory, 2026-10-28).

Path B (RESULTS_NOTES 2026-10-27): two (or three) elements leave the same cluster at almost the same time, one of
them returns. Hypothesis to test: the advanced one (fires before the rest of its cluster) leaves for good and joins
another cluster, the retarded one returns. Only the saved states and the regenerated noise are used (no new run):
for every "two out" switch of moves_<index>.json at --th (period-min labels, hold 200), the trajectory is rebuilt
from the saved state about --before time units before the departure to --after time units after it (RK4, h = 0.005,
noise of switching_renoise.run drawn again in the same order as rerun_moves.py; the end of every 10-unit interval is
compared with the next saved state). Firing times: upward x = 0, linear interpolation within the step.
  - reference: per cycle the median firing time of the rest of the former cluster (its members at the period just
    before the departure, period-min distance < --th, minus every element that left it in this switch); the spread
    (max - min) of the rest is recorded as a check;
  - offset of each departing element: its firing time nearest to the reference, minus the reference, in (-T/2, T/2];
    negative = advanced (fires before the rest), positive = retarded (as heteroclinic_n50.py);
  - ambiguous cycles: the element has no firing or more than one within half a period of the reference;
  - sign: the sign of the offset at the first cycle where |offset| > --first (the growth out of the cluster), and
    every later change of sign while |offset| > --first;
  - fate (the same period-min distances as moves_<index>.json, looked at only after the element has left, v2):
    back to the former cluster, joined another
    cluster (period-min distance < --th at the end of the follow window), or still single.
  -> analysis/switching_check/rerun_states/pathB_offsets_<index>.json

  python3 scripts/pathB_firing_offsets.py --index 396
"""
from pathlib import Path
import argparse
import hashlib
import json

import numpy as np
from scipy.sparse.csgraph import connected_components

ROOT = Path(__file__).resolve().parents[1]
SC = ROOT / "analysis/switching_check"
RS = SC / "rerun_states"
B, G = .7, .5


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", type=int, required=True)
    ap.add_argument("--th", type=float, default=1e-3)
    ap.add_argument("--before", type=float, default=240.)
    ap.add_argument("--after", type=float, default=1500.)
    ap.add_argument("--first", type=float, default=1e-3)
    a = ap.parse_args()
    r = json.loads((SC / "selected.json").read_text())[a.index]
    e, K = r["eps"], r["K"]
    S = np.load(RS / f"rn_{a.index:04d}_states_z.npy")
    n = S.shape[1] // 2
    mv = json.loads((RS / f"moves_{a.index:04d}.json").read_text())
    T, w = mv["period"], mv["period_units"]
    Dm = np.load(RS / f"Dminmax_{a.index:04d}.npz")["Dmin"]

    def f(v):
        x, y = v[:n], v[n:]
        return np.r_[(x - x ** 3 / 3 - y) / e + K * (x.mean() - x), e * (x + B - G * y)]

    def rk(z, h):
        k1 = f(z)
        k2 = f(z + h / 2 * k1)
        k3 = f(z + h / 2 * k2)
        k4 = f(z + h * k3)
        return z + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)

    # the noise vectors in the order of switching_renoise.run (two initial draws, then one per saved state)
    rng = np.random.default_rng(10_000 + a.index)
    rng.normal(0, 1e-6, 2 * n)
    rng.standard_normal(2 * n)
    noise = rng.normal(0, 1e-10, (len(S), 2 * n))

    res = next(x for x in mv["results"] if x["th"] == a.th)
    out = dict(index=a.index, th=a.th, period=T, before=a.before, after=a.after, first=a.first,
               script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
               sign="offset = firing time - median firing time of the rest of the former cluster; < 0 advanced",
               events=[])
    for s in res["switches"]:
        for x in s.get("two_out", []):
            cl, c0 = x["cluster"], x["t_from"] // w
            leavers = [el for el, fr, to in s["moves"] if fr == cl and to == "free"]
            lab = connected_components(Dm[c0 - 1] < a.th, directed=False)[1]
            former = np.nonzero(lab == lab[x["pair"][0]])[0]
            rest = [int(i) for i in former if i not in leavers]
            k0 = max(0, int((x["t_from"] - a.before) // 10))
            k1 = min(len(S) - 1, int((x["t_from"] + a.after) // 10))
            fires = [[] for _ in range(n)]
            mism = []
            for k in range(k0, k1):
                z = S[k] + noise[k]
                t = 10. * k
                for _ in range(2000):
                    zn = rk(z, .005)
                    up = np.nonzero((z[:n] < 0) & (zn[:n] >= 0))[0]
                    for i in up:
                        fires[i].append(t + .005 * (-z[i]) / (zn[i] - z[i]))
                    z, t = zn, t + .005
                mism.append(float(np.max(abs(z - S[k + 1]))))
            ref_el = rest[0]
            cyc = []
            for tr in fires[ref_el][1:-1]:
                fr_ = [min(fires[i], key=lambda u: abs(u - tr)) for i in rest]
                ref = float(np.median(fr_))
                row = dict(t=round(ref, 3), rest_spread=float(np.ptp(fr_)), off={}, ambiguous=[])
                for el in leavers:
                    near = [u for u in fires[el] if abs(u - ref) < T / 2]
                    if len(near) != 1:
                        row["ambiguous"].append(int(el))
                    if near:
                        u = min(near, key=lambda u: abs(u - ref))
                        row["off"][str(el)] = float(u - ref)
                cyc.append(row)
            ev = dict(t_switch=s["t"], kind=s["kind"], cluster=cl, leavers=[int(el) for el in leavers],
                      pair=x["pair"], former_size=int(len(former)), rest=rest, t_from=x["t_from"],
                      reconstruction_mismatch=max(mism), exact_intervals=int(sum(m == 0 for m in mism)),
                      intervals=len(mism), max_rest_spread=max(c["rest_spread"] for c in cyc),
                      ambiguous_cycles={str(el): [c["t"] for c in cyc if el in c["ambiguous"]] for el in leavers},
                      cycles=cyc, elements={})
            for el in leavers:
                o = np.array([c["off"].get(str(el), np.nan) for c in cyc])
                tt = np.array([c["t"] for c in cyc])
                big = np.nonzero(abs(o) > a.first)[0]
                d = dict(first_cycle_above=None, first_sign=None, sign_changes=[])
                if len(big):
                    j = big[0]
                    d.update(first_cycle_above=float(tt[j]), first_sign="advanced" if o[j] < 0 else "retarded",
                             first_offset=float(o[j]), offset_before=o[max(0, j - 3):j].tolist())
                    sg = np.sign(o[big])
                    d["sign_changes"] = [float(tt[big[q]]) for q in range(1, len(big)) if sg[q] != sg[q - 1]]
                    d["fraction_advanced_above"] = float(np.mean(o[big] < 0))
                d["log10_abs_offset"] = np.round(np.log10(np.maximum(abs(o), 1e-30)), 2).tolist()
                d["offset"] = np.round(o, 5).tolist()
                ev["elements"][str(el)] = d
            # spread of the rest near the departure (from 3 cycles before the first |offset| > --first to 20 after)
            t1 = min(ev["elements"][str(el)]["first_cycle_above"] or np.inf for el in leavers)
            near_dep = [c["rest_spread"] for c in cyc if t1 - 3.5 * T < c["t"] < t1 + 20.5 * T]
            ev["max_rest_spread_near_departure"] = max(near_dep) if near_dep else None
            ev["all_moves"] = s["moves"]
            # where the other elements fire relative to the rest, 3 cycles before the departure (landmarks for the
            # offsets: the clusters and singletons the leavers pass)
            cb = [c for c in cyc if c["t"] < t1 - 2.5 * T]
            if cb:
                tr = cb[-1]["t"]
                lm = {}
                for i in range(n):
                    if i in former:
                        continue
                    u = min(fires[i], key=lambda u: abs(u - tr))
                    lm[int(i)] = round(float((u - tr + T / 2) % T - T / 2), 3)
                ev["landmarks_before"] = dict(t=tr, offsets=lm)
            # fate, from the same period-min distances as moves_<index>.json, only after the element has left
            # (v1 of this script looked from t_from on and called elements "back" before they had left)
            fate = {}
            for el in leavers:
                cend = min(len(Dm) - 1, c0 + int(a.after / w) - 1)
                fc = ev["elements"][str(el)]["first_cycle_above"]
                cl_ = int(fc // w) if fc is not None else c0
                left = [c for c in range(cl_, cend + 1) if Dm[c, el, rest].min() >= a.th]
                back = [c for c in range(left[0] + 1, cend + 1) if Dm[c, el, rest].min() < a.th] if left else []
                if back:
                    fate[str(el)] = dict(fate="back", t=int(back[0] * w))
                    continue
                labe = connected_components(Dm[cend] < a.th, directed=False)[1]
                mates = [int(i) for i in np.nonzero(labe == labe[el])[0] if i != el]
                fate[str(el)] = dict(fate="joined" if len(mates) >= 2 else ("pair" if mates else "single"),
                                     mates_at_end=mates, t_end=int(cend * w),
                                     mates_from_former=[i for i in mates if i in former])
            ev["fate"] = fate
            ev["fate_in_moves_json"] = x["back_to_former_cluster"]
            out["events"].append(ev)
            print(a.index, s["t"], s["kind"], "leavers", leavers, "rest", len(rest), "mismatch", max(mism),
                  {el: (ev["elements"][str(el)]["first_sign"], ev["elements"][str(el)]["sign_changes"][:5],
                        fate[str(el)]["fate"]) for el in leavers},
                  "ambiguous", {k_: len(v) for k_, v in ev["ambiguous_cycles"].items()}, flush=True)
    (RS / f"pathB_offsets_{a.index:04d}.json").write_text(json.dumps(out))


if __name__ == "__main__":
    main()
