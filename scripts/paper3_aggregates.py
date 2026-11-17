#!/usr/bin/env python3
"""Paper 3: re-aggregation of the residence times against ln(1/eta) from saved outputs only (2026-10-29).

No numerical integration. Every number is counted from the saved records with the rules below, fixed before the
results were looked at (paper3/evidence_inventory.md section 6, items 1 and 2). The values recorded earlier in
docs/RESULTS_NOTES.md (0.0169, 0.0223, 0.047, 930) are only compared at the end; they are not targets.

Claim 1 (N = 50; runs 379, 394, 397) -- "switching interval"
  input   analysis/switching_check/rn_<index>[_n<eta>].npz, labels: cluster labels every 10 time units (record k is
          the state at t = 10 (k + 1), before the noise of that time unit), single linkage, max norm, threshold 1e-4
          (as saved by switching_renoise.py; not recomputed).
  config  the set of clusters with >= 2 elements, each as a set of elements (singletons ignored); type = sorted sizes.
  main type: the most frequent type over the window, at eta = 1e-10 (the reference trajectory of each run).
  residence: a maximal run of consecutive records with the same configuration whose type is the main type, lasting
          >= MIN_RES_C1 time units (end - start, end = time of the last record + 10).
  episode: consecutive residences with the same configuration are merged (a return to the same membership after an
          excursion is not a switch).
  switch:  the start of an episode whose configuration differs from that of the previous episode.
  interval: start of episode j+1 minus start of episode j. Window [1000, 20000]: an episode starting before
          T0_C1 is left-censored (its interval is not used); the last episode is right-censored (no interval).
  eta used in the fit: those whose main type equals the main type at eta = 1e-10 (rule stated before counting).
  fit (primary): ordinary least squares of every interval against ln(1/eta), pooled over eta; rate = 1/slope.
  fit (secondary): the same on the per-eta mean interval (one point per eta).

Claim 3 (run 88, N = 400) -- "residence of (137,132,131)"
  input   analysis/switching_check/rn_0088[_n<eta>]_t80000_states_z.npy (full state every 10 time units, t = 10 (k+1)).
  labels  recomputed: single linkage, max norm over (x, y), threshold 1e-3.
  big     sorted sizes of clusters with >= 20 elements.
  residence: the rule of scripts/stageA_compare.py (A2): runs with big == (137,132,131); interruptions shorter than
          BRIDGE_C3 records are bridged; residences of >= MIN_REC_C3 records.
  censoring: a residence that contains the first record is the transient from the common starting point (left-
          censored); one that contains the last record is right-censored. Neither enters the fit.
  eta used in the fit (primary): 1e-14 ... 1e-8 (eta = 1e-6 excluded as in RESULTS_NOTES 2026-10-24, where its only
          (137,132,131) residence was the initial one; this is a recorded prior choice, tested as a sensitivity case).
  fit: OLS of every residence length L against ln(1/eta); rate = 1/slope. Auxiliary: log10 of the median
          within-cluster distance of the 137 cluster at the start of each residence (as in stageA_compare.py).

Claim 1 (a) -- transverse growth rate from saved full states (added 2026-10-29, after the author chose option 1)
  input   analysis/itinerancy_n50/trace_<index>[_n<eta>].npz, z: full state every time unit (eta = 1e-10 for 379,
          386, 393, 394, 397; 1e-14 for 394, 397).
  Dmin    per non-overlapping window of W1A = 25 time units (longer than the periods 20.4-24.0), the minimum over the
          window of the max-norm (x, y) distance of every pair (the per-cycle minimum distance).
  config  single linkage of Dmin < 1e-3, clusters of >= 10 elements, as element sets; residence: the same config in
          >= 8 consecutive windows (200 time units).
  leaver  for consecutive residences r, r+1: an element of a cluster C of r that is not in the cluster of r+1 with the
          largest overlap with C. Only events with exactly one leaver are fitted (others counted).
  g(w)    median over the other members of C of Dmin(leaver, member). Fit: ln g against window centre, OLS, over the
          windows between the minimum of g inside r and the first window after it with g >= 1e-3, restricted to
          log10 g in [log10 g_min + 1, -3.5]; >= 5 points.
  compare with the transverse exponent of the reduced node (heteroclinic_n50/node_*.json): A (17,17,16), B (17,16,16,1)
          for eK = -0.9, -1.0; (16,16,16,1,1) s2 for eK = -1.2.

Claim 1 (b) -- departure-time prediction in the noiseless tree runs (added 2026-10-29)
  input   analysis/noiseless_n50/nl_<index>_t5000.npz (group every time unit at 1e-3, time = 5000 + index + 1;
          spread every 10 time units: log10 of the median tree distance of each element to the other members).
  residence: the rule recorded in scripts/noiseless_switches.py (groups >= 10 as member sets, sampled every 10, same
          sets for >= 200 time units); events with exactly one element leaving the big groups and L >= 1500.
  d0      the leaver's spread at the first spread record after the residence start (log10), and the largest spread in
          its group there (the definition of noiseless_switches.py).
  rate    OLS of the leaver's spread (times ln 10) against time inside the residence, over log10 in [d0 + 1, -3.5].
  prediction without fitting: L_red = ln(10) (-3 - log10 d0) / lambda_u(node) (to the group threshold 1e-3) and
          ln(10) (-log10 d0) / lambda_u (to O(1), the earlier form); lambda_u from the reduced node of the residence's
          big sizes: (17,17,16) A, (17,16,16) B, (16,16,16) for eK = -1.2.

Claim 3 (direction) -- which cluster ends each (137,132,131) residence (added 2026-10-29)
  for each primary residence not left-censored: the members of the three big clusters at its first record; for every
  record to 50 records after its end, s_c = log10 of the median over members of the max-norm distance to the
  member-wise median point, then the minimum over non-overlapping triples of records (30 time units > period 24.12).
  Reported: the first cluster whose s_c reaches -3, its time relative to the residence end; OLS growth rate of each
  cluster (ln 10 s_c against time) over the triples with s_c in [s_min + 1, -3.5] for the cluster that crosses, and
  over the whole residence for the others; the Floquet exponents of the (137,132,131) orbit from
  analysis/switching_check/return_rate_0088_O.json.

Task 3 -- D-cycle length against eta in run 88 (descriptive, added 2026-10-29; part "dperiod")
  input   the fixed classification outputs (scripts/stageB_classify.py, unchanged): analysis/noiseless_n50/
          stageB_classify.json, _mid.json, _mid2_{horse,boar}.json, stageC_classify_{horse,boar}.json.
  runs    start S8 (r1600) only; cross-host reruns are not in these files. Protocol from the file: "_h0.0025" -> C-h;
          stored eta = eta_nominal/sqrt(10) (Stage C) -> C-k with eta_nominal = eta sqrt(10); otherwise B. eta = 0 is
          the noiseless reference.
  cycles  label D only; the first cycle of each run (transient from S8) is dropped. Cycles of one run are not
          independent samples: per (protocol, eta) the pooled cycle statistics and the per-run medians are both given.
  primary all D cycles. Secondary: the first three D cycles of each run (before the one-way drift seen at the end of
          the D stretches at intermediate eta).
  No model is assumed (no Hansel-Mato-Meunier law). A least-squares line of the per-run median T against ln(eta) over
          the B runs with eta > 0 is reported as a description only, with its points and residuals.
  figure  analysis/paper3_figs/d_period_vs_eta.png (T and the 132 depth of the D cycles against log10 eta).

Sensitivity cases (separate from the primary definitions, reported side by side):
  C1: minimum residence 100 instead of 200; all eta including those with another main type; fit on residence
      durations instead of start-to-start intervals.
  C3: no bridging; bridging < 100 records; minimum 100 records; eta = 1e-6 included.

  python scripts/paper3_aggregates.py
  -> analysis/paper3_figs/aggregates.json
"""
from pathlib import Path
from collections import Counter
import hashlib
import json
import sys
import time

import numpy as np
from scipy.sparse.csgraph import connected_components

ROOT = Path(__file__).resolve().parents[1]
SC = ROOT / "analysis/switching_check"
OUT = ROOT / "analysis/paper3_figs"

DT = 10  # time units per record
MIN_RES_C1, T0_C1, T1_C1 = 200, 1000, 20000
C1_RUNS = {379: [1e-14, 1e-12, 1e-10, 1e-8, 1e-6], 394: [1e-14, 1e-12, 1e-10, 1e-8, 1e-6],
           397: [1e-14, 1e-12, 1e-10, 1e-8, 1e-6]}
TARGET = (137, 132, 131)
BRIDGE_C3, MIN_REC_C3, BIG = 50, 200, 20
C3_ETAS = [1e-14, 1e-12, 1e-10, 1e-8, 1e-6]
C3_FIT = (1e-14, 1e-8)
RECORDED = {"C1_rate": {379: 0.0169, 394: 0.0223, 397: 0.047},
            "C1_mean_interval": {379: {1e-14: 1782, 1e-12: 1461, 1e-10: 1280, 1e-8: 936},
                                 394: {1e-14: 1383, 1e-12: 1167, 1e-10: 969, 1e-8: 761},
                                 397: {1e-14: 693, 1e-12: 562, 1e-10: 496}},
            "C1_switches": {379: {1e-14: 10, 1e-12: 13, 1e-10: 14, 1e-8: 20},
                            394: {1e-14: 14, 1e-12: 16, 1e-10: 19, 1e-8: 24},
                            397: {1e-14: 26, 1e-12: 33, 1e-10: 38}},
            "C3_slope_eta_le_1e-8": 930, "C3_slope_all": 840,
            "C3_residences": {1e-8: [2820, 4050, 2410, 4050], 1e-10: [6060, 6510, 6880, 4200],
                              1e-12: [11390, 12540, 12540], 1e-14: [16230, 15000]},
            "source": "docs/RESULTS_NOTES.md 2026-10-23 (S10.2 entry, N=50 itinerancy entry) and 2026-10-24 (88)"}


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def tag(eta):
    return "" if eta == 1e-10 else f"_n{eta:.0e}"


def ols(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    n = len(x)
    if n < 3 or np.ptp(x) == 0:
        return dict(n=n, note="fewer than 3 points or one x value; no fit")
    A = np.c_[x, np.ones(n)]
    (b, a), *_ = np.linalg.lstsq(A, y, rcond=None)
    r = y - (a + b * x)
    s2 = float(r @ r) / (n - 2)
    cov = s2 * np.linalg.inv(A.T @ A)
    return dict(n=n, slope=float(b), slope_se=float(np.sqrt(cov[0, 0])), intercept=float(a),
                intercept_se=float(np.sqrt(cov[1, 1])), rate_1_over_slope=float(1 / b) if b > 0 else None,
                r2=float(1 - (r @ r) / ((y - y.mean()) @ (y - y.mean()))), rms_residual=float(np.sqrt((r @ r) / n)),
                residuals=[float(v) for v in r])


def dist(v):
    v = np.asarray(v, float)
    if not len(v):
        return dict(n=0)
    return dict(n=len(v), mean=float(v.mean()), median=float(np.median(v)), sd=float(v.std(ddof=1)) if len(v) > 1 else None,
                min=float(v.min()), max=float(v.max()), cv=float(v.std(ddof=1) / v.mean()) if len(v) > 1 else None)


# ---------------- Claim 1 ----------------
def config(lab):
    cnt = np.bincount(lab)
    cl = [frozenset(np.nonzero(lab == c)[0].tolist()) for c in np.nonzero(cnt >= 2)[0]]
    return frozenset(cl), tuple(sorted((len(c) for c in cl), reverse=True))


def c1_one(labels, main_type, min_res):
    R = len(labels)
    cf = [config(l) for l in labels]
    runs, k0 = [], 0
    for k in range(1, R + 1):
        if k == R or cf[k][0] != cf[k0][0]:
            runs.append((k0, k))
            k0 = k
    res = []
    for k0, k1 in runs:
        if cf[k0][1] != main_type:
            continue
        t0, t1 = DT * (k0 + 1), DT * k1 + DT  # time of first record; time of last record + 10
        if t1 - t0 >= min_res:
            res.append(dict(start=t0, end=t1, L=t1 - t0, cfg=cf[k0][0]))
    eps = []
    for r in res:
        if eps and eps[-1]["cfg"] == r["cfg"]:
            eps[-1]["end"] = r["end"]
            eps[-1]["n_res"] += 1
            eps[-1]["L_sum"] += r["L"]
        else:
            eps.append(dict(start=r["start"], end=r["end"], n_res=1, L_sum=r["L"], cfg=r["cfg"]))
    intervals = []
    for j in range(len(eps) - 1):
        if eps[j]["start"] >= T0_C1:
            intervals.append(dict(start=eps[j]["start"], next_start=eps[j + 1]["start"],
                                  interval=eps[j + 1]["start"] - eps[j]["start"]))
    durations = [dict(start=r["start"], end=r["end"], L=r["L"], right_censored=r["end"] >= T1_C1,
                      left_censored=r["start"] < T0_C1) for r in res]
    switches = sum(1 for j in range(1, len(eps)) if eps[j]["start"] >= T0_C1)
    for e in eps:
        e.pop("cfg")
    return dict(n_records=R, n_residences=len(res), residences=durations, episodes=eps, switches_in_window=switches,
                intervals=intervals)


def most_type(labels):
    c = Counter(config(l)[1] for l in labels)
    (t, n), = c.most_common(1)
    return t, n / len(labels)


def claim1(inputs):
    out = {}
    for idx, etas in C1_RUNS.items():
        lab = {}
        w0 = T0_C1 // DT - 1  # record of t = 1000
        for eta in etas:
            p = SC / f"rn_{idx:04d}{tag(eta)}.npz"
            inputs[str(p.relative_to(ROOT))] = sha(p)
            lab[eta] = np.load(p)["labels"]
        ref_type, _ = most_type(lab[1e-10][w0:])
        per = {}
        for eta in etas:
            mt, frac = most_type(lab[eta][w0:])
            per[f"{eta:.0e}"] = dict(eta=eta, ln_inv_eta=float(np.log(1 / eta)), main_type=list(mt), main_type_frac=frac,
                                     same_main_type_as_ref=mt == ref_type,
                                     primary=c1_one(lab[eta], ref_type, MIN_RES_C1),
                                     sens_minres100=c1_one(lab[eta], ref_type, 100))
            pr = per[f"{eta:.0e}"]["primary"]
            pr["interval_distribution"] = dist([i["interval"] for i in pr["intervals"]])
        fits = {}

        def pts(key, which, sel):
            x, y = [], []
            for k, v in per.items():
                if not sel(v):
                    continue
                if which == "interval":
                    ys = [i["interval"] for i in v[key]["intervals"]]
                else:
                    ys = [r["L"] for r in v[key]["residences"] if not r["right_censored"] and not r["left_censored"]]
                x += [v["ln_inv_eta"]] * len(ys)
                y += ys
            return x, y

        incl = lambda v: v["same_main_type_as_ref"]  # noqa: E731
        x, y = pts("primary", "interval", incl)
        fits["primary_pooled_intervals"] = ols(x, y)
        fits["primary_pooled_intervals"]["eta_used"] = [v["eta"] for v in per.values() if incl(v)]
        mx, my = [], []
        for v in per.values():
            iv = [i["interval"] for i in v["primary"]["intervals"]]
            if incl(v) and iv:
                mx.append(v["ln_inv_eta"])
                my.append(float(np.mean(iv)))
        fits["secondary_mean_per_eta"] = ols(mx, my)
        x, y = pts("primary", "interval", lambda v: True)
        fits["sens_all_eta"] = ols(x, y)
        x, y = pts("sens_minres100", "interval", incl)
        fits["sens_minres100"] = ols(x, y)
        x, y = pts("primary", "duration", incl)
        fits["sens_residence_durations"] = ols(x, y)
        out[str(idx)] = dict(ref_main_type=list(ref_type), per_eta=per, fits=fits)
    return out


# ---------------- Claim 3 ----------------
def c3_labels(path):
    zs = np.load(path, mmap_mode="r")
    n = zs.shape[1] // 2
    big, sp137 = [], []
    for z in zs:
        P = np.c_[z[:n], z[n:]]
        D = np.max(abs(P[:, None] - P[None]), axis=2)
        lab = connected_components(D < 1e-3, directed=False)[1]
        cnt = np.bincount(lab)
        big.append(tuple(sorted((int(s) for s in cnt if s >= BIG), reverse=True)))
        s = None
        if 137 in cnt:
            mem = np.nonzero(lab == int(np.nonzero(cnt == 137)[0][0]))[0]
            sub = D[np.ix_(mem, mem)]
            s = float(np.log10(np.median([np.median(np.delete(sub[a], a)) for a in range(len(mem))])))
        sp137.append(s)
    return big, sp137


def c3_res(big, sp, bridge, minrec):
    R = len(big)
    runs, k0 = [], 0
    for k in range(1, R + 1):
        if k == R or big[k] != big[k0]:
            runs.append((k0, k, big[k0]))
            k0 = k
    segs = []
    for r0, r1, b in runs:
        if b != TARGET:
            continue
        if segs and r0 - segs[-1][1] < bridge:
            segs[-1][1] = r1
        else:
            segs.append([r0, r1])
    out = []
    for s0, s1 in segs:
        if s1 - s0 < minrec:
            continue
        out.append(dict(start=DT * (s0 + 1), end=DT * s1 + DT, L=DT * (s1 - s0), left_censored=s0 == 0,
                        right_censored=s1 == R, spread137_start_log10=sp[s0]))
    return out


def claim3(inputs):
    per = {}
    for eta in C3_ETAS:
        p = SC / f"rn_0088{tag(eta)}_t80000_states_z.npy"
        inputs[str(p.relative_to(ROOT))] = sha(p)
        big, sp = c3_labels(p)
        per[f"{eta:.0e}"] = dict(eta=eta, ln_inv_eta=float(np.log(1 / eta)), n_records=len(big),
                                 primary=c3_res(big, sp, BRIDGE_C3, MIN_REC_C3),
                                 sens_no_bridge=c3_res(big, sp, 1, MIN_REC_C3),
                                 sens_bridge100=c3_res(big, sp, 100, MIN_REC_C3),
                                 sens_minrec100=c3_res(big, sp, BRIDGE_C3, 100))
        u = [r["L"] for r in per[f"{eta:.0e}"]["primary"] if not (r["left_censored"] or r["right_censored"])]
        per[f"{eta:.0e}"]["primary_distribution"] = dist(u)

    def fit(key, lo, hi):
        x, y = [], []
        for v in per.values():
            if lo <= v["eta"] <= hi:
                ys = [r["L"] for r in v[key] if not (r["left_censored"] or r["right_censored"])]
                x += [v["ln_inv_eta"]] * len(ys)
                y += ys
        f = ols(x, y)
        f["eta_range"] = [lo, hi]
        return f

    fits = dict(primary=fit("primary", *C3_FIT), sens_include_1e_6=fit("primary", 1e-14, 1e-6),
                sens_no_bridge=fit("sens_no_bridge", *C3_FIT), sens_bridge100=fit("sens_bridge100", *C3_FIT),
                sens_minrec100=fit("sens_minrec100", *C3_FIT))
    return dict(per_eta=per, fits=fits)


# ---------------- Claim 1 (a) ----------------
W1A = 25
NODE = ROOT / "analysis/heteroclinic_n50"
LAMBDA_NODES = {(-0.9, "A"): "node_17-17-16_q-0.900_s1.json", (-0.9, "B"): "node_17-16-16-1_q-0.900_s1.json",
                (-1.0, "A"): "node_17-17-16_q-1.000_s1.json", (-1.0, "B"): "node_17-16-16-1_q-1.000_s1.json",
                (-1.2, "C"): "node_16-16-16-1-1_q-1.200_s2.json"}
RECORDED_GROWTH = {379: 0.0175, 386: 0.0174, 393: 0.0217, 394: 0.0215, 397: 0.040}


def lambda_nodes(inputs):
    out = {}
    for (q, typ), f in LAMBDA_NODES.items():
        inputs[str((NODE / f).relative_to(ROOT))] = sha(NODE / f)
        nd = json.loads((NODE / f).read_text())["node"]
        out[f"{q}_{typ}"] = dict(file=f, sizes=nd["sizes"], lambda_u=float(max(nd["internal"])))
    return out


def sets_of(D, big=10, th=1e-3):
    lab = connected_components(D < th, directed=False)[1]
    cnt = np.bincount(lab)
    return [frozenset(np.nonzero(lab == c)[0].tolist()) for c in np.nonzero(cnt >= big)[0]]


def claim1a(inputs):
    lam = lambda_nodes(inputs)
    out = dict(lambda_nodes=lam, runs={})
    for f in sorted((ROOT / "analysis/itinerancy_n50").glob("trace_0*.npz")):
        if "n1e-06" in f.name:
            continue
        inputs[str(f.relative_to(ROOT))] = sha(f)
        d = np.load(f)
        z, eK = d["z"], round(float(d["eps"] * d["K"]), 3)
        n = z.shape[1] // 2
        nw = len(z) // W1A
        Dm = np.full((nw, n, n), np.inf)
        for w in range(nw):
            for zz in z[w * W1A:(w + 1) * W1A]:
                P = np.c_[zz[:n], zz[n:]]
                Dm[w] = np.minimum(Dm[w], np.max(abs(P[:, None] - P[None]), axis=2))
        cfg = [frozenset(sets_of(Dm[w])) for w in range(nw)]
        res, k = [], 0
        while k < nw:
            j = k
            while j + 1 < nw and cfg[j + 1] == cfg[k]:
                j += 1
            if j - k + 1 >= 8:
                res.append((k, j, cfg[k]))
            k = j + 1
        ev, other = [], Counter()
        for (a0, a1, A), (b0, _, B) in zip(res[:-1], res[1:]):
            leav = []
            for C in A:
                succ = max(B, key=lambda X: len(X & C)) if B else frozenset()
                leav += [(e, C) for e in C if e not in succ]
            if len(leav) != 1:
                other[len(leav)] += 1
                continue
            e, C = leav[0]
            mem = sorted(C - {e})
            g = np.median(Dm[:, e, mem], axis=1)
            wmin = a0 + int(np.argmin(g[a0:a1 + 1]))
            cross = next((w for w in range(wmin, nw) if g[w] >= 1e-3), None)
            row = dict(res_start=W1A * a0, res_end=W1A * (a1 + 1), next_start=W1A * b0, leaver=int(e),
                       sizes=sorted((len(X) for X in A), reverse=True), U_size=len(C),
                       log10_gmin=float(np.log10(g[wmin])), t_min=W1A * wmin + W1A / 2,
                       t_cross=None if cross is None else W1A * cross + W1A / 2)
            if cross is not None:
                ws = [w for w in range(wmin, cross + 1)
                      if np.log10(g[wmin]) + 1 <= np.log10(g[w]) <= -3.5]
                if len(ws) >= 5:
                    f_ = ols([W1A * w + W1A / 2 for w in ws], np.log(g[ws]))
                    f_.pop("rate_1_over_slope")
                    row["fit"] = {k_: v for k_, v in f_.items() if k_ != "residuals"}
                    row["rate"] = f_["slope"]
            ev.append(row)
        rates = [r["rate"] for r in ev if "rate" in r]
        idx = int(f.name[6:10])
        out["runs"][f.stem] = dict(index=idx, eK=eK, n_windows=nw, n_residences=len(res), events=ev,
                                   events_other_leaver_counts=dict(other), rate_distribution=dist(rates),
                                   recorded_growth=RECORDED_GROWTH.get(idx) if "_n" not in f.stem else None,
                                   lambda_u={k_: v["lambda_u"] for k_, v in lam.items() if k_.startswith(f"{eK}_")})
    return out


# ---------------- Claim 1 (b) ----------------
def claim1b(inputs, lam):
    out = {}
    for idx in (379, 386, 393, 394, 397):
        f = ROOT / f"analysis/noiseless_n50/nl_{idx:04d}_t5000.npz"
        inputs[str(f.relative_to(ROOT))] = sha(f)
        d = np.load(f)
        g, sp, eK = d["group"], d["spread"].astype(float), round(float(d["eps"] * d["K"]), 3)
        st = []
        for t in range(9, len(g), 10):
            L = g[t]
            st.append((t + 1, frozenset(frozenset(np.nonzero(L == x)[0].tolist()) for x in np.unique(L)
                                        if np.sum(L == x) >= 10)))
        res, k = [], 0
        while k < len(st):
            j = k
            while j + 1 < len(st) and st[j + 1][1] == st[k][1]:
                j += 1
            if st[j][0] - st[k][0] >= 200:
                res.append((st[k][0], st[j][0], st[k][1]))
            k = j + 1
        rows = []
        for (a0, a1, A), (_, _, B) in zip(res[:-1], res[1:]):
            ha = {i: c for c in A for i in c}
            hb = {i for c in B for i in c}
            out_ = sorted(set(ha) - hb)
            if len(out_) != 1 or a1 - a0 < 1500:
                continue
            e = out_[0]
            U = sorted(ha[e])
            r0 = a0 // 10  # spread record j holds time 10 (j + 1): first record at or after a0
            sizes = tuple(sorted(map(len, A), reverse=True))
            typ = {(17, 17, 16): "A", (17, 16, 16): "B", (16, 16, 16): "C"}.get(sizes)
            lu = lam.get(f"{eK}_{typ}", {}).get("lambda_u") if typ else None
            d0e, d0max = float(sp[r0, e]), float(np.nanmax(sp[r0, U]))
            j1 = (a1 - 10) // 10
            js = [j for j in range(r0, j1 + 1) if np.isfinite(sp[j, e]) and d0e + 1 <= sp[j, e] <= -3.5]
            row = dict(start=int(a0), end=int(a1), L=int(a1 - a0), sizes=list(sizes), type=typ, leaver=int(e),
                       leaver_is_max_spread=bool(sp[r0, e] >= d0max), log10_d0_leaver=d0e, log10_d0_groupmax=d0max,
                       lambda_u_node=lu)
            if len(js) >= 5:
                fit = ols([10 * (j + 1) for j in js], np.log(10) * sp[js, e])
                row["rate_fit"] = fit["slope"]
                row["rate_fit_r2"] = fit["r2"]
                row["L_pred_fit_to_O1"] = float(-d0e * np.log(10) / fit["slope"])
            if lu:
                row["L_pred_node_to_1e-3"] = float(np.log(10) * (-3 - d0e) / lu)
                row["L_pred_node_to_O1"] = float(np.log(10) * (-d0e) / lu)
                row["rel_err_node_to_1e-3"] = row["L_pred_node_to_1e-3"] / row["L"] - 1
                row["rel_err_node_to_O1"] = row["L_pred_node_to_O1"] / row["L"] - 1
            rows.append(row)
        out[str(idx)] = dict(eK=eK, residences=rows)
    rec = ROOT / "analysis/noiseless_n50/residences.json"
    inputs[str(rec.relative_to(ROOT))] = sha(rec)
    return dict(runs=out, recorded_file="analysis/noiseless_n50/residences.json (scripts/noiseless_switches.py)",
                recorded=json.loads(rec.read_text()))


# ---------------- Claim 3 (direction) ----------------
def claim3_direction(res3, inputs):
    rr = SC / "return_rate_0088_O.json"
    inputs[str(rr.relative_to(ROOT))] = sha(rr)
    O = json.loads(rr.read_text())["O"]
    out = dict(floquet_O=dict(period=O["period"], internal=O["internal"], collective=O["collective"]), residences=[])
    for key, v in res3["per_eta"].items():
        zs = np.load(SC / f"rn_0088{tag(v['eta'])}_t80000_states_z.npy", mmap_mode="r")
        n = zs.shape[1] // 2
        for r in v["primary"]:
            if r["left_censored"]:
                continue
            k0, k1 = r["start"] // DT - 1, r["end"] // DT - 1  # record indices of start and of end
            z = zs[k0]
            P = np.c_[z[:n], z[n:]]
            lab = connected_components(np.max(abs(P[:, None] - P[None]), axis=2) < 1e-3, directed=False)[1]
            cnt = np.bincount(lab)
            mem = {int(s): np.nonzero(lab == int(np.nonzero(cnt == s)[0][0]))[0] for s in TARGET}
            kend = min(k1 + 50, len(zs))
            S = {s: [] for s in TARGET}
            for k in range(k0, kend):
                z = zs[k]
                for s, m in mem.items():
                    Q = np.c_[z[m], z[n + m]]
                    S[s].append(np.median(np.max(abs(Q - np.median(Q, axis=0)), axis=1)))
            nt = (kend - k0) // 3
            tt = np.array([DT * (k0 + 1) + DT * (3 * i + 1) for i in range(nt)], float)
            row = dict(eta=v["eta"], start=r["start"], end=r["end"], right_censored=r["right_censored"], clusters={})
            first = None
            for s in TARGET:
                a = np.log10(np.array(S[s][:3 * nt]).reshape(nt, 3).min(axis=1))
                inside = tt <= r["end"]
                c = next((i for i in range(nt) if a[i] >= -3), None)
                info = dict(log10_min_in_residence=float(a[inside].min()), log10_max_in_residence=float(a[inside].max()),
                            t_cross_1e_3=None if c is None else float(tt[c]))
                if c is not None and (first is None or tt[c] < first[1]):
                    first = (s, float(tt[c]))
                info["rate_whole_residence"] = ols(tt[inside], np.log(10) * a[inside]).get("slope")
                imin = int(np.argmin(a[inside]))
                sel = [i for i in range(imin, c if c is not None else int(inside.sum()))
                       if a[imin] + 1 <= a[i] <= -3.5]
                if len(sel) >= 5:
                    f_ = ols(tt[sel], np.log(10) * a[sel])
                    info["rate_growth_segment"] = f_["slope"]
                    info["rate_growth_segment_r2"] = f_["r2"]
                    info["growth_segment_points"] = len(sel)
                row["clusters"][str(s)] = info
            row["first_to_1e-3"] = None if first is None else first[0]
            row["t_first_minus_end"] = None if first is None else first[1] - r["end"]
            out["residences"].append(row)
    return out


# ---------------- Task 3: D-cycle length against eta ----------------
NL = ROOT / "analysis/noiseless_n50"
CLASSIFY = ["stageB_classify.json", "stageB_classify_mid.json", "stageB_classify_mid2_horse.json",
            "stageB_classify_mid2_boar.json", "stageC_classify_horse.json", "stageC_classify_boar.json"]


def protocol(r):
    if "_h0.0025" in r["file"]:
        return "C-h", r["eta"]
    for nom in (1e-20, 3e-20):
        if abs(r["eta"] - nom / np.sqrt(10)) < 1e-6 * nom:
            return "C-k", nom
    return "B", r["eta"]


def dperiod(inputs):
    runs = {}
    for f in CLASSIFY:
        inputs[str((NL / f).relative_to(ROOT))] = sha(NL / f)
        for r in json.loads((NL / f).read_text())["runs"]:
            if "r1600" in r["file"]:
                runs[r["file"]] = (f, r)
    groups, per_run = {}, []
    for fn, (src, r) in sorted(runs.items()):
        prot, eta = protocol(r)
        cyc = [c for c in r["cycles"][1:] if c["label"] == "D"]
        if not cyc:
            continue
        T = [float(c["T"]) for c in cyc]
        dep = [float(c["depth132small"]) for c in cyc if c["depth132small"] is not None]
        per_run.append(dict(file=fn.split("/")[-1], source=src, protocol=prot, eta=eta, seed=r["seed"],
                            classification=r["classification"], labels="".join(c["label"] for c in r["cycles"]),
                            n_D=len(T), T=T, depth132=dep, T_median=float(np.median(T)),
                            T_median_first3=float(np.median(T[:3])), depth132_median=float(np.median(dep))))
        groups.setdefault((prot, eta), []).append(per_run[-1])
    table = []
    for (prot, eta), rs in sorted(groups.items(), key=lambda kv: (kv[0][0], kv[0][1])):
        allT = [t for x in rs for t in x["T"]]
        f3 = [t for x in rs for t in x["T"][:3]]
        table.append(dict(protocol=prot, eta=eta, n_runs=len(rs), n_cycles=len(allT), cycles=dist(allT),
                          q25=float(np.percentile(allT, 25)), q75=float(np.percentile(allT, 75)),
                          first3=dist(f3), run_medians=sorted(x["T_median"] for x in rs),
                          depth132_median=float(np.median([d for x in rs for d in x["depth132"]]))))
    sel = [x for x in per_run if x["protocol"] == "B" and x["eta"] > 0]
    fit = ols([np.log(x["eta"]) for x in sel], [x["T_median"] for x in sel])
    fit["points"] = [dict(eta=x["eta"], seed=x["seed"], T_median=x["T_median"]) for x in sel]
    fit["note"] = "description only; eta values with D: see table (clustered at 1e-22 and 1e-20...2e-20)"
    fit3 = ols([np.log(x["eta"]) for x in sel], [x["T_median_first3"] for x in sel])
    out = dict(per_run=per_run, table=table, descriptive_fit_B_run_medians=fit,
               descriptive_fit_B_first3_medians=fit3,
               eta_values_with_D=sorted({(x["protocol"], x["eta"]) for x in per_run}))
    plot_dperiod(per_run)
    return out


def plot_dperiod(per_run):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    col = {"B": "#2a78d6", "C-h": "#eb6834", "C-k": "#1baf7a"}
    mk = {"B": "o", "C-h": "s", "C-k": "^"}
    off = {"B": 0.0, "C-h": 0.08, "C-k": 0.16}
    X0 = -23.0  # position of eta = 0
    fig, ax = plt.subplots(2, 1, figsize=(6.4, 6.0), sharex=True)
    seen = set()
    for x in per_run:
        lx = X0 if x["eta"] == 0 else np.log10(x["eta"])
        xx = lx + off[x["protocol"]] + 0.02 * ((x["seed"] or 0) - 4)
        lab = x["protocol"] if x["protocol"] not in seen else None
        seen.add(x["protocol"])
        ax[0].scatter([xx] * x["n_D"], x["T"], s=14, marker=mk[x["protocol"]], facecolor="none",
                      edgecolor=col[x["protocol"]], linewidth=1.0, label=lab)
        ax[1].scatter([xx] * len(x["depth132"]), x["depth132"], s=14, marker=mk[x["protocol"]], facecolor="none",
                      edgecolor=col[x["protocol"]], linewidth=1.0)
    for a in ax:
        a.grid(alpha=0.25, linewidth=0.6)
        a.spines[["top", "right"]].set_visible(False)
    ax[0].set_ylabel("length T of D cycles")
    ax[1].set_ylabel("log10 depth of 132 small group")
    ax[1].set_xlabel("log10 eta (noise per kick, nominal)   [leftmost: eta = 0]")
    ticks = [X0, -22, -21, -20, -19.5]
    ax[1].set_xticks(ticks, ["0", "-22", "-21", "-20", "-19.5"])
    ax[0].legend(title="protocol", frameon=False, fontsize=8)
    ax[0].set_title("Run 88 from S8: D cycles (first cycle of each run dropped)", fontsize=10)
    fig.tight_layout()
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / "d_period_vs_eta.png", dpi=150)
    plt.close(fig)


def compare(res, inputs):
    """Recorded values next to the re-aggregated ones (no value is overwritten)."""
    out = dict(C1={}, C3={})
    for idx in ("379", "394", "397"):
        v = res["claim1"][idx]
        rows = []
        for eta, m in RECORDED["C1_mean_interval"][int(idx)].items():
            e = v["per_eta"][f"{eta:.0e}"]
            di = e["primary"]["interval_distribution"]
            rows.append(dict(eta=eta, recorded_mean=m, recorded_switches=RECORDED["C1_switches"][int(idx)][eta],
                             primary_mean=di.get("mean"), primary_median=di.get("median"),
                             primary_switches=e["primary"]["switches_in_window"],
                             minres100_mean=dist([i["interval"] for i in e["sens_minres100"]["intervals"]]).get("mean"),
                             minres100_switches=e["sens_minres100"]["switches_in_window"]))
        x = [r["eta"] for r in rows]
        rec_fit = ols(np.log(1 / np.asarray(x)), [r["recorded_mean"] for r in rows])
        out["C1"][idx] = dict(rows=rows, recorded_rate=RECORDED["C1_rate"][int(idx)],
                              rate_from_recorded_means=rec_fit.get("rate_1_over_slope"),
                              primary_rate=v["fits"]["primary_pooled_intervals"].get("rate_1_over_slope"),
                              secondary_rate=v["fits"]["secondary_mean_per_eta"].get("rate_1_over_slope"),
                              sens_all_eta_rate=v["fits"]["sens_all_eta"].get("rate_1_over_slope"),
                              sens_minres100_rate=v["fits"]["sens_minres100"].get("rate_1_over_slope"))
    # C3: the recorded residences, and noise_scaling_0088.json (written by scripts/noise_scaling_check.py), where
    # residences of >= 2000 after the first match the recorded list
    ns = SC / "noise_scaling_0088.json"
    inputs[str(ns.relative_to(ROOT))] = sha(ns)
    nsd = json.loads(ns.read_text())
    ns_long = {f"{r['eta']:.0e}": [q["L"] for q in r["residences"][1:] if q["L"] >= 2000] for r in nsd}
    x, y = [], []
    for eta, L in RECORDED["C3_residences"].items():
        x += [np.log(1 / eta)] * len(L)
        y += L
    out["C3"] = dict(recorded_slope_eta_le_1e8=RECORDED["C3_slope_eta_le_1e-8"],
                     slope_refit_of_recorded_list=ols(x, y), noise_scaling_long_residences=ns_long,
                     primary_slope=res["claim3"]["fits"]["primary"]["slope"],
                     note="noise_scaling_check.py splits residences at every change of the full size tuple at 1e-3 "
                          "(singletons included); the primary rule here uses only clusters >= 20 and bridges < 50 records")
    return out


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--parts", nargs="*", default=None, help="only these parts (dperiod); others kept from the file")
    a = ap.parse_args()
    if a.parts:
        t0 = time.time()
        res = json.loads((OUT / "aggregates.json").read_text())
        inputs = res["inputs"]
        if "dperiod" in a.parts:
            res["dperiod"] = dperiod(inputs)
        res.setdefault("part_runs", []).append(dict(parts=a.parts, created=time.strftime("%Y-%m-%d %H:%M:%S"),
                                                    command=" ".join(["python"] + sys.argv),
                                                    script_sha256=sha(__file__)))
        (OUT / "aggregates.json").write_text(json.dumps(res, indent=1, default=str))
        print("updated", a.parts, round(time.time() - t0, 1), "s")
        return
    t0 = time.time()
    inputs = {}
    res = dict(created=time.strftime("%Y-%m-%d %H:%M:%S"), command=" ".join(["python"] + sys.argv),
               script="scripts/paper3_aggregates.py", script_sha256=sha(__file__),
               numpy=np.__version__, python=sys.version.split()[0],
               rules=dict(C1=dict(labels="saved labels, threshold 1e-4, every 10 time units", min_residence=MIN_RES_C1,
                                  window=[T0_C1, T1_C1], clusters=">=2 elements as element sets",
                                  censoring="episode starting before window start: interval dropped; last episode: no interval",
                                  eta_rule="fit uses eta whose main type equals that at eta=1e-10",
                                  record_time="t = 10 (k + 1); residence end = last record time + 10"),
                          C3=dict(threshold=1e-3, big=BIG, target=list(TARGET), bridge_records=BRIDGE_C3,
                                  min_records=MIN_REC_C3, fit_eta=list(C3_FIT),
                                  censoring="residence containing first or last record excluded from fits")),
               recorded_values=json.loads(json.dumps(RECORDED, default=str)), inputs=inputs)
    res["claim1"] = claim1(inputs)
    res["claim3"] = claim3(inputs)
    res["comparison"] = compare(res, inputs)
    res["claim1a_growth"] = claim1a(inputs)
    res["claim1b_noiseless"] = claim1b(inputs, res["claim1a_growth"]["lambda_nodes"])
    res["claim3_direction"] = claim3_direction(res["claim3"], inputs)
    res["dperiod"] = dperiod(inputs)
    res["elapsed_s"] = time.time() - t0
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "aggregates.json").write_text(json.dumps(res, indent=1, default=str))
    print("written", OUT / "aggregates.json", round(res["elapsed_s"], 1), "s")


if __name__ == "__main__":
    main()
