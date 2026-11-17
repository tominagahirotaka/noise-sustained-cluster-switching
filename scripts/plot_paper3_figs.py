#!/usr/bin/env python3
"""Figures for paper 3 from existing data only (no new integration; 2026-10-31, paper3/skeleton.md, figure plan).

fig1: switching at N = 50, run 394 at eta = 1e-10 (analysis/itinerancy_n50/trace_0394.npz, full state every time unit).
   Firing times (upward x = 0) are taken per cycle of element 0 with firing_matrix of ejections_n50.py.
   (a) the cluster of every element in each cycle (colour; white: not in a cluster of >= 10, i.e. in transit, or a
       cycle with a missing firing). The clusters are the groups of firing phases separated by gaps > 0.5 time units
       with at least 10 elements; their identity (colour, numbered in firing order in the first cycle) is followed from
       cycle to cycle by the largest overlap of members. Elements sorted by first cluster, then by first move.
   (b) the departures of analysis/heteroclinic_n50/ejections_0394.json (all 20 advanced, to the cluster that fires
       just before): cycle of departure against the tracked cluster it left, with the clusters in firing order.
fig2: schematic of the network of saddle orbits at N = 50 (no data; drawn after Sec. III of the text).
   (a) a type-A orbit as three clusters on the cycle of the collective oscillation, numbered in firing order, with the
       unstable cluster U; the two branches of the unstable direction (leading: the element joins the cluster that
       fires before U, giving a permuted copy of type A; lagging: it stays behind U as an isolated element, type B).
   (b) the classes of copies A_k, B_k (k: the cluster that plays U) and the connections between them.
fig3: departure time scale at N = 50.
   (a) run 394 at eta = 1e-14 (trace_0394_n1e-14.npz): for each departure with a fitted rate in aggregates.json
       (claim1a_growth), the median per-cycle minimum distance g of the leaver to the other members of its cluster
       (windows of 25 time units, as in paper3_aggregates.claim1a), from the window of its minimum to the window where
       it reaches 1e-3, against the time since the minimum. Dashed: slope lambda_u of the type-A orbit at eK = -1.0.
   (b) the 40 noiseless residences with an identified orbit type (aggregates.json claim1b_noiseless): observed length
       L against L_pred = (ln 1e-3 - ln d0)/lambda_u (Eq. lpred), coloured by eK. Diagonal: L = L_pred.
   (c) the relative error L_pred/L - 1 of the same residences against L.
fig4: partitions and transverse stability (analysis/heteroclinic_n50/balanced_partitions.json, n50_grid.json,
   analysis/switching_check/renoise_summary.json).
   (a) eps = 0.25: largest transverse exponent of the clusters of >= 2 elements (max_big) of the computed periodic
       orbits against N, for eK = -0.9, -1.0, -1.2 (colour). Filled: the most balanced three-cluster partition of each
       N; open: the other partitions computed (one more three-cluster partition and the partitions with one isolated
       element; for N = 100, (33,33,33,1) from seed 3 is the weakly unstable exception of Sec. IV).
   (b) N = 50, seed 1: points of the (eps, eK) grid where exactly one cluster of (17,17,16) or (17,16,16,1) is
       transversely unstable (n_unstable_big == 1; shaded), and for the 418 runs of renoise_summary.json with N = 50,
       the number with persistent switching / the number of runs at that point.
fig5: run 88 (N = 400), from aggregates.json (claim3, claim3_direction) and the saved states.
   (a) residence length L in (137,132,131) against ln(1/eta), primary counting rule. Filled: residences used in the fit
       (not censored, 1e-14 <= eta <= 1e-8); open: censored first or last residences; grey: eta = 1e-6 (excluded from
       the main fit in advance). Line: the fit L = a + b ln(1/eta) of fits.primary.
   (b) eta = 1e-14, the residence from t = 20770 to 41170: spread of each of the three clusters (members fixed at the
       start; median distance of the members from their median point; minimum over blocks of three records, as in
       paper3_aggregates.claim3_direction), from rn_0088_n1e-14_t80000_states_z.npy. Dashed: slope lambda_perp of O,
       drawn above the 137 curve;
       dotted: 1e-3; vertical line: end of the residence.
figS1 (supplement S5; the former Fig. 6): noiseless recurrence of run 88 from S8 and S6, from the two eta = 0 runs of
   analysis/noiseless_n50/stageB_classify.json (the classification of Sec. VI). (a) cycle length T (from one split of
   the 137 cluster to the next) against the cycle number; (b) log10 depth of the closest approach of the small subsets:
   depth132small (the 5-element subset to the rest of the 132 cluster) and depth137small (the 12- or 14-element subset
   to the rest of the 137 cluster). Open markers: cycles not labelled D (the transient ones).
fig7: path selection in run 88 from S8 (analysis/noiseless_n50/stageB_classify{,_mid,_mid2_horse,_mid2_boar}.json,
   stageC_classify_{horse,boar}.json; files with "r1600" in the name start from S8, "r1300" from S6, which is left
   out; the cross-host repeats of xhost_long_classify.json are left out). One row per run, grouped by eta and protocol
   (B; C-h: h0.0025 in the file name; C-k: kicks every time unit, eta in the file is the per-kick sd eta/sqrt(10),
   shown at the nominal eta). Each cycle is drawn from its switch time t_split to the next one (the last to t_split +
   T) and coloured by its label: D, F, or c (neither). At the right: the classification of the runs of each group
   (numbers of D, F and O runs; Table III).
Output: analysis/paper3_figs/<name>.{pdf,png}

  ~/miniforge3/envs/research/bin/python scripts/plot_paper3_figs.py [fig1 ...]
"""
from pathlib import Path
import json
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.colors as mcolors  # noqa: E402
import matplotlib.ticker  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from ejections_n50 import firing_matrix  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
IT = ROOT / "analysis/itinerancy_n50"
HN = ROOT / "analysis/heteroclinic_n50"
OUT = ROOT / "analysis/paper3_figs"
INK, INK2, GRID, MUTED = "#0b0b0b", "#52514e", "#e4e3df", "#a3a29e"
SER = ["#2a78d6", "#eb6834", "#1baf7a"]  # reference categorical slots 1-3, as in plot_paper2_figs.py


def style(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(INK2)
    ax.tick_params(colors=INK2, labelsize=8.2)
    ax.grid(color=GRID, lw=.5)
    ax.set_axisbelow(True)


def save(fig, name):
    OUT.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"{name}.{ext}", dpi=600 if ext == "pdf" else 200, bbox_inches="tight")  # PDF rasters at 600 dpi (AIP, combination art; 2026-11-03); PNG unchanged
    plt.close(fig)
    print("wrote", OUT / f"{name}.pdf")


def tracked_clusters(F, T, gap=0.5, big=10):
    """Per cycle: dict id -> set of members for the groups of >= big elements, ids followed by overlap."""
    out, prev = [], None
    for row in F:
        if np.isnan(row).any():
            out.append(None)
            continue
        ph = np.mod(row, T)
        o = np.argsort(ph)
        p = ph[o]
        cut = np.nonzero(np.diff(np.r_[p, p[0] + T]) > gap)[0]
        groups = []
        for a, b in zip(cut, np.r_[cut[1:], cut[0] + len(p)]):
            idx = [o[(a + 1 + j) % len(p)] for j in range(b - a)]
            if len(idx) >= big:
                groups.append(set(int(i) for i in idx))
        if prev is None:
            groups.sort(key=lambda g: np.median(ph[list(g)]))
            cur = dict(enumerate(groups))
        else:
            cur, p0 = {}, prev
            for g in groups:
                k = max(p0, key=lambda q: len(p0[q] & g))
                cur[k] = g
        out.append(cur)
        prev = cur if len(cur) == len(prev or cur) else prev
    return out


def fig1():
    F, T = firing_matrix(np.load(IT / "trace_0394.npz")["z"])
    tc = tracked_clusters(F, T)
    mem = np.full((len(tc), F.shape[1]), np.nan)
    for k, c in enumerate(tc):
        for q, g in (c or {}).items():
            mem[k, list(g)] = q
    k0 = next(k for k, c in enumerate(tc) if c)
    moves = [np.nonzero(np.diff(mem[:, i][~np.isnan(mem[:, i])]) != 0)[0] for i in range(F.shape[1])]
    order = sorted(range(F.shape[1]), key=lambda i: (mem[k0, i], moves[i][0] if len(moves[i]) else 1e9))
    fig, (a, b) = plt.subplots(2, 1, figsize=(3.4, 4.2), sharex=True, gridspec_kw=dict(height_ratios=[2.2, 1]))
    cmap = mcolors.ListedColormap(SER)
    cmap.set_bad("white")
    a.imshow(mem[:, order].T, aspect="auto", interpolation="nearest", cmap=cmap, vmin=-.5, vmax=2.5,
             extent=(0, len(tc) * T, F.shape[1] - .5, -.5), rasterized=True)
    a.set_ylabel("element", fontsize=8.6, color=INK)
    a.set_yticks([0, 49])
    a.set_yticklabels(["1", "50"])
    style(a)
    a.grid(False)
    dep = json.loads((HN / "ejections_0394.json").read_text())["departures"]
    for d in dep:
        q = mem[d["cycle"], d["element"]]
        if not np.isnan(q):
            b.plot(d["cycle"] * T, q, "o", ms=4, color=SER[int(q)], mec="white", mew=.6)
    b.set_yticks([0, 1, 2])
    b.set_yticklabels(["1", "2", "3"])
    b.set_ylim(-.5, 2.5)
    b.set_ylabel("cluster left", fontsize=8.6, color=INK)
    b.set_xlabel("time", fontsize=8.6, color=INK)
    style(b)
    a.text(.01, 1.0, "(a)", transform=a.transAxes, fontsize=9, va="bottom")
    b.text(.01, 1.0, "(b)", transform=b.transAxes, fontsize=9, va="bottom")
    fig.tight_layout(h_pad=.6)
    save(fig, "fig1_switching_394")


def fig3():
    from paper3_aggregates import W1A, sets_of
    ag = json.loads((OUT / "aggregates.json").read_text())
    lam = ag["claim1a_growth"]["lambda_nodes"]
    z = np.load(IT / "trace_0394_n1e-14.npz")["z"]
    n = z.shape[1] // 2
    nw = len(z) // W1A
    Dm = np.full((nw, n, n), np.inf)
    for w in range(nw):
        for zz in z[w * W1A:(w + 1) * W1A]:
            P = np.c_[zz[:n], zz[n:]]
            Dm[w] = np.minimum(Dm[w], np.max(abs(P[:, None] - P[None]), axis=2))
    fig, (a, b, c3) = plt.subplots(1, 3, figsize=(6.8, 2.5), gridspec_kw=dict(width_ratios=[1.15, 1, 1]))
    nev = 0
    for ev in ag["claim1a_growth"]["runs"]["trace_0394_n1e-14"]["events"]:
        if "fit" not in ev or ev.get("t_cross") is None:
            continue
        w0 = ev["res_start"] // W1A
        C = next(c for c in sets_of(Dm[w0]) if ev["leaver"] in c)
        g = np.median(Dm[:, ev["leaver"], sorted(C - {ev["leaver"]})], axis=1)
        wa, wb = int(ev["t_min"] // W1A), int(ev["t_cross"] // W1A)
        t = np.arange(wa, wb + 1) * W1A - wa * W1A
        a.plot(t, np.log10(g[wa:wb + 1]), color=SER[0], lw=.8, alpha=.7)
        nev += 1
    lu = lam["-1.0_A"]["lambda_u"]
    tt = np.array([0, 750])
    a.plot(tt, -10.8 + lu * tt / np.log(10), "--", color=INK, lw=1)
    a.text(330, -10.8 + lu * 330 / np.log(10) - .5, r"slope $\lambda_u$", fontsize=8, color=INK, rotation=0)
    a.set_xlabel("time since minimum", fontsize=8.6, color=INK)
    a.set_ylabel(r"$\log_{10}$ distance of leaving element", fontsize=8.6, color=INK)
    style(a)
    cols = {-0.9: SER[0], -1.0: SER[1], -1.2: SER[2]}
    npt = 0
    for run in ag["claim1b_noiseless"]["runs"].values():
        for r in run["residences"]:
            if "L_pred_node_to_1e-3" not in r:
                continue
            b.loglog(r["L_pred_node_to_1e-3"], r["L"], "o", ms=4, color=cols[run["eK"]], mec="white", mew=.6)
            c3.semilogx(r["L"], 100 * r["rel_err_node_to_1e-3"], "o", ms=4, color=cols[run["eK"]], mec="white",
                        mew=.6)
            npt += 1
    lim = np.array([1e3, 1.2e5])
    b.loglog(lim, lim, color=INK2, lw=.8)
    for q, c in cols.items():
        b.plot([], [], "o", ms=4, color=c, mec="white", mew=.6, label=rf"$\varepsilon K={q}$")
    b.legend(fontsize=7.5, frameon=False, loc="upper left")
    b.set_xlabel(r"predicted $L_{\mathrm{pred}}$", fontsize=8.6, color=INK)
    b.set_ylabel(r"observed $L$", fontsize=8.6, color=INK)
    style(b)
    c3.axhline(0, color=INK2, lw=.8)
    c3.set_xlim(1e3, 1.2e5)
    c3.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    c3.set_xlabel(r"observed $L$", fontsize=8.6, color=INK)
    c3.set_ylabel(r"$L_{\mathrm{pred}}/L-1$ (%)", fontsize=8.6, color=INK)
    style(c3)
    for ax, lab in ((a, "(a)"), (b, "(b)"), (c3, "(c)")):
        ax.text(-.02, 1.02, lab, transform=ax.transAxes, fontsize=9, va="bottom", ha="right")
    fig.tight_layout(w_pad=1.5)
    print("fig3: departures", nev, "residences", npt)
    save(fig, "fig3_departure_time")


def fig4():
    HC = ROOT / "analysis/heteroclinic_n50"
    bp = json.loads((HC / "balanced_partitions.json").read_text())
    grid = json.loads((HC / "n50_grid.json").read_text())
    ren = json.loads((ROOT / "analysis/switching_check/renoise_summary.json").read_text())
    most = {50: [17, 17, 16], 100: [34, 33, 33], 200: [67, 67, 66], 400: [134, 133, 133], 1000: [334, 333, 333]}
    cols = {-0.9: SER[0], -1.0: SER[1], -1.2: SER[2]}
    fig, (a, b) = plt.subplots(1, 2, figsize=(6.8, 2.7), gridspec_kw=dict(width_ratios=[1, 1.15]))
    a.axhline(0, color=INK2, lw=.8)
    dx = {-0.9: .96, -1.0: 1.0, -1.2: 1.04}
    for r in bp:
        N = sum(r["sizes"])
        filled = r["sizes"] == most[N]
        a.semilogx(N * dx[r["q"]], r["max_big"], "o", ms=4.5 if filled else 4, color=cols[r["q"]],
                   mfc=cols[r["q"]] if filled else "white", mec=cols[r["q"]] if not filled else "white",
                   mew=.9 if not filled else .6, zorder=3 if filled else 2)
    # exactly balanced partition, fractions 1/3 (possible only for N divisible by 3; equal_partition.json, 2026-11-03)
    eq = json.loads((HC / "equal_partition.json").read_text())["rows"]
    for r in eq:
        if r["sizes"] == [17, 17, 17] and r["q"] in cols:
            a.axhline(r["max_internal"], color=cols[r["q"]], lw=.8, ls="--", zorder=1)
    for q, c in cols.items():
        a.plot([], [], "o", ms=4.5, color=c, mec="white", mew=.6, label=rf"$\varepsilon K={q}$")
    a.plot([], [], "o", ms=4, color=INK2, mfc="white", label="other partitions")
    a.plot([], [], ls="--", lw=.8, color=INK2, label="fractions 1/3 each")
    a.legend(fontsize=7, frameon=False, loc="upper right")  # upper right since 2026-11-03 (dashed lines below)
    a.set_xticks([50, 100, 200, 400, 1000])
    a.set_xticklabels(["50", "100", "200", "400", "1000"])
    a.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    a.set_xlabel(r"$N$", fontsize=8.6, color=INK)
    a.set_ylabel("largest transverse exponent", fontsize=8.6, color=INK)
    style(a)
    cond = {}
    for r in grid:
        k = (r["eps"], round(r["q"], 2))
        cond[k] = cond.get(k, False) or r["n_unstable_big"] == 1
    qs = sorted({q for _, q in cond}, reverse=True)
    X = {q: i for i, q in enumerate(qs)}
    for (e, q), ok in cond.items():
        b.add_patch(plt.Rectangle((X[q] - .5, e - .025), 1, .05, facecolor=SER[0] if ok else "white", alpha=.22 if ok
                                  else 1, edgecolor=GRID, lw=.5))
    cnt = {}
    for r in ren:
        if r["N"] == 50:
            k = (r["eps"], round(r["eK"], 2))
            c = cnt.setdefault(k, [0, 0])
            c[0] += bool(r["sustained"])
            c[1] += 1
    for (e, q), (k, n) in cnt.items():
        b.text(X[q], e, f"{k}/{n}", fontsize=6.8, ha="center", va="center", color=INK if k else INK2,
               fontweight="bold" if k else "normal")
    b.set_xlim(-.5, len(qs) - .5)
    b.set_ylim(0.175, 0.425)
    b.set_xticks(range(len(qs)))
    b.set_xticklabels([f"{q:.1f}".replace("-", "\u2212") for q in qs])
    b.set_yticks([0.2, 0.25, 0.3, 0.35, 0.4])
    b.set_xlabel(r"$\varepsilon K$", fontsize=8.6, color=INK)
    b.set_ylabel(r"$\varepsilon$", fontsize=8.6, color=INK)
    style(b)
    b.grid(False)
    for ax, lab in ((a, "(a)"), (b, "(b)")):
        ax.text(-.02, 1.02, lab, transform=ax.transAxes, fontsize=9, va="bottom", ha="right")
    fig.tight_layout(w_pad=1.5)
    print("fig4: sustained N=50", sum(v[0] for v in cnt.values()), "of", sum(v[1] for v in cnt.values()))
    save(fig, "fig4_partitions")


def fig5():
    from scipy.sparse.csgraph import connected_components
    from paper3_aggregates import DT, SC, TARGET
    ag = json.loads((OUT / "aggregates.json").read_text())
    c3, fit = ag["claim3"], ag["claim3"]["fits"]["primary"]
    lp = ag["claim3_direction"]["floquet_O"]["internal"][0]
    assert lp["size"] == 137
    lperp = lp["exponents"][0]
    fig, (a, b) = plt.subplots(1, 2, figsize=(6.8, 2.7))
    nfit = 0
    for v in c3["per_eta"].values():
        x = v["ln_inv_eta"]
        for r in v["primary"]:
            cens = r["left_censored"] or r["right_censored"]
            used = not cens and 1e-14 <= v["eta"] <= 1e-8
            c = SER[0] if v["eta"] >= 1e-14 and v["eta"] <= 1e-8 else MUTED
            a.plot(x, r["L"], "o", ms=5 if used else 4.5, color=c, mfc=c if not cens else "white",
                   mec=c if cens else "white", mew=.9 if cens else .6, zorder=4 if used else 3)
            nfit += used
    assert nfit == fit["n"]
    xx = np.array([12, 34])
    a.plot(xx, fit["intercept"] + fit["slope"] * xx, color=INK, lw=.9)
    a.text(.04, .93, rf"slope $b={fit['slope']:.0f}$" "\n" rf"$1/\lambda_\perp={1 / lperp:.0f}$", transform=a.transAxes,
           fontsize=7.8, va="top", color=INK)
    a.plot([], [], "o", ms=4.5, color=SER[0], mec="white", label="used in fit")
    a.plot([], [], "o", ms=4.5, color=SER[0], mfc="white", label="censored")
    a.plot([], [], "o", ms=4.5, color=MUTED, mec="white", label=r"$\eta=10^{-6}$")
    a.legend(fontsize=7, frameon=False, loc="lower right")
    a.set_xlabel(r"$\ln(1/\eta)$", fontsize=8.6, color=INK)
    a.set_ylabel(r"residence $L$", fontsize=8.6, color=INK)
    style(a)
    t0, t1 = 20770, 41170
    zs = np.load(SC / "rn_0088_n1e-14_t80000_states_z.npy", mmap_mode="r")
    n = zs.shape[1] // 2
    k0, k1 = t0 // DT - 1, min(t1 // DT - 1 + 150, len(zs))
    z = zs[k0]
    P = np.c_[z[:n], z[n:]]
    lab = connected_components(np.max(abs(P[:, None] - P[None]), axis=2) < 1e-3, directed=False)[1]
    cnt = np.bincount(lab)
    mem = {sz: np.nonzero(lab == int(np.nonzero(cnt == sz)[0][0]))[0] for sz in TARGET}
    nt = (k1 - k0) // 3
    tt = np.array([DT * (k0 + 1) + DT * (3 * i + 1) for i in range(nt)], float)
    for sz, col in zip(TARGET, SER):
        S = []
        for k in range(k0, k0 + 3 * nt):
            Q = np.c_[zs[k][mem[sz]], zs[k][n + mem[sz]]]
            S.append(np.median(np.max(abs(Q - np.median(Q, axis=0)), axis=1)))
        lg = np.log10(np.array(S).reshape(nt, 3).min(axis=1))
        b.plot(tt, lg, color=col, lw=.9, label=f"{sz} cluster")
    ta = np.array([t0 + 1500, t0 + 14500])
    b.plot(ta, -9.6 + lperp * (ta - ta[0]) / np.log(10), "--", color=INK, lw=.9)
    b.text(t0 + 1000, -7.2, r"slope $\lambda_\perp$", fontsize=7.8, color=INK)
    b.axhline(-3, color=INK2, lw=.8, ls=":")
    b.axvline(t1, color=INK2, lw=.8)
    b.legend(fontsize=7, frameon=False, loc="center right")
    b.set_xlabel("time", fontsize=8.6, color=INK)
    b.set_ylabel(r"$\log_{10}$ spread", fontsize=8.6, color=INK)
    style(b)
    for ax, lab_ in ((a, "(a)"), (b, "(b)")):
        ax.text(-.02, 1.02, lab_, transform=ax.transAxes, fontsize=9, va="bottom", ha="right")
    fig.tight_layout(w_pad=1.5)
    save(fig, "fig5_run88")


def fig7():
    NLD = ROOT / "analysis/noiseless_n50"
    files = ["stageB_classify.json", "stageB_classify_mid.json", "stageB_classify_mid2_horse.json",
             "stageB_classify_mid2_boar.json", "stageC_classify_horse.json", "stageC_classify_boar.json"]
    runs = []
    for f in files:
        for r in json.loads((NLD / f).read_text())["runs"]:
            if "r1600" not in r["file"]:
                continue
            if "h0.0025" in r["file"]:
                prot, eta = "C-h", r["eta"]
            elif f.startswith("stageC"):
                prot, eta = "C-k", float(f"{r['eta'] * np.sqrt(10):.2g}")
            else:
                prot, eta = "B", r["eta"]
            runs.append((eta, prot, r))
    porder = {"B": 0, "C-h": 1, "C-k": 2}
    runs.sort(key=lambda x: (x[0], porder[x[1]], x[2].get("seed") or 0))
    short = {"Deterministic-cycle-like": "D", "Fragmented-switching": "F", "Other": "O", "Other/unresolved": "O"}
    lcol = {"D": SER[0], "F": SER[1], "c": GRID}
    fig, ax = plt.subplots(figsize=(6.8, 5.6))
    y, prev, ticks = 0, None, []
    for eta, prot, r in runs:
        if (eta, prot) != prev:
            if prev is not None:
                y += .8
            ticks.append([y, (eta, prot), 0])
            prev = (eta, prot)
        cy = r["cycles"]
        for k, c in enumerate(cy):
            t0 = c["t_split"]
            t1 = cy[k + 1]["t_split"] if k + 1 < len(cy) else c["t_split"] + c["T"]
            ax.add_patch(plt.Rectangle((t0, y - .42), t1 - t0, .84, facecolor=lcol[c["label"]], edgecolor="white",
                                       lw=.4))
        ticks[-1][2] += 1
        ticks[-1].append(short[r["classification"]])
        y += 1
    ax.axvline(150000, color=INK2, lw=.7, ls=":")
    ax.set_xlim(0, 300000)
    ax.set_ylim(y - .2, -.8)
    pos, labs = [], []
    for y0, (eta, prot), k, *cl in ticks:
        pos.append(y0 + (k - 1) / 2)
        summ = ", ".join(f"{cl.count(q)} {q}" for q in "DFO" if cl.count(q))
        ax.text(304000, y0 + (k - 1) / 2, summ, fontsize=7, va="center", ha="left", color=INK, clip_on=False)
        e = "0" if eta == 0 else (f"{eta:.1e}".replace("e-", r"\times10^{-").replace("1.0\\times", "") + "}")
        labs.append(rf"${e}$ {prot}" if prot != "B" else rf"${e}$")
    ax.set_yticks(pos)
    ax.set_yticklabels(labs, fontsize=7)
    ax.set_xlabel("time", fontsize=8.6, color=INK)
    ax.set_ylabel(r"$\eta$ (protocol)", fontsize=8.6, color=INK)
    for k, lab in (("D", "D cycle"), ("F", "F cycle"), ("c", "other cycle")):
        ax.add_patch(plt.Rectangle((0, 0), 0, 0, facecolor=lcol[k], label=lab))
    ax.legend(fontsize=7, frameon=False, loc="upper center", bbox_to_anchor=(.5, 1.06), ncol=3)
    style(ax)
    ax.grid(False)
    ax.tick_params(axis="y", length=0)
    cnt = {}
    for eta, prot, r in runs:
        cnt.setdefault((eta, prot), []).append(short[r["classification"]])
    for k, v in cnt.items():
        print("fig7", k, "D", v.count("D"), "F", v.count("F"), "O", v.count("O"))
    print("fig7 runs", len(runs))
    save(fig, "fig7_selection")


def fig2():
    from matplotlib.patches import FancyArrowPatch
    fig, (a, b) = plt.subplots(1, 2, figsize=(6.8, 3.0), gridspec_kw=dict(width_ratios=[1, 1.15]))
    for ax in (a, b):
        ax.set_aspect("equal")
        ax.axis("off")

    def arrow(ax, p, q, rad=0., ls="-", col=INK, lw=1.0, ms=9):
        ax.add_patch(FancyArrowPatch(p, q, connectionstyle=f"arc3,rad={rad}", arrowstyle="-|>", mutation_scale=ms,
                                     lw=lw, ls=ls, color=col, shrinkA=9, shrinkB=9))

    ang = {1: 90, 2: -30, 3: 210}  # firing order clockwise
    pos = {k: (np.cos(np.radians(v)), np.sin(np.radians(v))) for k, v in ang.items()}
    th = np.linspace(0, 2 * np.pi, 200)
    a.plot(np.cos(th), np.sin(th), color=GRID, lw=1.2, zorder=0)
    for k in (1, 2, 3):
        x, y = pos[k]
        u = k == 1
        a.plot(x, y, "o", ms=17, color=SER[1] if u else SER[0], mec="white", mew=1, zorder=3)
        a.text(x, y, str(k), ha="center", va="center", fontsize=8.5, color="white", zorder=4, fontweight="bold")
    a.text(pos[1][0], pos[1][1] + .2, "U", ha="center", va="bottom", fontsize=9, color=INK)
    arrow(a, (np.cos(np.radians(65)) * 1.25, np.sin(np.radians(65)) * 1.25),
          (np.cos(np.radians(35)) * 1.25, np.sin(np.radians(35)) * 1.25), rad=-.2, col=INK2, lw=.8, ms=7)
    a.text(1.0, 1.05, "firing\norder", fontsize=7, color=INK2, ha="left", va="bottom")
    # leading branch: one element of U moves back to cluster 3
    lx, ly = np.cos(np.radians(110)) * .78, np.sin(np.radians(110)) * .78
    a.plot(lx, ly, "o", ms=5, color=SER[1], mec="white", mew=.6, zorder=3)
    arrow(a, (lx, ly), (pos[3][0] + .05, pos[3][1] + .12), rad=.25, col=INK, lw=1.0)
    a.text(-1.35, .55, "leading branch:\njoins cluster 3\n\u2192 copy of type A,\nU passes to 2", fontsize=7, ha="center",
           va="center", color=INK)
    # lagging branch: element stays behind U
    gx, gy = np.cos(np.radians(70)) * .8, np.sin(np.radians(70)) * .8
    a.plot(gx, gy, "o", ms=5, color=SER[1], mec="white", mew=.6, zorder=3)
    a.plot([gx + .06, 1.0], [gy - .05, .45], color=INK2, lw=.5)
    a.text(0, -1.3, "type A orbit", fontsize=7.5, ha="center", va="center", color=INK2)
    a.text(1.05, .3, "lagging branch:\nstays behind U\nas an isolated element\n\u2192 type B", fontsize=7, ha="left",
           va="center", color=INK)
    a.set_xlim(-1.9, 1.9)
    a.set_ylim(-1.35, 1.45)
    # (b) classes of copies
    R = 1.0
    A = {k: (R * np.cos(np.radians(v)), R * np.sin(np.radians(v))) for k, v in ang.items()}
    Bp = {k: (2.0 * np.cos(np.radians(v - 40)), 2.0 * np.sin(np.radians(v - 40))) for k, v in ang.items()}
    for k in (1, 2, 3):
        k1 = k % 3 + 1
        arrow(b, A[k], A[k1], rad=-.25)
        arrow(b, A[k], Bp[k], rad=.0, ls="--", col=INK2)
        arrow(b, Bp[k], A[k1], rad=-.3)
    for k in (1, 2, 3):
        b.plot(*A[k], "o", ms=19, color=SER[0], mec="white", mew=1, zorder=3)
        b.text(*A[k], rf"$\mathrm{{A}}_{k}$", ha="center", va="center", fontsize=8.5, color="white", zorder=4)
        b.plot(*Bp[k], "s", ms=17, color=SER[2], mec="white", mew=1, zorder=3)
        b.text(*Bp[k], rf"$\mathrm{{B}}_{k}$", ha="center", va="center", fontsize=8.5, color="white", zorder=4)
    b.plot([], [], "-", color=INK, label="leading branch")
    b.plot([], [], "--", color=INK2, label="lagging branch")
    b.legend(fontsize=7, frameon=False, loc="lower center", bbox_to_anchor=(.5, -.08), ncol=2)
    b.set_xlim(-2.5, 2.5)
    b.set_ylim(-2.3, 2.3)
    fig.tight_layout(w_pad=.5)
    for ax, lab in ((a, "(a)"), (b, "(b)")):
        fig.text(ax.get_position().x0, .95, lab, fontsize=9, va="top", ha="left")
    save(fig, "fig2_network")


def figS1():
    runs = [r for r in json.loads((ROOT / "analysis/noiseless_n50/stageB_classify.json").read_text())["runs"]
            if r["eta"] == 0]
    fig, (a, b) = plt.subplots(1, 2, figsize=(6.8, 2.6))
    for r, c, lab in zip(runs, SER, ("from S8", "from S6")):
        assert ("r1600" in r["file"]) == (lab == "from S8")
        cy = r["cycles"]
        k = np.arange(1, len(cy) + 1)
        d = np.array([x["label"] == "D" for x in cy])
        T = np.array([x["T"] for x in cy])
        a.plot(k, T, "-", color=c, lw=.8)
        a.plot(k[d], T[d], "o", ms=4, color=c, mec="white", mew=.6, label=lab)
        a.plot(k[~d], T[~d], "o", ms=4, color=c, mfc="white", mew=.9)
        for key, mk in (("depth132small", "o"), ("depth137small", "s")):
            v = np.array([np.nan if x[key] is None else x[key] for x in cy])
            b.plot(k, v, "-", color=c, lw=.8)
            b.plot(k[d], v[d], mk, ms=4, color=c, mec="white", mew=.6)
            b.plot(k[~d], v[~d], mk, ms=4, color=c, mfc="white", mew=.9)
    a.legend(fontsize=7, frameon=False, loc="lower right")
    for ax in (a, b):
        ax.set_xticks([1, 5, 10, 15])
    a.set_xlabel("cycle", fontsize=8.6, color=INK)
    a.set_ylabel("cycle length", fontsize=8.6, color=INK)
    b.text(15.3, -16.0, "5 to 127", fontsize=7, va="center", ha="left", color=INK2)
    b.text(15.3, -6.6, "12 (14) to\n125 (123)", fontsize=7, va="center", ha="left", color=INK2)
    b.set_xlim(.5, 18.5)
    b.set_xlabel("cycle", fontsize=8.6, color=INK)
    b.set_ylabel(r"$\log_{10}$ closest approach", fontsize=8.6, color=INK)
    style(a)
    style(b)
    for ax, lab in ((a, "(a)"), (b, "(b)")):
        ax.text(-.02, 1.02, lab, transform=ax.transAxes, fontsize=9, va="bottom", ha="right")
    fig.tight_layout(w_pad=1.5)
    save(fig, "figS1_noiseless88")


FIGS = dict(figS1=figS1, fig1=fig1, fig2=fig2, fig3=fig3, fig4=fig4, fig5=fig5, fig7=fig7)

if __name__ == "__main__":
    for name in sys.argv[1:] or FIGS:
        FIGS[name]()
