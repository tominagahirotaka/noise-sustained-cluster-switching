"""Three-cluster branches at eps = 0.2 with (beta, gamma) = (0.7, 0.5) (excitable elements).

Method as in manuscript Sec. III (scripts/shoot_cluster_boundaries.py): fixed populations, shooting with
phase condition X_1 = 0, damped Newton, full collective/internal Floquet audit.

  homotopy  : carry branches A (31:39:30), B (12:43:45), C (34:33:33) from eps = 0.3 to 0.2 at fixed K
              (A, C at K = -2; B at K = -6.5, inside their eps = 0.3 stable intervals).
  ensemble  : regenerate random-initial-condition final states at eps = 0.2 (N = 100, K = -3, RK4 h = 0.005,
              8e6 steps, seeds 1001-1024) and shoot the three-cluster states.
  scan      : at eps = 0.2, follow one branch in K in both directions from its start, auditing every point
              and bisecting every change of full-system stability to width 5e-4.
"""
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import argparse
import json
import subprocess

import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import brentq

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "analysis/eps0p2_three_cluster"
BETA, GAMMA = 0.7, 0.5


def k1(eps):
    x = brentq(lambda x: x - x**3 / 3 - (x + BETA) / GAMMA, -3, 0)
    return (1 - x * x) / eps - eps * GAMMA


class Partition:
    def __init__(self, sizes):
        self.sizes = np.asarray(sizes)
        self.p = self.sizes / self.sizes.sum()
        self.m = len(sizes)
        self.n = 2 * self.m

    def rhs(self, z, K, e):
        m, p = self.m, self.p
        x, y = z[:m], z[m:]
        return np.r_[(x - x**3 / 3 - y) / e + K * (p @ x - x), e * (x + BETA - GAMMA * y)]

    def jac(self, z, K, e):
        m, n, p = self.m, self.n, self.p
        J = np.zeros((n, n))
        J[:m, :m] = np.diag((1 - z[:m] ** 2) / e - K) + K * np.ones((m, 1)) * p
        J[:m, m:] = -np.eye(m) / e
        J[m:, :m] = e * np.eye(m)
        J[m:, m:] = -e * GAMMA * np.eye(m)
        return J

    def flow(self, w, K, e, tol):
        n, T = self.n, w[-1]
        if not 1 < T < 300:
            return None

        def f(t, v):
            return np.r_[self.rhs(v[:n], K, e), (self.jac(v[:n], K, e) @ v[n:].reshape(n, n)).ravel()]

        s = solve_ivp(f, (0, T), np.r_[w[:n], np.eye(n).ravel()], method="DOP853", rtol=tol, atol=tol * .01,
                      max_step=.08)
        if not s.success:
            return None
        z, M = s.y[:n, -1], s.y[n:, -1].reshape(n, n)
        A = np.zeros((n + 1, n + 1))
        A[:n, :n] = M - np.eye(n)
        A[:n, -1] = self.rhs(z, K, e)
        A[-1, 0] = 1
        return np.r_[z - w[:n], w[0]], A

    def shoot(self, w, K, e, tol):
        w = w.copy()
        for _ in range(16):
            r = self.flow(w, K, e, tol)
            if r is None:
                return None
            res, A = r
            err = np.max(abs(res))
            if err < 2e-9:
                return w
            d = np.linalg.lstsq(A, -res, rcond=1e-12)[0]
            for s in [1, .5, .25, .125, .0625, .03125]:
                rr = self.flow(w + s * d, K, e, tol)
                if rr is not None and np.max(abs(rr[0])) < err:
                    w = w + s * d
                    break
            else:
                return None
        return None

    def audit(self, w, K, e, tol):
        m, n = self.m, self.n
        T, z = w[-1], w[:n]

        def f(t, v):
            zz = v[:n]
            A = np.zeros((m, 2, 2))
            A[:, 0, 0] = (1 - zz[:m] ** 2) / e - K
            A[:, 0, 1] = -1 / e
            A[:, 1, 0] = e
            A[:, 1, 1] = -e * GAMMA
            return np.r_[self.rhs(zz, K, e), (self.jac(zz, K, e) @ v[n:n + n * n].reshape(n, n)).ravel(),
                         (A @ v[n + n * n:].reshape(m, 2, 2)).ravel()]

        s = solve_ivp(f, (0, T), np.r_[z, np.eye(n).ravel(), np.tile(np.eye(2), (m, 1, 1)).ravel()],
                      method="DOP853", rtol=max(tol * .1, 2.3e-14), atol=tol * .001, max_step=.04,
                      dense_output=True)
        assert s.success
        v = s.y[:, -1]
        M, B = v[n:n + n * n].reshape(n, n), v[n + n * n:].reshape(m, 2, 2)
        eig = np.linalg.eigvals(M)
        neut = int(np.argmin(abs(eig - 1)))
        ce, be = np.delete(eig, neut), np.linalg.eigvals(B)
        cm, im = float(max(abs(ce))), np.max(abs(be), axis=1)
        t = np.linspace(0, T, 2001)
        tr = s.sol(t)[:n]
        x, y = tr[:m], tr[m:]
        d = np.max(np.maximum(abs(x[:, None, :] - x[None, :, :]), abs(y[:, None, :] - y[None, :, :])), axis=2)
        dist = float(d[np.triu_indices(m, 1)].min())
        ret, neutral = float(max(abs(v[:n] - z))), float(abs(eig[neut] - 1))
        g = self.rhs(z, K, e)
        tangent = float(np.linalg.norm(M @ g - g) / np.linalg.norm(g))
        valid = ret < 1e-7 and neutral < 1e-6 and tangent < 1e-6 and dist > 1e-4
        kind = "collective" if cm >= max(im) else "internal"
        j = int(np.argmax(im))
        dom = ce[np.argmax(abs(ce))] if kind == "collective" else be[j, np.argmax(abs(be[j]))]
        return dict(eps=float(e), K=float(K), rtol=tol, period=float(T), sizes=self.sizes.tolist(),
                    collective_max=cm, internal_max=im.tolist(), max_all=max(cm, float(max(im))),
                    dominant_kind=kind, dominant_population=int(self.sizes[j]) if kind == "internal" else None,
                    dominant_eigenvalue=[float(dom.real), float(dom.imag)], return_error=ret,
                    neutral_error=neutral, tangent_error=tangent, min_pair_cycle_distance=dist,
                    valid=bool(valid), stable=bool(valid and max(cm, float(max(im))) < 1))


SOURCES = {  # branch: (saved eps = 0.3 orbit, K)
    "A": ("analysis/three_cluster_stability/A_dir1_tol2e-12_step0.05_kmax0/point000.npz", -2.0),
    "B": ("analysis/three_cluster_stability/B_dir1_tol2e-12_step0.05/point000.npz", -6.5),
    "C": ("analysis/three_cluster_stability/C_dir1_tol2e-12_step0.025_kmax0/point000.npz", -2.0),
}


def arc(name, ds=.002, tol=2e-12, k_from=-.2, max_points=600, k_step=.05):
    """Pseudo-arclength continuation in (z, T, K) on the weak-coupling side, with full Floquet audit.

    Natural continuation (step 0.05) brings the branch from its start to K = k_from; arclength steps of
    size ds in (z, T, K) then follow the branch through turning points.
    """
    d = np.load(OUT / f"start_{name}.npz")
    part, e = Partition(d["sizes"]), .2
    n = part.n
    folder = OUT / f"arc_{name}_ds{ds:g}_from{k_from:g}"
    folder.mkdir(parents=True, exist_ok=True)
    K, w = float(d["K"]), part.shoot(d["w"], float(d["K"]), e, tol)
    while K < k_from - 1e-12:
        K = min(k_from, K + k_step)
        w = part.shoot(w, K, e, tol)
        assert w is not None
    w2 = part.shoot(w, K + .002, e, tol)
    Y0, Y1 = np.r_[w, K], np.r_[w2, K + .002]

    def evaluate(Y):
        z, T, Kc = Y[:n], Y[n], Y[-1]

        def f(t, v):
            zz = v[:n]
            J = part.jac(zz, Kc, e)
            g = np.r_[part.p @ zz[:part.m] - zz[:part.m], np.zeros(part.m)]  # d rhs / d K
            return np.r_[part.rhs(zz, Kc, e), (J @ v[n:n + n * n].reshape(n, n)).ravel(), J @ v[-n:] + g]

        s = solve_ivp(f, (0, T), np.r_[z, np.eye(n).ravel(), np.zeros(n)], method="DOP853", rtol=tol,
                      atol=tol * .01, max_step=.05)
        assert s.success
        v = s.y[:, -1]
        A = np.zeros((n + 1, n + 2))
        A[:n, :n] = v[n:n + n * n].reshape(n, n) - np.eye(n)
        A[:n, n] = part.rhs(v[:n], Kc, e)
        A[:n, -1] = v[-n:]
        A[n, 0] = 1
        return np.r_[v[:n] - z, Y[0]], A

    rows = []
    for j in range(max_points):
        tangent = (Y1 - Y0) / np.linalg.norm(Y1 - Y0)
        pred = Y1 + ds * tangent
        Y = pred.copy()
        for _ in range(12):
            res, A = evaluate(Y)
            r = np.r_[res, (Y - pred) @ tangent]
            if max(abs(r)) < 5e-10:
                break
            Y = Y + np.linalg.solve(np.vstack([A, tangent]), -r)
        else:
            rows.append(dict(status="arc_newton_failure", after_K=float(Y1[-1])))
            break
        row = part.audit(Y[:-1], Y[-1], e, tol)
        row["arc_index"] = j
        rows.append(row)
        np.savez(folder / f"point{j:03d}.npz", Y=Y)
        if j % 10 == 0 or not row["stable"]:
            print(j, {k: row[k] for k in ("K", "period", "collective_max", "max_all", "dominant_kind",
                                         "dominant_eigenvalue", "stable", "valid")}, flush=True)
        (folder / "results.json").write_text(json.dumps(rows, indent=2))
        if not row["valid"] or Y[-1] > -.002 or Y[-1] < -1.0:
            break
        Y0, Y1 = Y1, Y
    (folder / "results.json").write_text(json.dumps(rows, indent=2))
    print("ARC_DONE", name, len(rows), rows[-1].get("K"), rows[-1].get("status"), flush=True)


def eps_slice(eps, ds=.004, tol=2e-12, max_points=4000, k_arc_start=-.30):
    """Branch C (34:33:33) at fixed eps: weak-side pseudo-arclength in (z, T, K).

    C is carried from its eps = 0.3, K = -2 orbit to `eps` at K = -2, continued naturally to K = k_arc_start,
    then followed by arclength. Every point records the nontrivial collective multipliers (from the monodromy of
    the converged Newton step); the internal blocks are audited every 10 points. Collective crossings of -1 and
    +1 and turning points in K are listed in the summary.
    """
    src, K = SOURCES["C"]
    d = np.load(ROOT / src)
    part = Partition(d["sizes"])
    n = part.n
    folder = OUT / f"eps_slice/C_eps{eps:.3f}"
    folder.mkdir(parents=True, exist_ok=True)
    w = part.shoot(d["w"], K, .3, tol)
    e = .3
    while abs(e - eps) > 1e-12:  # eps homotopy from 0.3, downward or upward
        e2 = max(eps, e - .005) if eps < e else min(eps, e + .005)
        w2 = part.shoot(w, K, e2, tol)
        assert w2 is not None, ("eps homotopy failed", e2)
        e, w = e2, w2
    while K < k_arc_start - 1e-12:
        K2 = min(k_arc_start, K + .05)
        w2 = part.shoot(w, K2, eps, tol)
        assert w2 is not None, ("K continuation failed", K2)
        K, w = K2, w2
    wb = part.shoot(w, K + .002, eps, tol)
    Y0, Y1 = np.r_[w, K], np.r_[wb, K + .002]

    def evaluate(Y):
        z, T, Kc = Y[:n], Y[n], Y[-1]
        assert 1 < T < 300  # reject runaway Newton iterates (suspected cause of the OOM stops of 2026-10-21)

        def f(t, v):
            zz = v[:n]
            J = part.jac(zz, Kc, eps)
            g = np.r_[part.p @ zz[:part.m] - zz[:part.m], np.zeros(part.m)]
            return np.r_[part.rhs(zz, Kc, eps), (J @ v[n:n + n * n].reshape(n, n)).ravel(), J @ v[-n:] + g]

        s = solve_ivp(f, (0, T), np.r_[z, np.eye(n).ravel(), np.zeros(n)], method="DOP853", rtol=tol,
                      atol=tol * .01, max_step=.05)
        assert s.success
        v = s.y[:, -1]
        M = v[n:n + n * n].reshape(n, n)
        A = np.zeros((n + 1, n + 2))
        A[:n, :n] = M - np.eye(n)
        A[:n, n] = part.rhs(v[:n], Kc, eps)
        A[:n, -1] = v[-n:]
        A[n, 0] = 1
        return np.r_[v[:n] - z, Y[0]], A, M

    rows = []
    ds0, j = ds, 0
    while j < max_points:
        tangent = (Y1 - Y0) / np.linalg.norm(Y1 - Y0)
        pred = Y1 + ds * tangent
        Y = pred.copy()
        ok = False
        for _ in range(12):
            try:
                res, A, M = evaluate(Y)
            except AssertionError:
                break
            r = np.r_[res, (Y - pred) @ tangent]
            if max(abs(r)) < 5e-10:
                ok = True
                break
            Y = Y + np.linalg.solve(np.vstack([A, tangent]), -r)
        if not ok:
            ds /= 2
            if ds < 1e-6:
                rows.append(dict(status="arc_newton_failure", after_K=float(Y1[-1])))
                break
            continue
        eig = np.linalg.eigvals(M)
        eig = np.delete(eig, int(np.argmin(abs(eig - 1))))
        eig = eig[np.argsort(-abs(eig))]
        row = dict(j=j, K=float(Y[-1]), period=float(Y[n]), ds=ds, return_error=float(max(abs(res))),
                   collective=[[float(x.real), float(x.imag)] for x in eig[:3]])
        if j % 10 == 0:
            au = part.audit(Y[:-1], Y[-1], eps, tol)
            row.update(internal_max=max(au["internal_max"]), valid=au["valid"])
        rows.append(row)
        if j % 50 == 0:
            np.savez(folder / f"point{j:04d}.npz", Y=Y)
            (folder / "results.json").write_text(json.dumps(rows, indent=2))
        if Y[-1] > -.002 or Y[-1] < -.8:
            break
        Y0, Y1 = Y1, Y
        ds = min(ds0, ds * 1.5)
        j += 1
    (folder / "results.json").write_text(json.dumps(rows, indent=2))
    pts = [r for r in rows if "K" in r]
    events = []
    for a, b in zip(pts, pts[1:]):
        ma = max(abs(complex(*c)) for c in a["collective"])
        for x, y in [(a, b)]:
            ra = [complex(*c) for c in x["collective"]]
            rb = [complex(*c) for c in y["collective"]]
            neg_a = any(abs(c.imag) < 1e-8 and c.real < -1 for c in ra)
            neg_b = any(abs(c.imag) < 1e-8 and c.real < -1 for c in rb)
            pos_a = any(abs(c.imag) < 1e-8 and c.real > 1 for c in ra)
            pos_b = any(abs(c.imag) < 1e-8 and c.real > 1 for c in rb)
            cx_a = any(abs(c.imag) >= 1e-8 and abs(c) > 1 for c in ra)
            cx_b = any(abs(c.imag) >= 1e-8 and abs(c) > 1 for c in rb)
            if neg_a != neg_b:
                events.append(dict(type="collective_-1", j=b["j"], K=b["K"], period=b["period"], entering=neg_b))
            if pos_a != pos_b:
                events.append(dict(type="collective_+1", j=b["j"], K=b["K"], period=b["period"], entering=pos_b))
            if cx_a != cx_b:
                events.append(dict(type="collective_complex", j=b["j"], K=b["K"], period=b["period"], entering=cx_b))
        del ma
    for a, b, c in zip(pts, pts[1:], pts[2:]):
        if (b["K"] - a["K"]) * (c["K"] - b["K"]) < 0:
            events.append(dict(type="K_turn", j=b["j"], K=b["K"], period=b["period"],
                               kind="max" if b["K"] > a["K"] else "min"))
    events.sort(key=lambda x: x["j"])
    summary = dict(branch="C", eps=eps, ds=ds, rtol=tol, points=len(pts), end=rows[-1].get("status", "range"),
                   K_end=pts[-1]["K"] if pts else None, events=events)
    (folder / "summary.json").write_text(json.dumps(summary, indent=2))
    print("SLICE", json.dumps(summary), flush=True)


SEARCH_PARTITIONS = [(12, 43, 45), (10, 45, 45), (15, 40, 45), (20, 40, 40), (25, 37, 38), (20, 30, 50),
                     (30, 39, 31)]
SEARCH_K = [-1., -2., -3., -4., -5., -6., -7., -8., -9., -10., -11., -12.]


def _search_one(args):
    sizes, K, seed = args
    part = Partition(sizes)
    rng = np.random.Generator(np.random.PCG64(seed))
    z0 = np.r_[rng.uniform(-2.5, 2.5, 3), rng.uniform(-2.0, 2.0, 3)]

    def ev(t, z):
        return z[0]
    ev.direction = 1
    s = solve_ivp(lambda t, z: part.rhs(z, K, .2), (0, 3000), z0, method="DOP853", rtol=1e-10, atol=1e-12,
                  max_step=.05, events=ev)
    ts, ys = s.t_events[0], s.y_events[0]
    out = dict(sizes=list(sizes), K=K, seed=seed)
    if len(ts) < 6:
        return dict(out, status="no_oscillation")
    gap = min(max(abs(ys[-1][[a, a + 3]] - ys[-1][[b, b + 3]])) for a, b in ((0, 1), (0, 2), (1, 2)))
    if gap < 1e-4:
        return dict(out, status="merged_at_section")
    for lag in range(1, 5):
        if len(ts) > lag + 3 and max(abs(ys[-1] - ys[-1 - lag])) < 1e-6:
            w = part.shoot(np.r_[ys[-1 - lag], ts[-1] - ts[-1 - lag]], K, .2, 2e-12)
            if w is None:
                return dict(out, status="shooting_failed", lag=lag)
            r = part.audit(w, K, .2, 2e-12)
            return dict(out, status="periodic", lag=lag, w=w.tolist(), **{k: r[k] for k in (
                "period", "collective_max", "internal_max", "max_all", "dominant_kind", "dominant_population",
                "dominant_eigenvalue", "min_pair_cycle_distance", "valid", "stable")})
    return dict(out, status="no_recurrence")


def search():
    tasks = [(s, K, 5000 + i) for s in SEARCH_PARTITIONS for K in SEARCH_K for i in range(6)]
    with ProcessPoolExecutor(16) as pool:
        rows = list(pool.map(_search_one, tasks, chunksize=2))
    (OUT / "search_unbalanced.json").write_text(json.dumps(rows, indent=2))
    for s in SEARCH_PARTITIONS:
        for K in SEARCH_K:
            sel = [r for r in rows if tuple(r["sizes"]) == s and r["K"] == K]
            per = [r for r in sel if r["status"] == "periodic"]
            stab = [r for r in per if r["stable"]]
            print(s, K, f"periodic {len(per)}/{len(sel)} fullstable {len(stab)}",
                  sorted({round(r["period"], 4) for r in stab}), flush=True)


def homotopy_fixed_q(name, tol=2e-12):
    """Carry a branch from eps = 0.3 to 0.2 at fixed relative coupling q = eps*K (fast-time coupling)."""
    src, K0 = SOURCES[name]
    d = np.load(ROOT / src)
    part, q = Partition(d["sizes"]), .3 * K0
    w = part.shoot(d["w"], K0, .3, tol)
    rows = [dict(part.audit(w, K0, .3, tol), branch=name, q=q)]
    e, step, old = .3, .005, None
    while e > .2 + 1e-12:
        target = max(.2, e - step)
        guess = w if old is None else w + (w - old[1]) * (target - e) / (e - old[0])
        cand = part.shoot(guess, q / target, target, tol)
        if cand is None:
            step /= 2
            if step < 1e-5:
                rows.append(dict(branch=name, status="homotopy_failed", eps=e, K=q / e))
                break
            continue
        r = dict(part.audit(cand, q / target, target, tol), branch=name, q=q)
        rows.append(r)
        if not r["valid"]:
            break
        old, e, w = (e, w), target, cand
    if abs(e - .2) < 1e-12:
        np.savez(OUT / f"start_{name}.npz", w=w, K=q / .2, sizes=part.sizes, eps=.2)
    (OUT / f"homotopy_fixed_q_{name}.json").write_text(json.dumps(rows, indent=2))
    for r in rows[::4] + [rows[-1]]:
        print({k: r.get(k) for k in ("eps", "K", "period", "max_all", "dominant_kind", "dominant_population",
                                     "stable", "status")}, flush=True)


def homotopy(tol=2e-12):
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    for name, (src, K) in SOURCES.items():
        d = np.load(ROOT / src)
        assert abs(float(d["K"]) - K) < 1e-12
        part = Partition(d["sizes"])
        w = part.shoot(d["w"], K, .3, tol)
        assert w is not None
        r0 = part.audit(w, K, .3, tol)
        r0.update(branch=name)
        rows.append(r0)
        e, step, old = .3, .005, None
        while e > .2 + 1e-12:
            target = max(.2, e - step)
            guess = w if old is None else w + (w - old[1]) * (target - e) / (e - old[0])
            cand = part.shoot(guess, K, target, tol)
            if cand is None:
                step /= 2
                if step < 1e-5:
                    rows.append(dict(branch=name, status="homotopy_failed", eps=e, K=K))
                    break
                continue
            r = part.audit(cand, K, target, tol)
            r.update(branch=name)
            rows.append(r)
            if not r["valid"]:
                break
            old, e, w = (e, w), target, cand
        if abs(e - .2) < 1e-12:
            np.savez(OUT / f"start_{name}.npz", w=w, K=K, sizes=part.sizes, eps=.2)
        last = rows[-1]
        print(name, part.sizes.tolist(), f"eps={last.get('eps')} K={K} T={last.get('period')}",
              f"max={last.get('max_all')}", last.get("dominant_kind"), last.get("dominant_population"),
              "stable" if last.get("stable") else "NOT STABLE",
              "| unstable along homotopy at eps:", [round(r["eps"], 4) for r in rows
                                                     if r.get("branch") == name and not r.get("stable", True)][:5],
              flush=True)
    (OUT / "homotopy.json").write_text(json.dumps(rows, indent=2))


def _run_seed(args):
    exe, seed = args
    rng = np.random.Generator(np.random.PCG64(seed))
    pts = np.column_stack((rng.uniform(-2.5, 2.5, 100), rng.uniform(-2.0, 2.0, 100)))
    out = subprocess.run([str(exe), "-3", "0.005", "8000000", "400000", "0"],
                         input="".join(f"{x:.17g} {y:.17g}\n" for x, y in pts), text=True,
                         capture_output=True, check=True).stdout.splitlines()
    c = next(l for l in out if l.startswith("component_centroids,")).split(",")
    m, v = int(c[2]), c[3:]
    tail = [int(l.split(",")[5]) for l in out if len(l.split(",")) == 7 and l.split(",")[0].isdigit()][-5:]
    return dict(seed=seed, clusters=m, X=[float(v[3 * a]) for a in range(m)],
                Y=[float(v[3 * a + 1]) for a in range(m)], sizes=[int(v[3 * a + 2]) for a in range(m)],
                checkpoint_tail=tail)


def ensemble(tol=2e-12):
    OUT.mkdir(parents=True, exist_ok=True)
    exe = OUT / "fhn_e0p2_b0p7_g0p5"
    subprocess.run(["gcc", "-O3", "-std=c11", "-DN=100", "-DEPS=0.2", f"-DBETA={BETA}", f"-DGAMMA={GAMMA}",
                    str(ROOT / "src/fhn_replay.c"), "-lm", "-o", str(exe)], check=True)
    with ProcessPoolExecutor(12) as pool:
        runs = list(pool.map(_run_seed, [(exe, s) for s in range(1001, 1025)]))
    rows = []
    for run in runs:
        print(run["seed"], run["clusters"], sorted(run["sizes"], reverse=True), run["checkpoint_tail"], flush=True)
        if run["clusters"] != 3:
            continue
        part = Partition(run["sizes"])

        def ev(t, z):
            return z[0]
        ev.direction = 1
        s = solve_ivp(lambda t, z: part.rhs(z, -3, .2), (0, 600), np.r_[run["X"], run["Y"]], method="DOP853",
                      rtol=1e-11, atol=1e-13, max_step=.05, events=ev)
        ts, ys = s.t_events[0], s.y_events[0]
        w = part.shoot(np.r_[ys[-2], ts[-1] - ts[-2]], -3, .2, tol)
        r = part.audit(w, -3, .2, tol)
        r.update(seed=run["seed"])
        rows.append(r)
        np.savez(OUT / f"start_seed{run['seed']}.npz", w=w, K=-3.0, sizes=part.sizes, eps=.2)
        print("   T=%.6f max=%.6f %s %s %s" % (r["period"], r["max_all"], r["dominant_kind"],
                                             r["dominant_population"], "stable" if r["stable"] else "NOT STABLE"),
              flush=True)
    (OUT / "ensemble_states.json").write_text(json.dumps(runs, indent=2))
    (OUT / "ensemble_floquet.json").write_text(json.dumps(rows, indent=2))


def scan(name, tol, step0, kmin=-15.0, kmax=None, directions=(-1, 1)):
    """kmax defaults to K_1; pass a value up to 0 to follow a branch past the rest-state boundary."""
    d = np.load(OUT / f"start_{name}.npz")
    part, K0, e = Partition(d["sizes"]), float(d["K"]), .2
    kmax = k1(.2) if kmax is None else kmax
    tag = "" if kmax == k1(.2) and tuple(directions) == (-1, 1) else f"_kmax{kmax:g}_dir{''.join(map(str, directions))}"
    folder = OUT / f"scan_{name}_tol{tol:g}_step{step0:g}{tag}"
    folder.mkdir(parents=True, exist_ok=True)
    w0 = part.shoot(d["w"], K0, e, tol)
    r0 = part.audit(w0, K0, e, tol)
    rows, events = [r0], []
    for direction in directions:
        K, w, old, step = K0, w0.copy(), None, step0
        prev = r0
        while True:
            target = K + direction * step
            if target > kmax or target < kmin:
                events.append(dict(direction=direction, status="range_end", last_K=K))
                break
            guess = w if old is None else w + (w - old[1]) * (target - K) / (K - old[0])
            cand = part.shoot(guess, target, e, tol)
            if cand is None:
                step /= 2
                if step < 2.5e-4:
                    events.append(dict(direction=direction, status="shooting_failed", last_K=K, failed_K=target,
                                       last_stable=prev["stable"]))
                    break
                continue
            r = part.audit(cand, target, e, tol)
            rows.append(r)
            if not r["valid"]:
                events.append(dict(direction=direction, status="orbit_validation_failed", last_K=K,
                                   failed_K=target))
                break
            if r["stable"] != prev["stable"]:
                (sk, sw), (uk, uw) = ((K, w), (target, cand)) if prev["stable"] else ((target, cand), (K, w))
                while abs(uk - sk) > 5e-4:
                    mid = (sk + uk) / 2
                    mw = part.shoot((sw + uw) / 2, mid, e, tol)
                    if mw is None:
                        break
                    mr = part.audit(mw, mid, e, tol)
                    rows.append(mr)
                    if not mr["valid"]:
                        break
                    if mr["stable"]:
                        sk, sw = mid, mw
                    else:
                        uk, uw = mid, mw
                ur = next(x for x in reversed(rows) if x["K"] == uk)
                events.append(dict(direction=direction, status="stability_change", stable_K=sk, unstable_K=uk,
                                   becomes="unstable" if prev["stable"] else "stable", kind=ur["dominant_kind"],
                                   population=ur["dominant_population"], eigenvalue=ur["dominant_eigenvalue"]))
                print(name, json.dumps(events[-1]), flush=True)
            old, K, w, prev, step = (K, w), target, cand, r, step0 if step == step0 else step
        (folder / "results.json").write_text(json.dumps(rows, indent=2))
    summary = dict(branch=name, sizes=part.sizes.tolist(), eps=e, K_start=K0, rtol=tol, step=step0, K1=k1(e), kmax=kmax,
                   events=events)
    (folder / "summary.json").write_text(json.dumps(summary, indent=2))
    print("SUMMARY", json.dumps(summary), flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["homotopy", "homotopy_q", "ensemble", "scan", "arc", "search", "eps_slice"])
    ap.add_argument("--eps", type=float)
    ap.add_argument("--ds", type=float, default=.002)
    ap.add_argument("--k-from", type=float, default=-.2)
    ap.add_argument("--max-points", type=int, default=600)
    ap.add_argument("--branch")
    ap.add_argument("--rtol", type=float, default=2e-10)
    ap.add_argument("--step", type=float, default=.05)
    ap.add_argument("--kmax", type=float)
    ap.add_argument("--directions", default="-1,1")
    ap.add_argument("--slice-points", type=int, default=4000)
    a = ap.parse_args()
    {"homotopy": homotopy, "ensemble": ensemble, "homotopy_q": lambda: homotopy_fixed_q(a.branch),
     "arc": lambda: arc(a.branch, ds=a.ds, k_from=a.k_from, max_points=a.max_points, k_step=.01), "search": search,
     "eps_slice": lambda: eps_slice(a.eps, max_points=a.slice_points)}.get(
        a.mode, lambda: scan(a.branch, a.rtol, a.step, kmax=a.kmax,
                             directions=tuple(int(x) for x in a.directions.split(","))))()
