#!/usr/bin/env python3
"""Final-state cluster counts of the full N = 100 system as a function of eps (exploratory).

Question (docs/OPEN_QUESTIONS.md, 2026-10-21): does clustering into few clusters become less likely as eps grows?
(beta, gamma) = (0.7, 0.5) keeps the single element excitable for every eps (|x*| > 1).

Two families of conditions, because K enters the fast equation without the 1/eps factor:
  fixedK  : K = -3 for every eps
  fixedEK : eps*K = -0.9 (equal to K = -3 at eps = 0.3)
Initial conditions as in scripts/run_pilot_ensemble.py: NumPy PCG64(seed), x_i ~ U(-2.5, 2.5), y_i ~ U(-2, 2).
Integration: src/fhn_replay.c, RK4, h = 0.005, 8e6 steps (t = 4e4), checkpoints every 4e5 steps.
Cluster count: component count at threshold 1e-4 at the final step (fhn_replay.c), with two consistency flags:
equal counts at thresholds 1e-3..1e-8, and a constant count over the last 5 checkpoints (last 1.6e6 steps).

  python3 scripts/eps_cluster_ensemble.py [--seeds 2001:2201] [--h 0.005] [--steps 8000000] [--tag main]
  python3 scripts/eps_cluster_ensemble.py --grid-eps 0.15,0.2 --grid-K -1,-3 --seeds 3001:3051 --tag grid

Distributed runs: --hosts host:slots,... (default local:4; 'local' is this machine; other entries are hosts reachable
by ssh). Initial conditions are generated and outputs parsed here; each host runs scripts/sim_worker.py (stdlib only)
and the simulator compiled on that host with the same gcc flags. Remote files live in ~/ensemble_remote/. Each remote
worker runs in a systemd scope with MemoryMax (--remote-mem). Every row records its host. In the original computation
the runs were distributed over three Linux machines with gcc 13.3.0, which gave bitwise-identical final states for a
2e6-step test run.
(Public version: the default host list and the remote directory name were changed from the author's settings.)
"""
from pathlib import Path
import argparse
import json
import queue
import shlex
import subprocess
import threading

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "analysis/eps_cluster_ensemble"
BETA, GAMMA, N = 0.7, 0.5, 100
EPS_LIST = [0.15, 0.20, 0.25, 0.30, 0.35, 0.40]
REMOTE = "~/ensemble_remote"


def conditions():
    out = [("fixedK", e, -3.0) for e in EPS_LIST]
    out += [("fixedEK", e, -0.9 / e) for e in EPS_LIST if abs(e - .3) > 1e-12]  # eps = 0.3 duplicates fixedK
    return out


def exe_name(eps, n=None):
    n = N if n is None else n
    return f"fhn_e{eps:.3f}" if n == 100 else f"fhn_N{n}_e{eps:.3f}"


def gcc_cmd(eps, src, exe, n=None):
    return ["gcc", "-O3", "-std=c11", f"-DN={N if n is None else n}", f"-DEPS={eps!r}", f"-DBETA={BETA}", f"-DGAMMA={GAMMA}", src, "-lm",
            "-o", exe]


def setup_host(host, eps_values, n_values=None):
    """Build the simulators for every eps (and N in n_values, default [N]) on one host (always rebuilt, so binaries
    match src/fhn_replay.c)."""
    n_values = n_values or [N]
    if host == "local":
        (OUT / "bin").mkdir(parents=True, exist_ok=True)
        for e in eps_values:
            for n in n_values:
                subprocess.run(gcc_cmd(e, str(ROOT / "src/fhn_replay.c"), str(OUT / "bin" / exe_name(e, n)), n),
                               check=True)
        return subprocess.run(["gcc", "--version"], capture_output=True, text=True).stdout.splitlines()[0]
    subprocess.run(["ssh", "-o", "BatchMode=yes", host, f"mkdir -p {REMOTE}/src {REMOTE}/bin"], check=True)
    subprocess.run(["scp", "-q", str(ROOT / "src/fhn_replay.c"), str(ROOT / "scripts/sim_worker.py"),
                    f"{host}:{REMOTE[2:]}/src/"], check=True)
    build = " && ".join(shlex.join(gcc_cmd(e, "src/fhn_replay.c", "bin/" + exe_name(e, n), n))
                        for e in eps_values for n in n_values)
    subprocess.run(["ssh", host, f"cd {REMOTE} && {build}"], check=True)
    return subprocess.run(["ssh", host, "gcc --version"], capture_output=True, text=True).stdout.splitlines()[0]


def initial_text(seed):
    rng = np.random.Generator(np.random.PCG64(seed))
    pts = np.column_stack((rng.uniform(-2.5, 2.5, N), rng.uniform(-2.0, 2.0, N)))
    return "".join(f"{x:.17g} {y:.17g}\n" for x, y in pts)


def parse(task, stdout, host):
    family, eps, K, seed, h, steps = task
    out = stdout.splitlines()
    finals = [l.split(",") for l in out if l.startswith(f"{steps},") and len(l.split(",")) == 6]
    count = next(int(f[5]) for f in finals if f[3] == "1.0e-04")
    checkpoints = [int(l.split(",")[5]) for l in out if l[0].isdigit() and len(l.split(",")) == 7]
    sizes = [int(v) for v in next(l for l in out if l.startswith("component_sizes_descending,")).split(",")[2:]]
    cent = next(l for l in out if l.startswith("component_centroids,")).split(",")[3:]
    centroids = [[float(cent[3 * i]), float(cent[3 * i + 1]), int(cent[3 * i + 2])] for i in range(len(cent) // 3)]
    row = dict(family=family, N=N, eps=eps, K=K, seed=seed, h=h, steps=steps, clusters=count, sizes=sizes,
               centroids=centroids, thresholds_agree=len({int(f[5]) for f in finals}) == 1,
               tail_constant=len(set(checkpoints[-5:])) == 1, checkpoint_tail=checkpoints[-5:], host=host)
    state = [l.split(",") for l in out if l.startswith(f"state,{steps},")]
    if state:  # --save-final: the full final state (x_i, y_i), i = 0..N-1
        z = [0.0] * (2 * N)
        for f in state:
            z[int(f[2])], z[N + int(f[2])] = float(f[3]), float(f[4])
        row["final_state"] = z
    return row


def serve(host, slots, todo, done, mem, save_final=False, init=None, extra_args=None, parser=None):
    """Feed tasks from `todo` to one host's worker, keeping at most `slots` outstanding; results go to `done`.

    `init(task)` gives the initial-state text; default: the random initial condition of seed task[3].
    `extra_args` replaces the trailing simulator arguments; `parser(task, stdout, host)` replaces parse().
    An optional 7th task field is the N of that task's simulator (default: the module's N).
    """
    worker = ["python3", "-u"]
    if host == "local":
        cmd = worker + [str(ROOT / "scripts/sim_worker.py"), "--bin-dir", str(OUT / "bin"), "-j", str(slots)]
    else:
        cmd = ["ssh", "-o", "BatchMode=yes", "-o", "ServerAliveInterval=30", host,
               f"systemd-run --user --scope --quiet -p MemoryMax={mem} -p MemorySwapMax=0 "
               f"python3 -u {REMOTE}/src/sim_worker.py --bin-dir {REMOTE}/bin -j {slots}"]
    p = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True, bufsize=1)
    pending = {}

    def send():
        try:
            idx, task = todo.get_nowait()
        except queue.Empty:
            return False
        pending[idx] = task
        p.stdin.write(json.dumps(dict(id=idx, exe=exe_name(task[1], task[6] if len(task) > 6 else None), stdin=init(task) if init else initial_text(task[3]),
                                      args=[repr(task[2]), repr(task[4]), str(task[5]), str(task[5] // 20)]
                                      + (extra_args if extra_args is not None
                                         else ["0", "0", "0", "0", str(task[5])] if save_final else []))) + "\n")
        p.stdin.flush()
        return True

    for _ in range(slots):
        if not send():
            break
    if not pending:
        p.stdin.close()
    for line in p.stdout:
        r = json.loads(line)
        task = pending.pop(r["id"])
        if r["rc"] != 0:
            done.put((r["id"], RuntimeError(f"{host}: task {task} rc={r['rc']} {r['stderr']}")))
        else:
            done.put((r["id"], (parser or parse)(task, r["stdout"], host)))
        if not send() and not pending:
            p.stdin.close()
    p.wait()
    if pending:  # worker died: return its tasks to the queue for the other hosts
        for idx, task in pending.items():
            todo.put((idx, task))
        done.put((None, f"{host}: worker exited with {len(pending)} tasks outstanding (rc={p.returncode})"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default="2001:2201")
    ap.add_argument("--h", type=float, default=.005)
    ap.add_argument("--steps", type=int, default=8_000_000)
    ap.add_argument("--hosts", default="local:4",
                    help="host:slots,... ('local' = this machine; by the author's rule, heavy runs use only the "
                         "two remote hosts)")
    ap.add_argument("--remote-mem", default="8G", help="MemoryMax of each remote worker scope")
    ap.add_argument("--tag", default="main")
    ap.add_argument("--save-final", action="store_true", help="also store each run's final state")
    ap.add_argument("--eps", help="comma-separated subset of EPS_LIST")
    ap.add_argument("--grid-eps", help="comma-separated eps values: (eps, K) grid mode, family 'grid'")
    ap.add_argument("--grid-K", help="comma-separated K values for the grid mode")
    ap.add_argument("--grid-q", help="comma-separated eps*K values: grid at fixed eps*K, family 'gridq'")
    ap.add_argument("--N", type=int, default=100, help="number of elements (simulator compiled with -DN)")
    a = ap.parse_args()
    global N
    N = a.N
    s0, s1 = map(int, a.seeds.split(":"))
    if a.grid_eps and a.grid_q:
        conds = [("gridq", float(e), float(q) / float(e)) for e in a.grid_eps.split(",") for q in a.grid_q.split(",")]
    elif a.grid_eps:
        conds = [("grid", float(e), float(K)) for e in a.grid_eps.split(",") for K in a.grid_K.split(",")]
    else:
        conds = [c for c in conditions() if a.eps is None or any(abs(c[1] - float(e)) < 1e-12 for e in a.eps.split(","))]
    hosts = [(h, int(n)) for h, n in (x.rsplit(":", 1) for x in a.hosts.split(","))]
    eps_values = sorted({e for _, e, _ in conds})
    OUT.mkdir(parents=True, exist_ok=True)
    for h, n in hosts:
        print("host", h, n, setup_host(h, eps_values), flush=True)
    tasks = [(f, e, K, s, a.h, a.steps) for f, e, K in conds for s in range(s0, s1)]
    todo, done = queue.Queue(), queue.Queue()
    for item in enumerate(tasks):
        todo.put(item)
    threads = [threading.Thread(target=serve, args=(h, n, todo, done, a.remote_mem, a.save_final), daemon=True) for h, n in hosts]
    for t in threads:
        t.start()
    results = {}
    while len(results) < len(tasks):
        try:
            idx, r = done.get(timeout=60)
        except queue.Empty:
            if not any(t.is_alive() for t in threads):
                raise SystemExit(f"all workers exited with {len(tasks) - len(results)} tasks unfinished")
            continue
        if idx is None:
            print("WARNING", r, flush=True)
            if not any(t.is_alive() for t in threads):
                raise SystemExit("all workers exited")
            continue
        if isinstance(r, Exception):
            raise r
        results[idx] = r
        if len(results) % 200 == 0:
            print(len(results), "/", len(tasks), flush=True)
        if len(results) % 2000 == 0:
            (OUT / f"runs_{a.tag}.partial.json").write_text(json.dumps([results[i] for i in sorted(results)]))
    rows = [results[i] for i in range(len(tasks))]
    (OUT / f"runs_{a.tag}.json").write_text(json.dumps(rows))
    print("runs per host", {h: sum(r["host"] == h for r in rows) for h, _ in hosts}, flush=True)
    for f, e, K in conds:
        rr = [r for r in rows if r["family"] == f and r["eps"] == e and r["K"] == K]
        hist = {}
        for r in rr:
            hist[r["clusters"]] = hist.get(r["clusters"], 0) + 1
        flags = sum(not (r["thresholds_agree"] and r["tail_constant"]) for r in rr)
        print(f"{f:8s} eps={e:.2f} K={K:+.4f} n={len(rr)} counts={dict(sorted(hist.items()))} flagged={flags}",
              flush=True)


if __name__ == "__main__":
    main()
