#!/usr/bin/env python3
"""Paper 3, Table I: collective Floquet multipliers of the five N = 50 node orbits (2026-11-03).

The node files (heteroclinic_n50.py) store only the internal (2x2 block) exponents. This script takes the stored
orbit point and period of each node, without shooting again, and integrates one period of the full variational
equation of the reduced system with eps_three_cluster.Partition.audit (the routine of paper 1: DOP853, tol 2e-12).
Recorded: all collective multipliers (moduli, sorted), the trivial one removed by audit, the audit checks (return
error, neutral error, tangent error, validity), and the internal multiplier moduli for comparison with the node file.
For partitions with two singletons, the difference of the singletons is a collective direction of the reduced system;
it is listed with the others and flagged when the singletons coincide on the orbit.
  -> analysis/heteroclinic_n50/node_collective.json

  python3 scripts/paper3_node_collective.py
"""
from pathlib import Path
import hashlib
import json
import sys
import time

import numpy as np
from scipy.integrate import solve_ivp

sys.path.insert(0, str(Path(__file__).resolve().parent))
from eps_three_cluster import Partition, ROOT  # noqa: E402

D = ROOT / "analysis/heteroclinic_n50"
NODES = [("-0.9", "A", "node_17-17-16_q-0.900_s1.json"), ("-0.9", "B", "node_17-16-16-1_q-0.900_s1.json"),
         ("-1.0", "A", "node_17-17-16_q-1.000_s1.json"), ("-1.0", "B", "node_17-16-16-1_q-1.000_s1.json"),
         ("-1.2", "C", "node_16-16-16-1-1_q-1.200_s2.json")]
TOL = 2e-12


def spectrum(part, z, T, K, e):
    n = part.n
    s = solve_ivp(lambda _t, v: np.r_[part.rhs(v[:n], K, e), (part.jac(v[:n], K, e) @ v[n:].reshape(n, n)).ravel()],
                  (0, T), np.r_[z, np.eye(n).ravel()], method="DOP853", rtol=max(TOL * .1, 2.3e-14), atol=TOL * .001,
                  max_step=.04)
    eig = np.linalg.eigvals(s.y[n:, -1].reshape(n, n))
    neut = int(np.argmin(abs(eig - 1)))
    return eig[neut], np.delete(eig, neut)


def main():
    t0 = time.time()
    out = dict(script=Path(__file__).name, sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
               command="python3 scripts/paper3_node_collective.py", tol=TOL, nodes=[])
    for q, typ, f in NODES:
        p = D / f
        node = json.loads(p.read_text())["node"]
        part = Partition(node["sizes"])
        z, T, K, e = np.asarray(node["orbit_point"]), node["period"], node["K"], node["eps"]
        a = part.audit(np.r_[z, T], K, e, TOL)
        neut, ce = spectrum(part, z, T, K, e)
        mod = sorted(abs(ce).tolist(), reverse=True)
        out["nodes"].append(dict(
            q=float(q), type=typ, file=str(p.relative_to(ROOT)), file_sha256=hashlib.sha256(p.read_bytes()).hexdigest(),
            sizes=node["sizes"], period=T, coincide=node["coincide"], trivial_multiplier=[neut.real, neut.imag],
            collective_moduli=mod, collective=[[c.real, c.imag] for c in ce[np.argsort(-abs(ce))]],
            collective_max=a["collective_max"], internal_max_audit=a["internal_max"],
            internal_exponents_node=node["internal"],
            checks={k: a[k] for k in ("return_error", "neutral_error", "tangent_error", "min_pair_cycle_distance",
                                      "valid")}))
        print(q, typ, node["sizes"], "collective moduli", np.round(mod, 5).tolist(), "valid", a["valid"],
              "ret", f"{a['return_error']:.1e}", flush=True)
    out["run_seconds"] = time.time() - t0
    (D / "node_collective.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
