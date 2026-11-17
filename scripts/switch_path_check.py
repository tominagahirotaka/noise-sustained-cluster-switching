#!/usr/bin/env python3
"""Do the switches of one persistent-noise run follow the same path at every noise level? (exploratory, 2026-10-24)

Input: analysis/switching_check/rn_<index>[_n<eta>]_t<tmax>_states_z.npy for eta in 1e-6 ... 1e-14
(switching_renoise.py --states; same start, same noise sequence up to the factor eta). Every 10 time units the
clusters are cut at 1e-3 (single linkage, max norm) and the sizes of the clusters of >= 20 elements are taken.
A residence is a stretch of >= 2000 time units with the big sizes equal to --sizes (interruptions shorter than 500
time units bridged). At the end of each residence:
  - the leaving set: members of a big cluster 20 records before the end that, 50 records after the end, are no
    longer with the majority of that cluster (the cluster with the most leavers is taken);
  - the curve: log10 of the distance (max norm) of the centroid of the leaving set to the centroid of the rest of its
    cluster, from the start of the residence to 3000 time units after its end;
  - the same curve aligned at a fixed transverse amplitude (first crossing of 1e-3 after its minimum in the
    residence), from 2000 before to 3000 after;
  - the path after the end: the sequence of big-size tuples visited (consecutive repeats removed, stretches of
    >= 5 records) until the next residence.
For noise-sustained switching on a heteroclinic connection the curve should start near the noise floor and rise at
the transverse rate, and the path after the end should not depend on eta.

  python3 scripts/switch_path_check.py --index 88 --sizes 137 132 131  -> analysis/switching_check/switch_path_<index>.json
"""
from pathlib import Path
import argparse
import json
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from renoise_summary import D  # noqa: E402
from renoise_threshold_check import cut, merges  # noqa: E402

ETAS = (1e-6, 1e-8, 1e-10, 1e-12, 1e-14)
AFTER = 300  # records after the end of a residence


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", type=int, required=True)
    ap.add_argument("--tmax", type=int, default=80000)
    ap.add_argument("--sizes", type=int, nargs="+", required=True)
    a = ap.parse_args()
    target = tuple(a.sizes)
    res = []
    for eta in ETAS:
        tag = f"rn_{a.index:04d}" + ("" if eta == 1e-10 else f"_n{eta:.0e}") + f"_t{a.tmax}_states"
        zs = np.load(D / f"{tag}_z.npy")
        R, n = len(zs), zs.shape[1] // 2
        L, big = [], []
        for z in zs:
            _, _, Z = merges(z)
            _, inv = np.unique(cut(Z, n, 1e-3), return_inverse=True)
            L.append(inv)
            big.append(tuple(sorted((int(s) for s in np.bincount(inv) if s >= 20), reverse=True)))
        L = np.asarray(L)
        runs, k0 = [], 0
        for k in range(1, R + 1):
            if k == R or big[k] != big[k0]:
                runs.append((k0, k, big[k0]))
                k0 = k
        # residences: runs of the target tuple joined across interruptions shorter than 50 records
        segs = []
        for r0, r1, b in runs:
            if b != target:
                continue
            if segs and r0 - segs[-1][1] < 50:
                segs[-1][1] = r1
            else:
                segs.append([r0, r1])
        segs = [(s0, s1) for s0, s1 in segs if s1 - s0 >= 200]
        rows = []
        for s0, s1 in segs:
            # first record after the end that starts a run of another tuple of >= 20 records
            nxt = next((r0 for r0, r1, b in runs if r0 >= s1 and b != target and r1 - r0 >= 20), None)
            if nxt is None or nxt + 10 >= R:
                continue
            ref, late = L[s1 - 20], L[nxt + 10]
            best = None
            for c in np.unique(ref):
                mem = np.nonzero(ref == c)[0]
                if mem.size < 20:
                    continue
                maj = np.bincount(late[mem]).argmax()
                out = mem[late[mem] != maj]
                if out.size and (best is None or out.size > best[1].size):
                    best = (mem[late[mem] == maj], out, mem.size)
            if best is None:
                continue
            rest, out, src = best
            curve = []
            for j in range(s0, min(s1 + AFTER, R)):
                z = zs[j]
                d = max(abs(z[:n][out].mean() - z[:n][rest].mean()), abs(z[n:][out].mean() - z[n:][rest].mean()))
                curve.append(float(np.log10(max(d, 1e-300))))
            path, j, nxt = [], s1, s1
            while j < R:
                k = j
                while k < R and big[k] == big[j]:
                    k += 1
                if k - j >= 5 and (not path or path[-1] != list(big[j])):
                    path.append(list(big[j]))
                if big[j] == target and k - j >= 200:
                    break
                j = k
            c = np.asarray(curve)
            m = s1 - s0
            # alignment at a fixed transverse amplitude: first record after the minimum with distance >= 1e-3
            imin = int(np.argmin(c[:m]))
            up = np.nonzero(c[imin:] >= -3)[0]
            ka = imin + int(up[0]) if up.size else None
            # rise: fit over the part of the residence where the curve is between floor + 1 and the value at the end - 1
            floor = float(np.min(c[:m]))
            sel = np.nonzero((c[:m] > floor + 1) & (c[:m] < c[m - 1] - 1))[0]
            rate = float(np.polyfit(10 * sel, c[sel] * np.log(10), 1)[0]) if sel.size > 20 else None
            rows.append(dict(start=10 * s0, end=10 * s1, L=10 * m, source=int(src), leaving=int(out.size),
                             floor=floor, argmin_from_end=10 * (m - int(np.argmin(c[:m]))), rise_rate=rate,
                             after=[c[m + k] if m + k < len(c) else None for k in range(0, AFTER, 25)], path=path[:8],
                             t_align=None if ka is None else 10 * (s0 + ka), align_to_end=None if ka is None else 10 * (m - ka),
                             aligned=None if ka is None else [c[ka + k] if 0 <= ka + k < len(c) else None
                                                              for k in range(-200, AFTER, 25)]))
        res.append(dict(eta=eta, residences=rows))
        print(f"eta {eta:.0e}: {len(rows)} residences of {list(target)}")
        for r in rows:
            print(f"   {r['start']:6d}-{r['end']:6d} L {r['L']:6d} source {r['source']} leaving {r['leaving']:3d}"
                  f" floor {r['floor']:.1f} ({r['argmin_from_end']} before end) rise {r['rise_rate']}"
                  f"\n      after (0..+3000 by 250): {' '.join('nan' if x is None else f'{x:.1f}' for x in r['after'])}"
                  f"\n      aligned at 1e-3 ({r['align_to_end']} before end), -2000..+3000 by 250: "
                  f"{' '.join('nan' if x is None else f'{x:.1f}' for x in (r['aligned'] or []))}"
                  f"\n      path {r['path'][:5]}", flush=True)
    (D / f"switch_path_{a.index:04d}.json").write_text(json.dumps(res))


if __name__ == "__main__":
    main()
