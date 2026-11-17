#!/usr/bin/env python3
"""Summary of the two-element departures (path B) at eK = -1.2 from the saved offsets (no integration; 2026-10-31,
supplement S3 of paper 3; evidence_inventory: the firing intervals at the reversal were computed by hand on 2026-10-28).

Input: analysis/switching_check/rerun_states/pathB_offsets_0396.json, _0398.json (scripts/pathB_firing_offsets.py,
51a59957). In each event, cycle k has the reference firing time t_k (median firing time of the 14 remaining members of
the former cluster) and the offset o_k of each departing element (firing time - t_k; < 0: advanced).
Per event:
  returned: the departing element whose offset changes sign from negative to positive (k_rev: the first cycle with
            o_k > 0 after o_{k-1} < -0.5); other: the second departing element
  first_above: first cycle with |o| > 1e-3 (both elements; its sign), cycles from there to k_rev
  offsets of the returned element at k_rev - 1 and k_rev; lag of the returned element behind the other at k_rev - 1
  firing intervals at k_rev: reference t_k - t_{k-1}, element (t_k + o_k) - (t_{k-1} + o_{k-1}) for both elements;
  returned element's intervals at k_rev - 1 and k_rev + 1
  rest_spread (range of the 14 firing times) at k_rev - 1 and k_rev, and its maximum before k_rev
  fate of the other element (from the input)
Output: analysis/switching_check/rerun_states/pathB_summary.json

  python3 scripts/paper3_pathB_summary.py
"""
from pathlib import Path
import hashlib
import json

ROOT = Path(__file__).resolve().parents[1]
RS = ROOT / "analysis/switching_check/rerun_states"


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    inputs: dict[str, str] = {}
    ev: list[dict] = []
    for idx in (396, 398):
        f = RS / f"pathB_offsets_{idx:04d}.json"
        inputs[str(f.relative_to(ROOT))] = sha(f)
        for e in json.loads(f.read_text())["events"]:
            cy = e["cycles"]
            t = [c["t"] for c in cy]
            a, b = (str(x) for x in e["pair"])
            rev = {}
            for el in (a, b):
                o = [c["off"][el] for c in cy]
                rev[el] = next((k for k in range(1, len(o)) if o[k] > 0 and o[k - 1] < -0.5), None)
            ret = [el for el in (a, b) if rev[el] is not None]
            assert len(ret) == 1, (idx, e["t_switch"], rev)
            r = ret[0]
            oth = b if r == a else a
            k = rev[r]
            o_r = [c["off"][r] for c in cy]
            o_o = [c["off"][oth] for c in cy]
            first = next(j for j in range(len(cy)) if abs(o_r[j]) > 1e-3 or abs(o_o[j]) > 1e-3)

            def iv(o, j):
                return (t[j] + o[j]) - (t[j - 1] + o[j - 1])

            ev.append(dict(
                index=idx, t_switch=e["t_switch"], kind=e["kind"], returned=int(r), other=int(oth),
                first_above_signs=[o_r[first] < 0 and "advanced" or "retarded", o_o[first] < 0 and "advanced" or "retarded"],
                cycles_first_to_reversal=k - first,
                offset_returned_before=o_r[k - 1], offset_returned_after=o_r[k],
                lag_returned_minus_other_before=o_r[k - 1] - o_o[k - 1],
                interval_reference=t[k] - t[k - 1], interval_returned=iv(o_r, k), interval_other=iv(o_o, k),
                interval_returned_prev=iv(o_r, k - 1), interval_returned_next=iv(o_r, k + 1),
                rest_spread_before=cy[k - 1]["rest_spread"], rest_spread_at=cy[k]["rest_spread"],
                rest_spread_max_before=max(c["rest_spread"] for c in cy[:k]),
                fate_other=e["fate"][oth]["fate"]))
    def rng(key):
        v = [x[key] for x in ev]
        return [min(v), max(v)]

    summary = dict(
        n=len(ev), both_advanced=sum(x["first_above_signs"] == ["advanced", "advanced"] for x in ev),
        returned_lagging_before=sum(x["lag_returned_minus_other_before"] > 0 for x in ev),
        **{k: rng(k) for k in ("cycles_first_to_reversal", "offset_returned_before", "offset_returned_after",
                               "lag_returned_minus_other_before", "interval_reference", "interval_returned",
                               "interval_other", "interval_returned_prev", "interval_returned_next",
                               "rest_spread_before", "rest_spread_at")},
        fate_other={f_: sum(x["fate_other"] == f_ for x in ev) for f_ in sorted({x["fate_other"] for x in ev})})
    out = dict(script="scripts/paper3_pathB_summary.py", script_sha256=sha(__file__), inputs=inputs, events=ev,
               summary=summary)
    p = RS / "pathB_summary.json"
    p.write_text(json.dumps(out, indent=1))
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
