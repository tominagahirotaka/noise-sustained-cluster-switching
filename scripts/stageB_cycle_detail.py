#!/usr/bin/env python3
"""How the D stretches of the run-88 tree runs end, from the per-cycle output of stageB_classify.py (2026-10-28).

No new run and no change of the fixed rules: this only reads the cycles of <classify>.json. Per run:
  - the D stretches (cycles, start, end) and what follows the last one (c, F, back to D, end of record);
  - the sets: the small group of 137 and of 132 in every cycle, compared with those of the first D stretch
    (did the 12-element set of 137 change? did the set of 132 change?);
  - the first cycle after a D stretch that is not D, and why it fails the D conditions of stageB_classify.py:
    F window of 137, other sets, no 132 depth, length off the median by > 5 %, 132 depth off the median by > 0.5
    decades. The medians are given twice: over the D stretch itself, and over the run of consecutive cycles with the
    same sets and no F window (the medians the classifier uses);
  - per cycle the length T and the 132 depth relative to the D-stretch medians (which one leaves its band first).
  -> <out>.json

  python3 scripts/stageB_cycle_detail.py analysis/noiseless_n50/stageB_classify_mid2_boar.json ... --out x.json
"""
from pathlib import Path
import argparse
import json

import numpy as np


def detail(r):
    cy = r["cycles"]
    lab = r["labels"]
    out = dict(file=Path(r["file"]).name, eta=r["eta"], seed=r["seed"], classification=r["classification"],
               labels=lab, stretches=[], cycles=[])
    st = r["stretches"]
    if not st:
        return out
    s0 = st[0]
    key = (cy[s0["first"]]["small137"], cy[s0["first"]]["small132"])
    for s in st:
        Ts = np.array([c["T"] for c in cy[s["first"]:s["last"] + 1]])
        ds = np.array([c["depth132small"] for c in cy[s["first"]:s["last"] + 1]])
        mT, md = float(np.median(Ts)), float(np.median(ds))
        e = dict(first=s["first"], last=s["last"], n=s["n"], t0=s["t0"], t1=s["t1"], median_T=mT, median_depth132=md)
        k = s["last"] + 1
        if k >= len(cy):
            e["after"] = "end of record"
        else:
            c = cy[k]
            # the run of consecutive cycles with the same sets and no F window around this stretch (classifier medians)
            i = s["first"]
            kk = (cy[i]["small137"], cy[i]["small132"])
            while i > 0 and cy[i - 1]["F137"] == 0 and (cy[i - 1]["small137"], cy[i - 1]["small132"]) == kk:
                i -= 1
            j = s["last"] + 1
            while j < len(cy) and cy[j]["F137"] == 0 and (cy[j]["small137"], cy[j]["small132"]) == kk \
                    and cy[j]["depth132small"] is not None:
                j += 1
            Tr = float(np.median([x["T"] for x in cy[i:j]]))
            dr = float(np.median([x["depth132small"] for x in cy[i:j] if x["depth132small"] is not None]))
            why = []
            if c["F137"] > 0:
                why.append("F window of 137")
            if c["small137"] != key[0]:
                why.append("other 137 set")
            if c["small132"] != key[1]:
                why.append("other 132 set")
            if c["depth132small"] is None:
                why.append("no 132 depth")
            for name, mT_, md_ in (("stretch", mT, md), ("run", Tr, dr)):
                if abs(c["T"] - mT_) > .05 * mT_:
                    why.append(f"T off {name} median by {100 * (c['T'] / mT_ - 1):+.1f}%")
                if c["depth132small"] is not None and abs(c["depth132small"] - md_) > .5:
                    why.append(f"132 depth off {name} median by {c['depth132small'] - md_:+.2f} dec")
            nxt = lab[k:]
            e.update(after=lab[k], first_non_D=dict(cycle=k, t=c["t_split"], T=c["T"], depth132=c["depth132small"],
                                                    run_median_T=Tr, run_median_depth132=dr, why=why,
                                                    run_cycles=[i, j - 1]),
                     then=("back to D" if "D" in nxt else ("F" if "F" in nxt else "c to the end")),
                     next_F_cycle=(k + nxt.index("F")) if "F" in nxt else None)
        out["stretches"].append(e)
    mT, md = out["stretches"][0]["median_T"], out["stretches"][0]["median_depth132"]
    for k, c in enumerate(cy):
        out["cycles"].append(dict(k=k, t=c["t_split"], label=c["label"], same137=c["small137"] == key[0],
                                  same132=c["small132"] == key[1], F137=c["F137"],
                                  T_rel=round(c["T"] / mT - 1, 4),
                                  depth132_minus_median=None if c["depth132small"] is None
                                  else round(c["depth132small"] - md, 3)))
    out["set137_changed_after_first_D"] = any(not x["same137"] for x in out["cycles"][s0["first"]:])
    out["set132_changed_after_first_D"] = any(not x["same132"] for x in out["cycles"][s0["first"]:])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("classify", nargs="+")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    res = []
    for fn in a.classify:
        for r in json.loads(Path(fn).read_text())["runs"]:
            d = detail(r)
            res.append(d)
            print(d["file"], d["classification"], d["labels"], flush=True)
            for e in d["stretches"]:
                print("   D", e["first"], "-", e["last"], f"({e['n']})", "after", e.get("after"), e.get("then", ""),
                      e.get("first_non_D", {}).get("why", ""), flush=True)
            print("   137 set changed:", d.get("set137_changed_after_first_D"), " 132 set changed:",
                  d.get("set132_changed_after_first_D"), flush=True)
            print("   T_rel:", [x["T_rel"] for x in d["cycles"]], flush=True)
            print("   d132-med:", [x["depth132_minus_median"] for x in d["cycles"]], flush=True)
    Path(a.out).write_text(json.dumps(dict(inputs=a.classify, runs=res), indent=1))


if __name__ == "__main__":
    main()
