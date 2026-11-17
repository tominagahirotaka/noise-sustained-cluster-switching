#!/usr/bin/env python3
"""Per-cycle depth of the split subsets in a noiseless tree run of run 88 (exploratory, 2026-10-25).

  python3 scripts/noiseless_cycles.py analysis/noiseless_n50/nl_<tag>.npz analysis/noiseless_n50/cycles_<tag>.json
"""
import json
import sys

import numpy as np

d=np.load(sys.argv[1]); g=d['group']; sp=d['spread'].astype(float); ev=int(d['every'])
R=len(g); ts=10*ev*np.arange(1,len(sp)+1)
k5=max(k for k in range(R) if len(np.bincount(g[k])[np.bincount(g[k])>=5])==5); lab=g[k5]; c=np.bincount(lab)
sets={int(s):np.nonzero(lab==np.nonzero(c==s)[0][0])[0] for s in c if s>=5}
s5,s12=sorted(s for s in sets if s<20); bigA=max(s for s in sets if 100<s<130 and s!=127)  # 125 or 123
A,B,C5,C=sets[s12],sets[bigA],sets[s5],sets[127]
# joined indicator at the spread times (every 10 records): subset in the same group as its partner
gi=g[10*np.arange(1,len(sp)+1)-1]
j12=np.array([len(np.unique(r[np.r_[A,B]]))==1 for r in gi]); j5=np.array([len(np.unique(r[np.r_[C5,C]]))==1 for r in gi])
x12=np.nanmedian(sp[:,A],1); x5=np.nanmedian(sp[:,C5],1)
def stretches(j):
    out=[];k0=None
    for k in range(len(j)+1):
        on=k<len(j) and j[k]
        if on and k0 is None:k0=k
        if not on and k0 is not None:
            if k-k0>=5: out.append((k0,k))
            k0=None
    # bridge gaps < 5 samples (500 units)
    m=[]
    for a,b in out:
        if m and a-m[-1][1]<5: m[-1]=(m[-1][0],b)
        else: m.append((a,b))
    return m
out={}
for name,j,x,lam in (("137 (%d+%d)"%(bigA,s12),j12,x12,1.045e-3),("132 (127+%d)"%s5,j5,x5,-2.516e-3)):
    st=stretches(j); rows=[]
    for a,b in st:
        y=x[a:b]; i=int(np.nanargmin(y))
        sl=np.polyfit(ts[a+i:b],y[i:]*np.log(10),1)[0] if b-a-i>5 else np.nan
        sd=np.polyfit(ts[a:a+i+1],y[:i+1]*np.log(10),1)[0] if i>5 else np.nan
        rows.append(dict(join=int(ts[a]),split=int(ts[b-1]),life=int(ts[b-1]-ts[a]),rmin=float(y[i]),tmin=int(ts[a+i]),down=float(sd),up=float(sl)))
    print(name); 
    for r in rows: print("  ",r)
    out[name]=rows
json.dump(out,open(sys.argv[2],'w'),indent=1)
