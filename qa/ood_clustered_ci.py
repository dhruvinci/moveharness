#!/usr/bin/env python3
# RELEASE NOTE: inputs (gt_v2/ raw review labels, ood_v2_extracted.json) are
# PRIVATE/withheld for consent; released OUTPUTS are confusion_data_v2.json and
# ood_v2_metrics.json. This script is provided for methodology inspection.
"""
Video-clustered (cluster-bootstrap) CIs for the OOD result.

The headline OOD CIs in score_ood_v2.py are Wilson intervals over all 4,620
segments, which treats segments as independent. They are not: segments within one
video are correlated. This script keeps the SAME segment-weighted point estimates
but recomputes 95% CIs by resampling VIDEOS with replacement (cluster bootstrap),
which honours within-video correlation. The result is a slightly wider, more
honest interval. Point estimates are unchanged.

Deterministic (fixed seed). No network, no model inference. Run from repo root.
"""
import json, glob, os, sys, random
sys.path.insert(0, "benchmark/qa")
from score_ood_v2 import canon, parse_correction, SEM_FAMILY, CLUSTER, adjacent

GT_DIR = "p1-measurement-reasoning/gt_v2"
EXTRA  = "benchmark/qa/ood_v2_extracted.json"
B      = 5000
random.seed(17)

extra = json.load(open(EXTRA)) if os.path.exists(EXTRA) else {}

# ---- per-video lists of per-segment (strict, semantic, adjacency) flags ----
videos = []
for f in sorted(glob.glob(os.path.join(GT_DIR, "gt_v2_*.json"))):
    d = json.load(open(f)); fid = os.path.basename(f); rows = []
    for r in d.get("segment_reviews") or []:
        st = r.get("status"); mp = canon(r.get("position", ""))
        if st == "accepted":
            rows.append((1, 1, 1))
        elif st == "unreviewed":
            rows.append((1, 1, 1))            # assumed correct (review protocol)
        elif st == "rejected":
            key = f"{fid}::{r.get('index')}"
            ap = canon(extra[key]) if key in extra else parse_correction(r.get("correction_text", "") or "", mp)[0]
            if ap is None:
                rows.append((0, 0, 0)); continue
            sem = SEM_FAMILY.get(mp) is not None and SEM_FAMILY.get(mp) == SEM_FAMILY.get(ap)
            adj = adjacent(CLUSTER.get(mp, "?"), CLUSTER.get(ap, "?"))
            rows.append((0, 1 if sem else 0, 1 if (sem or adj) else 0))
    if rows:
        videos.append(rows)

def rates(vids):
    n = s = se = a = 0
    for rows in vids:
        for st, sm, ad in rows:
            n += 1; s += st; se += sm; a += ad
    return (100*s/n, 100*se/n, 100*a/n)

pt = rates(videos)
nseg = sum(len(v) for v in videos)
print(f"videos={len(videos)} segments={nseg}")
print(f"point estimates: exact={pt[0]:.1f}  coarse={pt[1]:.1f}  adjacent={pt[2]:.1f}")

# ---- cluster bootstrap: resample VIDEOS with replacement ----
bs = {0: [], 1: [], 2: []}
nv = len(videos)
for _ in range(B):
    samp = [videos[random.randrange(nv)] for _ in range(nv)]
    r = rates(samp)
    for k in range(3): bs[k].append(r[k])
def ci(xs):
    xs = sorted(xs); lo = xs[int(0.025*len(xs))]; hi = xs[int(0.975*len(xs))]
    return round(lo, 1), round(hi, 1)
names = ["exact", "coarse", "adjacent"]
out = {"n_videos": len(videos), "n_segments": nseg, "B": B, "method": "video cluster bootstrap"}
print(f"\nvideo-clustered 95% CIs (B={B}):")
for k in range(3):
    lo, hi = ci(bs[k])
    out[names[k]] = {"pct": round(pt[k], 1), "ci": [lo, hi]}
    print(f"  {names[k]:9s} {pt[k]:.1f}%  [{lo}, {hi}]")
json.dump(out, open("benchmark/qa/ood_clustered_ci.json", "w"), indent=2)
