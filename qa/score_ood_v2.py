#!/usr/bin/env python3
# RELEASE NOTE: inputs (gt_v2/ raw review labels, ood_v2_extracted.json) are
# PRIVATE/withheld for consent; released OUTPUTS are confusion_data_v2.json and
# ood_v2_metrics.json. This script is provided for methodology inspection.
"""
OOD correctness scorer (v2, n=200 user-reviewed in-the-wild uploads).

Input : p1-measurement-reasoning/gt_v2/gt_v2_*.json  (one per video)
        Each file is a human macro-review of the model's predicted position
        timeline: every predicted segment carries status accepted/rejected/
        unreviewed and, when rejected, a free-text correction "<actual>, not
        <model> - <reason>".

Scoring (the human is the scorer; accept == model strictly correct):
  - strict     : segment accepted (model's exact position judged correct).
  - semantic   : strict OR (rejected but model position in the SAME semantic
                 family as the corrected/actual position) -- credits
                 granularity/synonym misses.
  - adjacency  : semantic OR (model cluster == actual cluster, or the two
                 clusters are adjacent in the 5-state control graph) -- credits
                 "one position off in a pass/scramble sequence".
  unreviewed segments (0.4%) are assumed correct per the review protocol; we
  also report the result with them excluded.

All maps are explicit and documented below so the tiers are auditable.
Deterministic: no network, no model inference. Run from repo root.
"""
import json, glob, re, math, collections, os, sys

GT_DIR = "p1-measurement-reasoning/gt_v2"
# optional: a JSON of LLM-extracted actual positions for free-text corrections
# that regex can't parse: { "<file>::<idx>": "<canonical_position>" }
EXTRA_EXTRACT = "benchmark/qa/ood_v2_extracted.json"

# ----------------------------------------------------------------------------
# 1. canonicalization: every model/actual surface form -> a canonical position
# ----------------------------------------------------------------------------
def canon(s):
    s = (s or "").strip().lower()
    s = re.sub(r"[^a-z0-9 ]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    # drop filler words ONLY -- never structural words that appear inside
    # position names (side, on, half, north, south, knee, leg, single, etc.)
    s = re.sub(r"\b(position|reset|here|the|a|an|player|person|athlete|"
               r"started|already|attacking|defending|continues|goes|into)\b",
               " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    t = s.replace(" ", "_")
    # ---- explicit alias folding ----
    A = {
        # guard family
        "closed_guard":"guard","open_guard":"guard","seated_guard":"guard",
        "seated":"guard","butterfly":"guard","butterfly_guard":"guard",
        "de_la_riva":"guard","dlr":"guard","x_guard":"guard","spider_guard":"guard",
        "guard_bottom":"guard","guard_recovery":"guard","full_guard":"guard",
        "playing_guard":"guard","bottom_guard":"guard","z_guard":"guard",
        # half guard family
        "half_guard_bottom":"half_guard","half_guard_top":"half_guard",
        "half_guard_with_knee_shield":"half_guard","knee_shield":"half_guard",
        "half_guard_knee_shield":"half_guard","deep_half":"half_guard",
        "lockdown":"half_guard","z_guard_half":"half_guard","quarter_guard":"half_guard",
        # side control / top pins
        "side_mount":"side_control","scarf_hold":"side_control",
        "side_control_top":"side_control","kesa_gatame":"side_control",
        "twister_side_control":"side_control",
        # mount
        "high_mount":"mount","low_mount":"mount","s_mount":"mount","mounted":"mount",
        "technical_mount":"mount",
        # back
        "back_control":"back","back_mount":"back","back_take":"back",
        "taking_the_back":"back","rear_mount":"back","back_grab":"back",
        "crab_ride":"back",
        # turtle
        "turtled":"turtle","turtle_bottom":"turtle","front_headlock":"turtle",
        # leg entanglement
        "leg_lock":"leg_entanglement","ashi_garami":"leg_entanglement",
        "ashi":"leg_entanglement","50_50":"leg_entanglement","fifty_fifty":"leg_entanglement",
        "ankle_lock_position":"leg_entanglement","single_leg_x":"leg_entanglement",
        "saddle":"leg_entanglement","leg_entangle":"leg_entanglement",
        "leg_entanglements":"leg_entanglement","outside_ashi":"leg_entanglement",
        # top control generic
        "top":"top_control","top_pressure":"top_control","top_position":"top_control",
        "knee_on_belly":"knee_on_belly","north_south":"north_south",
        # transitions / events
        "dog_fight":"scramble","dogfight":"scramble","scramble_reversal":"scramble",
        "single_leg":"takedown","double_leg":"takedown","o_uchi_gari":"takedown",
        "takedown_attempt":"takedown","shot":"takedown","body_lock_takedown":"takedown",
        "guard_pass":"guard_pass","passing":"guard_pass","passing_guard":"guard_pass",
        "half_guard_passing":"guard_pass","pass":"guard_pass","guard_passing":"guard_pass",
        "guard_pull":"guard_pull","pulling_guard":"guard_pull",
        "sweep":"sweep","reversal":"sweep",
        # submission family
        "submission_attempt":"submission","submission_in_progress":"submission",
        "submission_threat":"submission","submission_setup":"submission",
        "submission_defense":"submission","choke":"submission","armbar":"submission",
        "triangle":"submission","kimura":"submission","heel_hook":"submission",
        "guillotine":"submission","rear_naked_choke":"submission","rnc":"submission",
        "darce":"submission","americana":"submission","straight_ankle":"submission",
        "ankle_lock":"submission","kneebar":"submission","arm_triangle":"submission",
        "finishing_submission":"submission",
        # idle / standing
        "pre_match":"idle","break":"idle","waiting":"idle","stalling":"idle",
        "out_of_bounds":"idle","standing_reset":"standing","feet":"standing",
        "neutral_standing":"standing","grip_fighting":"standing","clinch":"standing",
        "pummeling":"standing","wrestling":"standing",
    }
    return A.get(t, t)

# ----------------------------------------------------------------------------
# 2. semantic family (finer than cluster) -- same family => "semantic" credit
# ----------------------------------------------------------------------------
SEM_FAMILY = {
    # "semantic" = the model named the right COARSE position class, wrong fine
    # label/granularity. Documented groupings (a coach would call these "the same
    # situation, wrong detail"):
    "guard":"guard", "half_guard":"guard",            # bottom playing guard (half = one leg passed)
    "side_control":"toppin", "mount":"toppin",        # dominant top pin (which pin differs)
    "knee_on_belly":"toppin", "north_south":"toppin", "top_control":"toppin",
    "back":"back", "turtle":"turtle",                  # kept DISTINCT: who-is-winning differs
    "leg_entanglement":"legs",                         # distinct leg-lock game
    "scramble":"scramble", "takedown":"takedown", "guard_pass":"pass",
    "guard_pull":"pull", "sweep":"sweep", "submission":"submission",
    "idle":"neutral", "standing":"neutral",            # non-engaged / neutral
}
# Conservative by design: back vs turtle and leg_entanglement are NOT folded
# (those are correctness-critical who's-winning distinctions), so they earn
# only cluster-adjacency credit, never semantic credit.

# ----------------------------------------------------------------------------
# 3. cluster (5 control states) + adjacency graph
# ----------------------------------------------------------------------------
CLUSTER = {
    "side_control":"Top control","mount":"Top control","top_control":"Top control",
    "knee_on_belly":"Top control","north_south":"Top control",
    "back":"Back/turtle","turtle":"Back/turtle",
    "guard":"Guard","half_guard":"Guard","leg_entanglement":"Guard",
    "scramble":"Transition","takedown":"Transition","guard_pull":"Transition",
    "guard_pass":"Transition","sweep":"Transition","submission":"Transition",
    "idle":"Idle","standing":"Idle",
}
# adjacency: the three ground-control states interconvert (pass/recover/take-back),
# Transition borders all engaged states, Idle borders only Transition (stand-up).
ADJ = {frozenset(p) for p in [
    ("Top control","Guard"), ("Top control","Back/turtle"), ("Back/turtle","Guard"),
    ("Transition","Top control"), ("Transition","Guard"),
    ("Transition","Back/turtle"), ("Transition","Idle"),
]}
def cluster(c): return CLUSTER.get(c, "?")
def adjacent(a, b):
    return a == b or frozenset((a, b)) in ADJ

# ----------------------------------------------------------------------------
# 4. correction_text parser  ->  actual canonical position
# ----------------------------------------------------------------------------
DASH = re.compile(r"\s+[—–-]\s+")          # em/en dash or " - " => reason
PAT_A = re.compile(r"^(.*?)[,;:]?\s+not\s+(.+)$", re.I)     # "<actual>, not <model>"
PAT_B = re.compile(r"^not\s+([^,]+),\s*(.+)$", re.I)        # "not <model>, <actual>"
KNOWN = ["half_guard","leg_entanglement","guard_pull","guard_pass","side_control",
         "knee_on_belly","north_south","back_control","top_control","submission",
         "guard","mount","back","turtle","scramble","takedown","sweep","standing",
         "idle","half","passing","clinch","pull"]
def kw_scan(text, avoid):
    """fallback: first known position phrase in free text, != model position."""
    raw = text.lower().replace("_"," ")
    best=None; bestpos=10**9
    for k in KNOWN:
        kk=k.replace("_"," ")
        i=raw.find(kk)
        if i>=0 and canon(k)!=avoid and i<bestpos:
            best, bestpos = canon(k), i
    return best

def parse_correction(ct, model_canon):
    head = DASH.split(ct.strip())[0]
    m = PAT_B.match(head)
    if m:                                   # "not <model>, <actual>"
        return canon(m.group(2)), "B"
    m = PAT_A.match(head)
    if m:                                   # "<actual>, not <model>"
        return canon(m.group(1)), "A"
    kw = kw_scan(ct, model_canon)
    if kw:
        return kw, "kw"
    return None, "unparsed"

# ----------------------------------------------------------------------------
# 5. score
# ----------------------------------------------------------------------------
def wilson(k, n, z=1.96):
    if n == 0: return (0.0, 0.0)
    p = k / n; den = 1 + z*z/n
    c = (p + z*z/(2*n)) / den
    h = z*math.sqrt(p*(1-p)/n + z*z/(4*n*n)) / den
    return (round(100*(c-h),1), round(100*(c+h),1))

def main():
    extra = {}
    if os.path.exists(EXTRA_EXTRACT):
        extra = json.load(open(EXTRA_EXTRACT))

    files = sorted(glob.glob(os.path.join(GT_DIR, "gt_v2_*.json")))
    N=acc=rej=unrev=0
    strict=semantic=adjacency=0
    methods=collections.Counter()
    unparsed=[]
    conf = collections.Counter()        # model_cluster|actual_cluster
    pairs = collections.Counter()       # (model_canon, actual_canon) for rejects
    sem_from_reject=0; adj_from_reject=0
    for f in files:
        d=json.load(open(f)); fid=os.path.basename(f)
        for r in d.get("segment_reviews") or []:
            st=r.get("status"); mp=canon(r.get("position",""))
            if st=="accepted":
                N+=1; acc+=1; strict+=1; semantic+=1; adjacency+=1
                conf[f"{cluster(mp)}|{cluster(mp)}"]+=1
            elif st=="rejected":
                N+=1; rej+=1
                key=f"{fid}::{r.get('index')}"
                if key in extra:
                    ap, meth = canon(extra[key]), "llm"
                else:
                    ap, meth = parse_correction(r.get("correction_text","") or "", mp)
                methods[meth]+=1
                if ap is None:
                    unparsed.append((fid, r.get("index"), (r.get("correction_text") or "")[:90]))
                    conf[f"{cluster(mp)}|?"]+=1
                    continue
                pairs[(mp,ap)]+=1
                conf[f"{cluster(mp)}|{cluster(ap)}"]+=1
                sem = SEM_FAMILY.get(mp) is not None and SEM_FAMILY.get(mp)==SEM_FAMILY.get(ap)
                adj = adjacent(cluster(mp), cluster(ap))
                if sem: semantic+=1; sem_from_reject+=1
                if sem or adj: adjacency+=1; adj_from_reject+=1
            elif st=="unreviewed":
                unrev+=1   # handled below (assumed correct)

    # unreviewed assumed correct (per protocol) -> add to all tiers
    Nf=N+unrev
    res = {
      "n_videos": len(files),
      "n_segments_reviewed": N,
      "n_unreviewed_assumed_correct": unrev,
      "counts": {"accepted":acc,"rejected":rej,"unreviewed":unrev},
      "extraction_methods": dict(methods),
      "n_unparsed_rejections": len(unparsed),
      # primary numbers: unreviewed assumed correct (denominator includes them)
      "strict_pct": round(100*(strict+unrev)/Nf,1),
      "semantic_pct": round(100*(semantic+unrev)/Nf,1),
      "adjacency_pct": round(100*(adjacency+unrev)/Nf,1),
      "strict_ci": wilson(strict+unrev,Nf),
      "semantic_ci": wilson(semantic+unrev,Nf),
      "adjacency_ci": wilson(adjacency+unrev,Nf),
      # reviewed-only (exclude the 20 unreviewed) -- robustness
      "reviewed_only": {
        "strict_pct": round(100*strict/N,1),
        "semantic_pct": round(100*semantic/N,1),
        "adjacency_pct": round(100*adjacency/N,1),
      },
      "semantic_credit_from_rejections": sem_from_reject,
      "adjacency_credit_from_rejections": adj_from_reject,
      "top_confusion_pairs": [
        {"model":m,"actual":a,"n":n} for (m,a),n in pairs.most_common(20)],
      "cluster_confusion_matrix": dict(conf),
      "cluster_order": ["Top control","Back/turtle","Guard","Transition","Idle"],
    }
    os.makedirs("benchmark/qa", exist_ok=True)
    json.dump(res, open("benchmark/qa/ood_v2_metrics.json","w"), indent=2)

    # also emit confusion_data.json in the figure's schema (n=200)
    cd = {"matrix":dict(conf),
          "order":["Top control","Back/turtle","Guard","Transition","Idle"],
          "pairs":[{"model":m,"actual":a,"n":n} for (m,a),n in pairs.most_common(25)],
          "n_videos":len(files),"n_segments":N}
    json.dump(cd, open("benchmark/qa/confusion_data_v2.json","w"), indent=2)

    print(f"videos={len(files)}  reviewed_segments={N}  (acc={acc} rej={rej} unrev={unrev})")
    print(f"extraction: {dict(methods)}  unparsed={len(unparsed)}")
    print(f"  STRICT    {res['strict_pct']}%  CI{res['strict_ci']}")
    print(f"  SEMANTIC  {res['semantic_pct']}%  CI{res['semantic_ci']}   (+{sem_from_reject} from rejects)")
    print(f"  ADJACENCY {res['adjacency_pct']}%  CI{res['adjacency_ci']}   (+{adj_from_reject} from rejects)")
    print(f"  reviewed-only: strict {res['reviewed_only']['strict_pct']}% "
          f"sem {res['reviewed_only']['semantic_pct']}% adj {res['reviewed_only']['adjacency_pct']}%")
    print("  top confusions:", [(m,a,n) for (m,a),n in pairs.most_common(8)])
    if unparsed:
        print(f"\n  {len(unparsed)} unparsed (need LLM extraction); sample:")
        for u in unparsed[:8]: print("    ", u[2])

if __name__=="__main__":
    main()
