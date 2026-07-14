#!/usr/bin/env python3
"""
V1 — Technique-ID accuracy scorer.
Direct Gemini File-API calls (gemini-2.5-flash) on ~12s MMA grappling clips.
Ontology-constrained prompt -> strict JSON {position, technique, confidence}.
Score against manifest gt_tags via ontology-normalized semantic matching.

Privacy: outputs are anonymized aggregate stats only. No user IDs / paths / raw
transcripts in paper-bound outputs. "MMA / label dataset" terminology only.
"""
import os, sys, json, time, threading, traceback, random
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

# ---------------------------------------------------------------------------
# Config
#
# Paths are configurable and contain no user-specific absolute path:
#   * benchmark/ is resolved relative to this script (results/ -> benchmark/).
#   * The clip directory holds the (withheld, private) technique clips; override
#     with the TECHNIQUE_CLIP_DIR env var.
#   * The API key is read from the environment (GOOGLE_API_KEY / GEMINI_API_KEY).
#     No key value or user-specific .env path is embedded in the bundle.
# ---------------------------------------------------------------------------
_BENCH     = Path(__file__).resolve().parent.parent      # .../benchmark
CLIP_DIR   = Path(os.environ.get("TECHNIQUE_CLIP_DIR")
                  or (_BENCH / "data" / "technique_clips"))
MANIFEST   = CLIP_DIR / "manifest.json"
RESULTS_DIR= _BENCH / "results"
MODEL      = "models/gemini-2.5-flash"
MAX_INFLIGHT = 4
TIME_BUDGET_SEC = int(os.environ.get("VLM_TIME_BUDGET", "2100"))   # ~35 min default
MAX_CLIPS  = int(os.environ.get("VLM_MAX_CLIPS", "119"))           # cap if needed
RETRIES    = 5

# ---------------------------------------------------------------------------
# Load API key from the environment (no key value / .env path embedded here).
# Export GOOGLE_API_KEY (or GEMINI_API_KEY) before running; if you keep your key
# in an external .env, source it into the environment first, e.g.
#   set -a; source /path/to/your/.env; set +a
# ---------------------------------------------------------------------------
def load_env():
    key = os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")
    if not key:
        raise RuntimeError(
            "No API key found. Set GOOGLE_API_KEY (or GEMINI_API_KEY) in the "
            "environment before running."
        )
    os.environ["GOOGLE_API_KEY"] = key
    os.environ["GEMINI_API_KEY"] = key
    return key

KEY = load_env()
from google import genai
from google.genai import types
client = genai.Client(api_key=KEY)

# ---------------------------------------------------------------------------
# Ontology -> the constrained label space we ask the model to pick from
# ---------------------------------------------------------------------------
POSITIONS = [
    "standing", "guard", "open_guard", "closed_guard", "half_guard", "mount",
    "side_control", "back_control", "north_south", "knee_on_belly", "turtle",
    "clinch", "passing_guard", "leg_entanglement", "scramble", "unknown",
]

# technique / action vocabulary the model chooses from (concise, ontology-derived)
TECHNIQUES = [
    # takedowns / standing
    "single_leg", "double_leg", "ankle_pick", "takedown", "takedown_defense",
    "guard_pull", "body_lock", "underhook", "grip_fight", "hand_fight", "kuzushi",
    "wrestle_up",
    # passing
    "guard_pass", "knee_slice", "leg_drag", "stack_pass", "toreando_pass",
    "over_under_pass", "half_guard_pass", "outside_passing", "loose_passing",
    # sweeps / reversals
    "sweep", "scramble_reversal", "destabilization",
    # back / control entries
    "back_take", "mount_entry", "side_control_entry", "turtle_entry", "go_behind",
    # leg locks / entanglement
    "leg_entanglement", "heel_hook", "ankle_lock", "knee_bar", "leg_lock",
    "leg_lock_defense",
    # submissions
    "armbar", "kimura", "triangle", "guillotine", "rear_naked_choke", "kata_gatame",
    "arm_triangle", "submission", "submission_escape",
    # guard play
    "open_guard", "seated_open_guard", "guard_retention", "knee_shield",
    # other
    "escape", "control", "unknown",
]

POS_LIST_STR = ", ".join(POSITIONS)
TECH_LIST_STR = ", ".join(TECHNIQUES)

PROMPT = f"""This is a ~12s MMA grappling clip. Watch it carefully and identify:
(a) the dominant body position, and
(b) the primary technique/action being performed.

Choose the POSITION from exactly this list:
{POS_LIST_STR}

Choose the TECHNIQUE from exactly this list:
{TECH_LIST_STR}

Pick the single best-fitting value from each list (use "unknown" only if truly unclear).
Output STRICT JSON on a single line, no markdown, no commentary:
{{"position": "<one of the position list>", "technique": "<one of the technique list>", "confidence": <float 0-1>}}"""

# ---------------------------------------------------------------------------
# Semantic normalization: map predicted labels AND gt hashtags to canonical
# concept families so we can do semantic matching.
# ---------------------------------------------------------------------------

# Canonical POSITION families. Each maps to a set of gt-hashtag substrings/tokens
# that indicate that position is present in the clip's GT.
POSITION_FAMILIES = {
    "standing":       {"standing", "wrestling", "takedown", "singleleg", "doubleleg",
                       "gripfight", "handfight", "clinch", "bodylock", "kuzushi",
                       "wrestleup", "counterwrestling", "takedowndefense", "ankle pick",
                       "anklepick", "footsweep", "snapdown", "headandarmgrip", "collartie",
                       "elbowtie", "underhook", "duckunder", "armdrag"},
    "guard":          {"guard", "openguard", "closedguard", "seatedopenguard",
                       "guardretention", "guardpull", "pullguard", "kneeshield",
                       "delariva", "dlr", "rdlr", "xguard", "butterfly", "spiderguard",
                       "lassoguard", "halfguard", "zguard", "deephalf"},
    "open_guard":     {"openguard", "seatedopenguard", "delariva", "xguard", "butterfly",
                       "singlelegx", "slx", "spiderguard", "lassoguard"},
    "seated_open_guard": {"seatedopenguard", "openguard", "singlelegx", "slx", "butterfly",
                       "delariva", "xguard", "spiderguard"},
    "closed_guard":   {"closedguard"},
    "half_guard":     {"halfguard", "halfguardpass", "kneeshield", "lockdown", "deephalf"},
    "mount":          {"mount", "mountentry", "highmount", "lowmount", "smount", "technicalmount"},
    "side_control":   {"sidecontrol", "sidecontrolentry", "kesa", "katagatame", "headandarm"},
    "back_control":   {"back", "backtake", "backcontrol", "seatbelt", "bodytriangle",
                       "hooksin", "rearnakedchoke", "rnc", "gobehind", "turtle", "turtleentry"},
    "north_south":    {"northsouth", "north-south"},
    "knee_on_belly":  {"kneeonbelly", "kob", "kneeonstomach"},
    "turtle":         {"turtle", "turtleentry", "turtlemaintenance", "matreturn", "breakdown"},
    "clinch":         {"clinch", "bodylock", "overunder", "collartie", "underhook", "pummel"},
    "passing_guard":  {"guardpass", "pass", "passing", "loosepassing", "outsidepassing",
                       "halfguardpass", "kneeslice", "legdrag", "torreando", "toreando",
                       "stackpass", "openguardpass", "supineopenguardpass", "kneeshieldpass",
                       "chesttochesthalfguardpass", "bodylockpass", "smashpass", "longstep"},
    "leg_entanglement": {"legentanglement", "ashi", "ashigarami", "slx", "singlelegx",
                       "xguard", "5050", "fiftyfifty", "irimiashi", "outsideashi",
                       "innerashi", "reap", "leglock", "heelhook", "kneebar", "kneereap",
                       "saddle", "honeyhole", "411", "outsidecamping", "camping",
                       "crossleg", "doubletrouble", "dilemma"},
    "scramble":       {"scramble", "scrambling", "reversal", "wrestleup"},
    "unknown":        set(),
}

# Canonical TECHNIQUE families: predicted technique -> set of gt-hashtag tokens
# (substring match) that count as a correct semantic hit.
TECHNIQUE_FAMILIES = {
    "single_leg":       {"singleleg", "singlelegtakedown", "singlelegx"},  # note: slx tagged distinctly below
    "double_leg":       {"doubleleg", "doublelegtakedown"},
    "ankle_pick":       {"anklepick"},
    "takedown":         {"takedown", "wrestling", "singleleg", "doubleleg", "footsweep",
                         "throw", "hiptoss", "trip", "bodylocktakedown", "snapdown"},
    "takedown_defense": {"takedowndefense", "counterwrestling", "sprawl", "defense"},
    "guard_pull":       {"guardpull", "pullguard"},
    "body_lock":        {"bodylock", "bodylockentry", "rearbodylockentry", "rearbodylock"},
    "underhook":        {"underhook", "underhooks", "doubleunders"},
    "grip_fight":       {"gripfight", "handfight", "gripfighting", "elbowtie", "collartie",
                         "headandarmgrip", "headandarm", "twoonone", "2on1"},
    "hand_fight":       {"handfight", "gripfight", "collartie", "elbowtie"},
    "kuzushi":          {"kuzushi", "destabilization", "offbalance", "offbalancing"},
    "wrestle_up":       {"wrestleup", "leglockwrestleupdilemma"},
    "guard_pass":       {"guardpass", "pass", "passing", "loosepassing", "outsidepassing",
                         "openguardpass", "supineopenguardpass", "kneeslice", "legdrag",
                         "toreando", "torreando", "stackpass", "bodylockpass", "smashpass"},
    "knee_slice":       {"kneeslice", "kneecut"},
    "leg_drag":         {"legdrag"},
    "stack_pass":       {"stackpass", "stack"},
    "toreando_pass":    {"toreando", "torreando"},
    "over_under_pass":  {"overunder", "overunderpass"},
    "half_guard_pass":  {"halfguardpass", "chesttochesthalfguardpass", "kneeshieldpass"},
    "outside_passing":  {"outsidepassing", "outsidecamping", "camping"},
    "loose_passing":    {"loosepassing"},
    "sweep":            {"sweep", "reversal", "scramblereversal", "hipbump", "scissor",
                         "butterflysweep", "pendulum"},
    "scramble_reversal":{"scramble", "reversal", "scramblereversal"},
    "destabilization":  {"destabilization", "kuzushi", "offbalance", "breakdown"},
    "back_take":        {"backtake", "back", "gobehind", "backtakepinningdilemma",
                         "seatbelt", "rearbodylock", "backcontrol"},
    "mount_entry":      {"mountentry", "mount", "34mountentry"},
    "side_control_entry":{"sidecontrolentry", "sidecontrol"},
    "turtle_entry":     {"turtleentry", "turtle", "matreturn", "turtlemaintenance", "breakdown"},
    "go_behind":        {"gobehind", "backtake"},
    "leg_entanglement": {"legentanglement", "ashi", "ashigarami", "slx", "singlelegx",
                         "irimiashi", "outsideashi", "innerashi", "saddle", "honeyhole",
                         "5050", "fiftyfifty", "crossleg", "doubletrouble", "reap"},
    "heel_hook":        {"heelhook", "outsideheelhook", "insideheelhook", "heelexposure"},
    "ankle_lock":       {"anklelock", "achilles", "achillesgrip", "straightankle", "aoki"},
    "knee_bar":         {"kneebar"},
    "leg_lock":         {"leglock", "heelhook", "kneebar", "anklelock", "toehold",
                         "legentanglement", "ashi", "reap", "leglockwrestleupdilemma"},
    "leg_lock_defense": {"leglockdefense", "legdefense", "bootstrap", "hideheel"},
    "armbar":           {"armbar", "jujigatame", "armlock"},
    "kimura":           {"kimura"},
    "triangle":         {"triangle", "trianglechoke"},
    "guillotine":       {"guillotine"},
    "rear_naked_choke": {"rearnakedchoke", "rnc", "strangle"},
    "kata_gatame":      {"katagatame", "armtriangle", "headandarmchoke"},
    "arm_triangle":     {"armtriangle", "katagatame", "headandarmchoke"},
    "submission":       {"submission", "tap", "finish", "choke", "armbar", "heelhook",
                         "kimura", "triangle", "guillotine", "rearnakedchoke", "armtriangle",
                         "katagatame"},
    "submission_escape":{"submissionescape", "escape", "collartieescape", "singlelegescape",
                         "leglockdefense"},
    "open_guard":       {"openguard", "seatedopenguard", "delariva", "xguard", "butterfly"},
    "seated_open_guard":{"seatedopenguard", "seated"},
    "guard_retention":  {"guardretention", "retention", "reguard", "recover"},
    "knee_shield":      {"kneeshield"},
    "escape":           {"escape", "submissionescape", "collartieescape", "singlelegescape",
                         "guardrecovery", "reguard"},
    "control":          {"control", "maintenance", "pin", "pinning", "camping"},
    "unknown":          set(),
}

def gt_token_set(gt_tags):
    """Lowercased set of gt hashtags (already lowercase, alnum)."""
    return {t.lower() for t in gt_tags}

def gt_blob(gt_tags):
    """One concatenated string for substring search."""
    return " " + " ".join(t.lower() for t in gt_tags) + " "

def family_hits(family_tokens, gt_tags):
    """True if any family token matches a gt tag (exact token or substring of a tag)."""
    if not family_tokens:
        return False
    gts = gt_token_set(gt_tags)
    blob = gt_blob(gt_tags)
    for tok in family_tokens:
        tok = tok.replace(" ", "")
        if tok in gts:
            return True
        # substring: token appears inside any compound hashtag
        if tok in blob:
            return True
    return False

def score_position(pred_pos, gt_tags):
    pred_pos = (pred_pos or "").strip().lower().replace(" ", "_")
    fam = POSITION_FAMILIES.get(pred_pos)
    if fam is None:
        # loose normalization: prefer the longest (most specific) key match
        cands = [k for k in POSITION_FAMILIES if k in pred_pos or pred_pos in k]
        if cands:
            k = max(cands, key=len)
            fam = POSITION_FAMILIES[k]; pred_pos = k
    if fam is None:
        return False, pred_pos
    # "guard" / "open_guard" etc. — also accept the generic 'guard' umbrella
    hit = family_hits(fam, gt_tags)
    # umbrella: any guard-variant pred matches if generic guard family present
    if not hit and pred_pos in ("guard", "open_guard", "closed_guard", "half_guard",
                                 "seated_open_guard"):
        hit = family_hits(POSITION_FAMILIES["guard"], gt_tags)
    return hit, pred_pos

def score_technique(pred_tech, gt_tags):
    pred_tech = (pred_tech or "").strip().lower().replace(" ", "_")
    fam = TECHNIQUE_FAMILIES.get(pred_tech)
    if fam is None:
        cands = [k for k in TECHNIQUE_FAMILIES if k in pred_tech or pred_tech in k]
        if cands:
            k = max(cands, key=len)
            fam = TECHNIQUE_FAMILIES[k]; pred_tech = k
    if fam is None:
        return False, pred_tech
    return family_hits(fam, gt_tags), pred_tech

# ---------------------------------------------------------------------------
# Gemini per-clip call: upload -> poll ACTIVE -> generate -> delete
# ---------------------------------------------------------------------------
def parse_json(text):
    text = (text or "").strip()
    # strip markdown fences
    if text.startswith("```"):
        text = text.strip("`")
        if text.lower().startswith("json"):
            text = text[4:]
    # find first { ... }
    s = text.find("{"); e = text.rfind("}")
    if s != -1 and e != -1 and e > s:
        frag = text[s:e+1]
        try:
            return json.loads(frag)
        except Exception:
            pass
    return None

usage_lock = threading.Lock()
USAGE = {"prompt_tokens": 0, "cand_tokens": 0, "calls": 0}

def is_retryable(err):
    s = str(err).lower()
    return any(x in s for x in ("429", "rate", "quota", "resource_exhausted",
                                "500", "503", "internal", "unavailable", "deadline",
                                "timeout"))

def process_clip(clip_path: Path, gt_tags):
    """Returns dict result for one clip (never raises)."""
    name = clip_path.name
    uploaded = None
    for attempt in range(RETRIES):
        try:
            # upload
            uploaded = client.files.upload(file=str(clip_path))
            # poll until ACTIVE
            t0 = time.time()
            while uploaded.state.name == "PROCESSING":
                if time.time() - t0 > 180:
                    raise TimeoutError("file processing >180s")
                time.sleep(2)
                uploaded = client.files.get(name=uploaded.name)
            if uploaded.state.name != "ACTIVE":
                raise ValueError(f"file state {uploaded.state.name}")
            # generate
            resp = client.models.generate_content(
                model=MODEL,
                contents=[uploaded, PROMPT],
                config=types.GenerateContentConfig(
                    temperature=0.0,
                    max_output_tokens=256,
                    thinking_config=types.ThinkingConfig(thinking_budget=0),
                ),
            )
            # usage
            if getattr(resp, "usage_metadata", None):
                with usage_lock:
                    USAGE["prompt_tokens"] += resp.usage_metadata.prompt_token_count or 0
                    USAGE["cand_tokens"]   += resp.usage_metadata.candidates_token_count or 0
                    USAGE["calls"] += 1
            parsed = parse_json(resp.text)
            # cleanup file
            try:
                client.files.delete(name=uploaded.name)
            except Exception:
                pass
            if parsed is None:
                return {"clip": name, "ok": False, "error": "json_parse_fail",
                        "raw": (resp.text or "")[:200], "gt_tags": gt_tags}
            return {"clip": name, "ok": True,
                    "pred_position": parsed.get("position"),
                    "pred_technique": parsed.get("technique"),
                    "confidence": parsed.get("confidence"),
                    "gt_tags": gt_tags}
        except Exception as e:
            # cleanup file on failure before retry
            try:
                if uploaded is not None:
                    client.files.delete(name=uploaded.name)
            except Exception:
                pass
            uploaded = None
            if attempt < RETRIES - 1 and is_retryable(e):
                backoff = (2 ** attempt) + random.uniform(0, 1.5)
                time.sleep(backoff)
                continue
            return {"clip": name, "ok": False, "error": str(e)[:200], "gt_tags": gt_tags}
    return {"clip": name, "ok": False, "error": "exhausted_retries", "gt_tags": gt_tags}

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    manifest = json.load(open(MANIFEST))
    clips = []
    for fname, entry in manifest.items():
        p = CLIP_DIR / fname
        if p.exists():
            clips.append((p, entry.get("gt_tags", [])))
    clips = clips[:MAX_CLIPS]
    total = len(clips)
    print(f"[INFO] {total} clips queued (model={MODEL}, inflight={MAX_INFLIGHT}, "
          f"budget={TIME_BUDGET_SEC}s)")

    results = []
    start = time.time()
    submitted = 0
    with ThreadPoolExecutor(max_workers=MAX_INFLIGHT) as ex:
        futures = {}
        it = iter(clips)
        # prime
        for _ in range(min(MAX_INFLIGHT, total)):
            p, gt = next(it)
            futures[ex.submit(process_clip, p, gt)] = p.name
            submitted += 1
        done_count = 0
        while futures:
            for fut in as_completed(list(futures.keys())):
                name = futures.pop(fut)
                try:
                    res = fut.result()
                except Exception as e:
                    res = {"clip": name, "ok": False, "error": f"future:{e}"[:200], "gt_tags": []}
                results.append(res)
                done_count += 1
                status = "OK" if res.get("ok") else "FAIL"
                print(f"[{done_count}/{total}] {name} {status} "
                      f"pos={res.get('pred_position')} tech={res.get('pred_technique')}")
                # submit next if budget allows
                elapsed = time.time() - start
                if elapsed < TIME_BUDGET_SEC and submitted < total:
                    try:
                        p, gt = next(it)
                        futures[ex.submit(process_clip, p, gt)] = p.name
                        submitted += 1
                    except StopIteration:
                        pass
                elif elapsed >= TIME_BUDGET_SEC:
                    print(f"[WARN] time budget reached at {elapsed:.0f}s; not submitting more")
                break  # re-evaluate as_completed with updated futures set
    elapsed = time.time() - start

    # ----------------------------------------------------------------- scoring
    scored = [r for r in results if r.get("ok")]
    failed = [r for r in results if not r.get("ok")]
    n_scored = len(scored)

    pos_hits = 0
    tech_hits = 0
    overall_hits = 0
    per_clip = []
    # per-tag accuracy uses the manifest target tags (meaningful) as denominator.
    target_tags = ["standing","guardpass","singleleg","destabilization","turtleentry",
                   "underhook","wrestling","legentanglement","takedown","mount",
                   "openguard","backtake","sweep","submission","gripfight"]
    per_tag = {t: {"clips": 0, "pos_hit": 0, "tech_hit": 0, "either_hit": 0} for t in target_tags}

    for r in scored:
        gt = r["gt_tags"]
        ph, npos = score_position(r["pred_position"], gt)
        th, ntech = score_technique(r["pred_technique"], gt)
        pos_hits += int(ph)
        tech_hits += int(th)
        overall_hits += int(ph and th)
        per_clip.append({
            "clip": r["clip"],
            "pred_position": r["pred_position"], "norm_position": npos,
            "pred_technique": r["pred_technique"], "norm_technique": ntech,
            "confidence": r.get("confidence"),
            "position_correct": ph, "technique_correct": th,
        })
        gset = {t.lower() for t in gt}
        for t in target_tags:
            if t in gset:
                per_tag[t]["clips"] += 1
                per_tag[t]["pos_hit"] += int(ph)
                per_tag[t]["tech_hit"] += int(th)
                per_tag[t]["either_hit"] += int(ph or th)

    def pct(a, b):
        return round(100.0 * a / b, 1) if b else 0.0

    pos_acc = pct(pos_hits, n_scored)
    tech_acc = pct(tech_hits, n_scored)
    overall_acc = pct(overall_hits, n_scored)

    per_tag_out = {}
    for t in target_tags:
        d = per_tag[t]
        per_tag_out[t] = {
            "clips": d["clips"],
            "position_acc_pct": pct(d["pos_hit"], d["clips"]),
            "technique_match_pct": pct(d["tech_hit"], d["clips"]),
            "either_match_pct": pct(d["either_hit"], d["clips"]),
        }

    # ----------------------------------------------------------------- cost
    # gemini-2.5-flash pricing (per 1M tokens): input $0.30, output $2.50 (incl video).
    in_cost = USAGE["prompt_tokens"] / 1e6 * 0.30
    out_cost = USAGE["cand_tokens"] / 1e6 * 2.50
    approx_cost = round(in_cost + out_cost, 4)

    summary = {
        "task": "V1_technique_id_accuracy",
        "model": MODEL,
        "domain": "MMA grappling (label dataset)",
        "n_clips_total_on_disk": total,
        "n_clips_scored": n_scored,
        "n_clips_failed": len(failed),
        "position_top1_accuracy_pct": pos_acc,
        "technique_match_rate_pct": tech_acc,
        "overall_accuracy_pct": overall_acc,
        "matching": "ontology-normalized semantic matching against gt_tags",
        "per_tag_accuracy_top_tags": per_tag_out,
        "wall_clock_sec": round(elapsed, 1),
        "api_calls": USAGE["calls"],
        "prompt_tokens": USAGE["prompt_tokens"],
        "candidate_tokens": USAGE["cand_tokens"],
        "approx_cost_usd": approx_cost,
        "failures": [{"clip": f["clip"], "error": f.get("error")} for f in failed],
    }

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    full = {"summary": summary, "per_clip": per_clip}
    with open(RESULTS_DIR / "vlm_technique_id.json", "w") as f:
        json.dump(full, f, indent=2)

    # markdown
    md = []
    md.append("# VLM Technique-ID Accuracy (V1)\n")
    md.append(f"- Model: `{MODEL}`")
    md.append(f"- Domain: MMA grappling label dataset (~12s clips)")
    md.append(f"- Matching: ontology-normalized semantic matching vs `gt_tags`")
    md.append(f"- Clips scored: **{n_scored}** / {total} on disk ({len(failed)} failed/skipped)")
    md.append(f"- Wall-clock: {round(elapsed,1)}s | API calls: {USAGE['calls']} | approx cost: **${approx_cost}**\n")
    md.append("## Headline accuracy\n")
    md.append("| Metric | Value |")
    md.append("|---|---|")
    md.append(f"| Position top-1 accuracy | **{pos_acc}%** |")
    md.append(f"| Technique match rate (pred in GT set, semantic) | **{tech_acc}%** |")
    md.append(f"| Overall (position AND technique correct) | **{overall_acc}%** |\n")
    md.append("## Per-tag accuracy (top tags by clip count)\n")
    md.append("| Tag | Clips | Position acc | Technique match | Either match |")
    md.append("|---|---|---|---|---|")
    for t in sorted(target_tags, key=lambda x: -per_tag_out[x]["clips"]):
        d = per_tag_out[t]
        md.append(f"| {t} | {d['clips']} | {d['position_acc_pct']}% | "
                  f"{d['technique_match_pct']}% | {d['either_match_pct']}% |")
    md.append("\n## Notes\n")
    md.append("- Anonymized aggregate stats only; no user IDs, paths, or raw transcripts.")
    md.append("- `gt_tags` are noisy multi-label hashtags (incl. non-technique tokens); "
              "scoring uses semantic family mapping so a prediction counts as correct if it "
              "names a position/technique actually present in the clip's GT tag set.")
    md.append(f"- Failures/skips: {len(failed)}"
              + ("" if not failed else " (see JSON `failures`)"))
    with open(RESULTS_DIR / "vlm_technique_id.md", "w") as f:
        f.write("\n".join(md) + "\n")

    print("\n=== SUMMARY ===")
    print(json.dumps(summary, indent=2)[:2000])

if __name__ == "__main__":
    main()
