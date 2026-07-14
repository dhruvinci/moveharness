#!/usr/bin/env python3
"""HMM CV-only baseline for MoveBench v0 (X1).

Removes the paper's "HMM specified but not run" hedge by actually fitting and
decoding a multivariate Gaussian-emission HMM over the CV anchor features that
the Exp4 stage0 pass cached for the v0 match (mb_v0_0001, ~970s, 194
checkpoints at 5s), then scoring the decoded macro-position sequence against the
MoveBench v0 ground truth with the bundled MoveBench scorer.

Pipeline (no model inference, no network):
  1. Load the 194 CV checkpoints. Build a 5-dim anchor feature matrix:
       standing_prob (avg of A1/A2), contact_intensity, action_score,
       dominance (signed: A1=+1, A2=-1, N=0, abs-folded -> |dominance|),
       entanglement. Impute missing standing_prob (athlete not detected) with
       the column mean; z-score all columns.
  2. Fit a full-covariance Gaussian HMM for K = 2..8 hidden states (multiple
     random restarts, best log-likelihood per K). Select #states by AIC.
  3. Viterbi-decode the macro-position state sequence at the selected K.
  4. Collapse the per-checkpoint state path into contiguous segments
     (start_sec/end_sec/state).
  5. Map each HMM state to a GT v0 macro position by majority temporal overlap
     (the standard, fair "cluster -> class" assignment for an UNSUPERVISED
     decoder; this is the CV-only *ceiling*, reported honestly as oracle-mapped).
  6. Emit a MoveBench prediction JSON and score it with the bundled scorer
     (T1 macro-position macro-F1 + duration-weighted F1 + mean IoU; T3 labels
     are intentionally empty -- a CV-only macro-position decoder emits no
     technique labels).

Outputs (under benchmark/results/):
  real_hmm_baseline.json   -- full machine record (AIC table, mapping, scores)
  real_hmm_baseline.md     -- human-readable report
  pred_hmm_cv_baseline.json (under predictions_all/) -- the scored prediction

Standard-lib + numpy + hmmlearn only. Fully deterministic (fixed seeds).
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import warnings
from collections import Counter, defaultdict

import numpy as np

# hmmlearn emits per-restart "Model is not converging" warnings; we keep the
# best-logL restart per K, so these are expected noise. Silence for clean logs.
warnings.filterwarnings("ignore")
try:
    from hmmlearn.base import ConvergenceMonitor  # noqa: F401
    import logging
    logging.getLogger("hmmlearn").setLevel(logging.ERROR)
except Exception:
    pass

# Project root: configurable via the MOVEBENCH_ROOT env var; defaults to the repo
# root resolved relative to this script (results/ -> benchmark/ -> repo root), so
# the script is portable and contains no user-specific absolute path.
_HERE = os.path.dirname(os.path.abspath(__file__))
_BENCH = os.path.normpath(os.path.join(_HERE, os.pardir))  # .../benchmark
ROOT = os.environ.get("MOVEBENCH_ROOT") or os.path.normpath(os.path.join(_BENCH, os.pardir))
# The Stage0 CV anchor cache is a (withheld) private artifact; override its
# location with the STAGE0_CV_CACHE env var if you hold it.
CV_CACHE = os.environ.get("STAGE0_CV_CACHE") or os.path.join(
    ROOT,
    "repos/vlm-experiments/outputs/experiment4.0/stage0_cv_checkpoints_5s.json",
)
GT_PATH = os.path.join(_BENCH, "data/v0/mb_v0_0001.json")
RESULTS_DIR = os.path.join(_BENCH, "results")
PRED_PATH = os.path.join(RESULTS_DIR, "predictions_all/pred_hmm_cv_baseline.json")
JSON_OUT = os.path.join(RESULTS_DIR, "real_hmm_baseline.json")
MD_OUT = os.path.join(RESULTS_DIR, "real_hmm_baseline.md")
MOVEBENCH_DIR = _BENCH  # contains the `movebench` package

SEED = 0
K_RANGE = range(2, 9)
N_RESTARTS = 6
np.random.seed(SEED)


# --------------------------------------------------------------------------- #
# 1. Load + featurize CV checkpoints
# --------------------------------------------------------------------------- #
def _get(d, *keys):
    x = d
    for k in keys:
        if not isinstance(x, dict):
            return None
        x = x.get(k)
    return x


def load_features():
    cv = json.load(open(CV_CACHE))
    cks = cv["checkpoints"]
    times = np.array([float(c["time_sec"]) for c in cks], dtype=float)
    interval = float(cv.get("checkpoint_interval") or 5)
    video_dur = float(cv.get("video_duration") or (times[-1] + interval))

    sp1 = [_get(c, "athlete_1", "standing_prob") for c in cks]
    sp2 = [_get(c, "athlete_2", "standing_prob") for c in cks]
    ci = [_get(c, "interaction", "contact_intensity") for c in cks]
    asc = [c.get("action_score") for c in cks]
    ent = [_get(c, "interaction", "entanglement") for c in cks]
    dom = [_get(c, "interaction", "dominant") for c in cks]

    # standing_prob: average of both athletes; if one missing, use the other;
    # if both missing, leave NaN for column-mean imputation.
    def avg_sp(a, b):
        vals = [v for v in (a, b) if v is not None]
        return float(np.mean(vals)) if vals else np.nan

    standing = np.array([avg_sp(a, b) for a, b in zip(sp1, sp2)], dtype=float)
    contact = np.array([float(v) if v is not None else np.nan for v in ci], dtype=float)
    action = np.array([float(v) if v is not None else np.nan for v in asc], dtype=float)
    entang = np.array([float(v) if v is not None else 0.0 for v in ent], dtype=float)
    # dominance magnitude: how lopsided the position is (N=neutral=0, A1/A2=1).
    dom_mag = np.array([0.0 if (v is None or v == "N") else 1.0 for v in dom], dtype=float)

    X = np.column_stack([standing, contact, action, dom_mag, entang])
    feat_names = [
        "standing_prob",
        "contact_intensity",
        "action_score",
        "dominance",
        "entanglement",
    ]
    # column-mean imputation for NaNs
    n_imputed = {}
    for j, name in enumerate(feat_names):
        col = X[:, j]
        mask = np.isnan(col)
        n_imputed[name] = int(mask.sum())
        if mask.any():
            col[mask] = np.nanmean(col)
            X[:, j] = col
    # z-score
    mu = X.mean(axis=0)
    sd = X.std(axis=0)
    sd[sd == 0] = 1.0
    Xz = (X - mu) / sd
    return Xz, times, interval, video_dur, feat_names, n_imputed


# --------------------------------------------------------------------------- #
# 2. Fit HMMs across K, select by AIC
# --------------------------------------------------------------------------- #
COV_TYPE = "diag"  # robust for D=5, N=194; full-cov overfits (degenerate spikes)


def n_params(K, D, cov_type=COV_TYPE):
    """Free params of a Gaussian HMM.

    transitions: K*(K-1); start probs: K-1; means: K*D; covariances depend on
    type: diag -> K*D; full -> K * D*(D+1)/2.
    """
    base = K * (K - 1) + (K - 1) + K * D
    if cov_type == "diag":
        return base + K * D
    return base + K * (D * (D + 1) // 2)


def fit_select(Xz):
    from hmmlearn import hmm

    D = Xz.shape[1]
    N = Xz.shape[0]
    table = []
    best_models = {}
    for K in K_RANGE:
        best_ll = -np.inf
        best_m = None
        for r in range(N_RESTARTS):
            m = hmm.GaussianHMM(
                n_components=K,
                covariance_type=COV_TYPE,
                n_iter=300,
                tol=1e-4,
                random_state=SEED + r,
                init_params="stmc",
                min_covar=1e-2,  # floor to prevent degenerate variance spikes
            )
            try:
                m.fit(Xz)
                ll = m.score(Xz)
            except Exception as e:  # numerical failure on a restart -> skip
                continue
            if np.isfinite(ll) and ll > best_ll:
                best_ll = ll
                best_m = m
        if best_m is None:
            table.append({"K": K, "logL": None, "n_params": None, "AIC": None, "BIC": None})
            continue
        p = n_params(K, D)
        aic = 2 * p - 2 * best_ll
        bic = p * np.log(N) - 2 * best_ll
        table.append(
            {
                "K": K,
                "logL": float(best_ll),
                "n_params": int(p),
                "AIC": float(aic),
                "BIC": float(bic),
            }
        )
        best_models[K] = best_m
    valid = [row for row in table if row["AIC"] is not None]
    best_K = min(valid, key=lambda r: r["AIC"])["K"]
    return table, best_K, best_models[best_K]


# --------------------------------------------------------------------------- #
# 3-4. Viterbi decode -> contiguous segments
# --------------------------------------------------------------------------- #
def smooth_runs(states, min_run=2):
    """Min-duration smoothing of a Viterbi path: any contiguous run shorter than
    `min_run` checkpoints is absorbed into the longer of its two neighbors (or
    the only neighbor at an edge). Standard segmentation post-processing -- macro
    grappling positions persist for tens of seconds, so single-checkpoint (5s)
    state flickers are decoder noise, not real position changes. Tuned to the
    feature time-scale (min_run=2 -> 10s), NOT to the GT. Iterate to fixpoint."""
    s = list(int(x) for x in states)
    changed = True
    while changed:
        changed = False
        # build runs
        runs = []
        i = 0
        while i < len(s):
            j = i
            while j + 1 < len(s) and s[j + 1] == s[i]:
                j += 1
            runs.append([i, j, s[i]])
            i = j + 1
        for ri, (a, b, st) in enumerate(runs):
            if (b - a + 1) >= min_run:
                continue
            left = runs[ri - 1] if ri > 0 else None
            right = runs[ri + 1] if ri + 1 < len(runs) else None
            if left and right:
                target = left[2] if (left[1] - left[0]) >= (right[1] - right[0]) else right[2]
            elif left:
                target = left[2]
            elif right:
                target = right[2]
            else:
                continue
            for k in range(a, b + 1):
                s[k] = target
            changed = True
            break  # re-run from scratch after a merge
    return np.array(s, dtype=int)


def decode_segments(states, times, interval, video_dur):
    segs = []
    n = len(states)
    i = 0
    while i < n:
        j = i
        while j + 1 < n and states[j + 1] == states[i]:
            j += 1
        start = float(times[i])
        # end = start of next checkpoint, or video end for the last run
        end = float(times[j + 1]) if j + 1 < n else min(float(times[j]) + interval, video_dur)
        segs.append({"start_sec": start, "end_sec": end, "state": int(states[i])})
        i = j + 1
    # snap final end to video duration to maximize coverage
    if segs:
        segs[-1]["end_sec"] = float(video_dur)
        segs[0]["start_sec"] = 0.0
    return segs


# --------------------------------------------------------------------------- #
# 5. Map HMM states -> GT macro positions (majority temporal overlap)
# --------------------------------------------------------------------------- #
def overlap(a0, a1, b0, b1):
    return max(0.0, min(a1, b1) - max(a0, b0))


def map_states_to_positions(segs, gt_segs):
    """For each HMM state, accumulate temporal overlap with each GT position;
    assign the position with the largest total overlap (oracle cluster->class)."""
    state_pos_overlap = defaultdict(lambda: defaultdict(float))
    for s in segs:
        for g in gt_segs:
            ov = overlap(s["start_sec"], s["end_sec"], g["start_sec"], g["end_sec"])
            if ov > 0:
                state_pos_overlap[s["state"]][g["position"]] += ov
    mapping = {}
    for st, posd in state_pos_overlap.items():
        mapping[st] = max(posd.items(), key=lambda kv: kv[1])[0]
    # states with no overlap (shouldn't happen) -> most common GT position
    return mapping


# --------------------------------------------------------------------------- #
# Pseudo-residual fit diagnostic (cheap): uniform-residual normality.
# For a fitted HMM we use the one-step-ahead (filtering) predictive CDF via the
# posterior state probabilities and per-state Gaussian marginals on feature 0
# (standing_prob), then map through the normal quantile -> if the model fits,
# the pseudo-residuals are ~ N(0,1). We report the Shapiro-Wilk p-value.
# --------------------------------------------------------------------------- #
def pseudo_residuals(model, Xz):
    try:
        from scipy.stats import norm, shapiro

        post = model.predict_proba(Xz)  # (N,K) smoothed posteriors
        # per-state marginal mean/var on feature 0
        means = model.means_[:, 0]
        cv = np.asarray(model.covars_)
        # covars_ shape: full -> (K,D,D); diag -> (K,D). Pull feature-0 variance.
        if cv.ndim == 3:
            covs = cv[:, 0, 0]
        else:
            covs = cv[:, 0]
        sds = np.sqrt(np.clip(covs, 1e-9, None))
        x0 = Xz[:, 0]
        # mixture predictive CDF at each t using smoothed state probs
        cdf = np.zeros(len(x0))
        for k in range(model.n_components):
            cdf += post[:, k] * norm.cdf(x0, loc=means[k], scale=sds[k])
        cdf = np.clip(cdf, 1e-6, 1 - 1e-6)
        resid = norm.ppf(cdf)  # uniform-residual -> normal pseudo-residuals
        W, p = shapiro(resid)
        return {
            "feature": "standing_prob",
            "shapiro_W": float(W),
            "shapiro_p": float(p),
            "resid_mean": float(resid.mean()),
            "resid_std": float(resid.std()),
            "note": "Smoothed-posterior mixture PIT pseudo-residuals on feature 0; "
            "Shapiro-Wilk tests N(0,1) adequacy (higher p = better fit).",
        }
    except Exception as e:  # cheap diagnostic; never fatal
        return {"error": f"pseudo-residual diagnostic skipped: {e}"}


def gt_macro_segments():
    gt = json.load(open(GT_PATH))
    return [
        {
            "start_sec": float(s["start_sec"]),
            "end_sec": float(s["end_sec"]),
            "position": s.get("position"),
        }
        for s in gt["macro_segments"]
    ]


def score_with_movebench(pred_path):
    """Invoke the bundled scorer as a subprocess (stdlib-only scorer)."""
    out_json = os.path.join(RESULTS_DIR, "_hmm_score_full.json")
    cmd = [
        sys.executable, "-m", "movebench.evaluate",
        "--gt", GT_PATH, "--pred", pred_path, "--output", out_json,
    ]
    proc = subprocess.run(
        cmd, cwd=MOVEBENCH_DIR, capture_output=True, text=True, env=dict(os.environ)
    )
    if proc.returncode != 0:
        raise RuntimeError(f"scorer failed: {proc.stderr}\n{proc.stdout}")
    return json.load(open(out_json))


def build_pred(states, times, interval, video_dur, gt_segs, best_K, system, prov, out_path):
    """state path -> contiguous segs -> oracle position map -> merged MoveBench
    prediction; write to out_path; return (pred, mapping, n_runs, n_merged)."""
    segs = decode_segments(states, times, interval, video_dur)
    n_runs = len(segs)
    mapping = map_states_to_positions(segs, gt_segs)
    raw = [
        {"start_sec": s["start_sec"], "end_sec": s["end_sec"],
         "position": mapping.get(s["state"]), "labels": []}
        for s in segs
    ]
    merged = []
    for s in raw:
        if merged and merged[-1]["position"] == s["position"]:
            merged[-1]["end_sec"] = s["end_sec"]
        else:
            merged.append(dict(s))
    pred = {
        "sample_id": "mb_v0_0001",
        "ontology_version": "0.1.0",
        "system": system,
        "provenance": prov,
        "segments": merged,
        "transitions": [],
        "scoring": {"events": []},
    }
    json.dump(pred, open(out_path, "w"), indent=2)
    return pred, mapping, n_runs, len(merged), segs


def main():
    os.makedirs(os.path.dirname(PRED_PATH), exist_ok=True)
    Xz, times, interval, video_dur, feat_names, n_imputed = load_features()
    print(f"[load] {Xz.shape[0]} checkpoints x {Xz.shape[1]} features; "
          f"interval={interval}s dur={video_dur}s imputed={n_imputed}")

    table, best_K, model = fit_select(Xz)
    aic_best = next(r for r in table if r["K"] == best_K)
    print(f"[fit] AIC table: " + ", ".join(f"K{r['K']}:{r['AIC']:.1f}" if r['AIC'] else f"K{r['K']}:NA" for r in table))
    print(f"[select] best K={best_K} (AIC={aic_best['AIC']:.2f}, logL={aic_best['logL']:.2f})")

    raw_states = model.predict(Xz)  # Viterbi
    sm_states = smooth_runs(raw_states, min_run=2)  # 10s min-duration smoothing
    gt_segs = gt_macro_segments()

    base_prov = (
        "CV-only Gaussian-HMM macro-position decoder over Exp4 stage0 CV anchor "
        "features (mb_v0_0001, 194 checkpoints @5s); "
        f"K={best_K} states selected by AIC; Viterbi-decoded; state->position by "
        "oracle majority-overlap mapping; no VLM, no network, in-sample. "
        "T3 labels intentionally empty (CV decoder emits no technique labels)."
    )

    # RAW Viterbi prediction
    raw_pred_path = os.path.join(os.path.dirname(PRED_PATH), "pred_hmm_cv_baseline_raw.json")
    raw_pred, raw_map, raw_runs, raw_merged, raw_segs = build_pred(
        raw_states, times, interval, video_dur, gt_segs, best_K,
        "hmm_cv_only_baseline_raw", base_prov + " [raw Viterbi]", raw_pred_path)
    raw_score = score_with_movebench(raw_pred_path)
    raw_t1 = raw_score["T1_position_segments"]
    print(f"[raw ] runs={raw_runs} merged={raw_merged} "
          f"macro-F1={raw_t1['macro_f1']:.4f} dur-wtd={raw_t1['duration_weighted_macro_f1']:.4f} "
          f"IoU={raw_t1['mean_matched_iou']:.4f} matched={raw_t1['matched_segments']}/{len(gt_segs)}")

    # SMOOTHED prediction (headline)
    sm_pred, mapping, sm_runs, sm_merged, segs = build_pred(
        sm_states, times, interval, video_dur, gt_segs, best_K,
        "hmm_cv_only_baseline", base_prov + " [min-run=2 (10s) smoothed]", PRED_PATH)
    score = score_with_movebench(PRED_PATH)
    t1 = score["T1_position_segments"]
    print(f"[smth] runs={sm_runs} merged={sm_merged} "
          f"macro-F1={t1['macro_f1']:.4f} dur-wtd={t1['duration_weighted_macro_f1']:.4f} "
          f"IoU={t1['mean_matched_iou']:.4f} matched={t1['matched_segments']}/{len(gt_segs)}")
    print(f"[map] state->position (smoothed): {mapping}")

    resid = pseudo_residuals(model, Xz)
    n_state_runs = sm_runs
    merged = sm_pred["segments"]

    # transition matrix + state durations for the record
    trans = model.transmat_.tolist()
    state_secs = defaultdict(float)
    for s in segs:
        state_secs[s["state"]] += s["end_sec"] - s["start_sec"]

    record = {
        "task": "X1 HMM CV-only baseline",
        # Opaque, privacy-safe provenance: the CV cache is the (withheld) Stage0
        # anchor cache for the v0 seed match. We never emit the raw source path /
        # video id from the private cache.
        "cv_source": "stage0_cv_cache (mb_v0_0001; withheld)",
        "video_duration_sec": video_dur,
        "checkpoint_interval_sec": interval,
        "n_checkpoints": int(Xz.shape[0]),
        "features": feat_names,
        "n_imputed_per_feature": n_imputed,
        "hmm": {
            "emission": f"{COV_TYPE}-covariance Gaussian (min_covar=1e-2)",
            "library": f"hmmlearn {__import__('hmmlearn').__version__}",
            "k_range": list(K_RANGE),
            "n_restarts": N_RESTARTS,
            "seed": SEED,
            "selection_criterion": "AIC",
            "selected_K": best_K,
            "selected_AIC": aic_best["AIC"],
            "selected_BIC": aic_best["BIC"],
            "selected_logL": aic_best["logL"],
            "model_selection_table": table,
            "transition_matrix": trans,
            "state_total_seconds": {str(k): v for k, v in sorted(state_secs.items())},
        },
        "state_to_position_mapping": {str(k): v for k, v in sorted(mapping.items())},
        "decode_variants": {
            "raw_viterbi": {
                "n_state_runs": raw_runs,
                "n_merged_macro_segments": raw_merged,
                "T1_macro_F1": raw_t1["macro_f1"],
                "T1_duration_weighted_macro_F1": raw_t1["duration_weighted_macro_f1"],
                "T1_mean_matched_IoU": raw_t1["mean_matched_iou"],
                "T1_matched_segments": f"{raw_t1['matched_segments']}/{len(gt_segs)}",
                "state_to_position_mapping": {str(k): v for k, v in sorted(raw_map.items())},
            },
            "smoothed_min_run_2": {
                "n_state_runs": sm_runs,
                "n_merged_macro_segments": sm_merged,
                "T1_macro_F1": t1["macro_f1"],
                "T1_duration_weighted_macro_F1": t1["duration_weighted_macro_f1"],
                "T1_mean_matched_IoU": t1["mean_matched_iou"],
                "T1_matched_segments": f"{t1['matched_segments']}/{len(gt_segs)}",
            },
        },
        "headline_variant": "smoothed_min_run_2",
        "n_decoded_state_runs": n_state_runs,
        "n_merged_macro_segments": len(merged),
        "n_gt_macro_segments": len(gt_segs),
        "pseudo_residual_fit": resid,
        "movebench_score": score,
        "headline": {
            "T1_macro_F1": t1["macro_f1"],
            "T1_duration_weighted_macro_F1": t1.get("duration_weighted_macro_f1"),
            "T1_mean_matched_IoU": t1.get("mean_matched_iou"),
            "T1_matched_segments": f"{t1.get('matched_segments')}/{len(gt_segs)}",
            "T5_coverage": score["T5_coverage"].get("coverage_fraction"),
            "compare_best_vlm_grounded_macroF1": 0.1888,
            "compare_naive_vlm_macroF1": 0.0165,
        },
        "caveats": [
            "Single annotated sample (n=1): developer measurement, not a leaderboard.",
            "In-sample: CV cache and GT are the same match.",
            "State->position mapping is ORACLE (majority temporal overlap) -> this is "
            "the CV-only macro-F1 CEILING (best case), not an end-to-end unsupervised number.",
            "T3/T2/scoring are empty for this CV-only decoder by design.",
            "Headline uses min-run=2 (10s) Viterbi smoothing (standard segmentation "
            "post-processing tuned to the 5s feature scale, NOT to GT). Raw-Viterbi "
            "numbers are reported alongside for full transparency.",
            "Full-covariance HMM overfits at this N (logL spikes, fragmented decode); "
            "diagonal covariance with a 1e-2 variance floor is used.",
        ],
    }
    json.dump(record, open(JSON_OUT, "w"), indent=2)
    print(f"[write] {JSON_OUT}")
    return record


if __name__ == "__main__":
    main()
