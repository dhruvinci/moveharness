#!/usr/bin/env python3
"""Download short ~12s clips centered on labeled timestamps from the local label
dataset, and emit a ground-truth manifest for the technique-ID benchmark.

Robust to throttling/unavailable videos: one retry, skip+log failures, global
time cap (~18 min of download wall-clock), early-stop on repeated YouTube blocks.
Clips are for LOCAL evaluation only (author's own labeled dataset), not redistributed.
"""
import json, os, sys, time, glob, traceback
import yt_dlp

# benchmark/ base dir: resolved relative to this script (data/ -> benchmark/),
# overridable via the MOVEBENCH_BASE env var. No user-specific absolute path.
BASE = os.environ.get('MOVEBENCH_BASE') or os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))
SEL = os.path.join(BASE, 'data', 'technique_clips_selection.json')
OUTDIR = os.path.join(BASE, 'data', 'technique_clips')
LOGPATH = os.path.join(BASE, 'results', 'technique_clips_download.log')
MANIFEST = os.path.join(OUTDIR, 'manifest.json')

CLIP_HALF = 6.0          # seconds each side of timestamp -> ~12s clips
TIME_CAP = int(os.environ.get('DL_TIME_CAP', 18*60))  # env-overridable
BLOCK_STREAK_STOP = 8    # consecutive block-like failures -> early stop

os.makedirs(OUTDIR, exist_ok=True)
os.makedirs(os.path.dirname(LOGPATH), exist_ok=True)

logf = open(LOGPATH, 'a')
def log(msg):
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    logf.write(line + "\n"); logf.flush()

BLOCK_SIGNS = ('sign in to confirm', 'not a bot', 'http error 429',
               'too many requests', 'rate-limit', 'rate limit',
               'blocked', 'forbidden', 'unable to download webpage')

def is_block(errstr):
    e = errstr.lower()
    return any(s in e for s in BLOCK_SIGNS)

def download_clip(yt_id, center, out_path):
    start = max(0.0, center - CLIP_HALF)
    end = center + CLIP_HALF
    base_noext = out_path[:-4]
    def rng(info, ydl):
        return [{'start_time': start, 'end_time': end}]
    opts = {
        'format': 'bv*[height<=480]+ba/b[height<=480]/best',
        'outtmpl': base_noext + '.%(ext)s',
        'download_ranges': rng,
        'force_keyframes_at_cuts': True,
        'quiet': True, 'no_warnings': True,
        'retries': 1, 'fragment_retries': 1,
        'socket_timeout': 30,
        'merge_output_format': 'mp4',
        'noprogress': True,
    }
    with yt_dlp.YoutubeDL(opts) as ydl:
        ydl.download([f'https://www.youtube.com/watch?v={yt_id}'])
    # yt_dlp may emit .mp4 or .mkv depending on merge; normalize
    if not os.path.exists(out_path):
        cands = sorted(glob.glob(base_noext + '.*'),
                       key=lambda p: os.path.getsize(p), reverse=True)
        cands = [c for c in cands if not c.endswith('.part')]
        if cands:
            os.replace(cands[0], out_path)
    if not os.path.exists(out_path) or os.path.getsize(out_path) < 1024:
        raise RuntimeError('no output produced')
    return os.path.getsize(out_path)

def main():
    sel = json.load(open(SEL))
    log(f"=== START: {len(sel)} target clips, time cap {TIME_CAP}s ===")
    t0 = time.time()
    manifest = {}
    n_ok = n_skip = n_fail = 0
    block_streak = 0
    stopped_early = None

    for i, s in enumerate(sel):
        if time.time() - t0 > TIME_CAP:
            stopped_early = 'time_cap'
            log(f"TIME CAP hit after {i} attempts; stopping.")
            break
        tid = s['id']; yt = s['youtube_id']; sec = s['seconds']
        out_path = os.path.join(OUTDIR, f"{tid}.mp4")
        if os.path.exists(out_path) and os.path.getsize(out_path) > 1024:
            n_ok += 1
            manifest[f"{tid}.mp4"] = dict(
                gt_tags=s['gt_tags'], gt_label_text=s['label'],
                seconds=sec, video_youtube_id=yt)
            continue
        ok = False; last_err = ''
        for attempt in (1, 2):
            try:
                size = download_clip(yt, sec, out_path)
                ok = True; block_streak = 0
                log(f"[{i+1}/{len(sel)}] id={tid} yt={yt} OK ({size} bytes)")
                break
            except Exception as e:
                last_err = f"{type(e).__name__}: {str(e)[:200]}"
                if is_block(last_err):
                    block_streak += 1
                if attempt == 1:
                    time.sleep(1.5)
        if ok:
            n_ok += 1
            manifest[f"{tid}.mp4"] = dict(
                gt_tags=s['gt_tags'], gt_label_text=s['label'],
                seconds=sec, video_youtube_id=yt)
        else:
            n_fail += 1
            log(f"[{i+1}/{len(sel)}] id={tid} yt={yt} FAIL {last_err}")
            # cleanup partials
            for p in glob.glob(out_path[:-4] + '.*'):
                try: os.remove(p)
                except OSError: pass
        if block_streak >= BLOCK_STREAK_STOP:
            stopped_early = 'youtube_block'
            log(f"YOUTUBE BLOCK streak={block_streak}; stopping early.")
            break

    # write manifest
    with open(MANIFEST, 'w') as f:
        json.dump(manifest, f, indent=1)

    elapsed = time.time() - t0
    throttled = stopped_early == 'youtube_block' or block_streak > 0
    summary = dict(
        attempted=min(len(sel), n_ok + n_fail + n_skip) if not stopped_early else None,
        n_target=len(sel), n_downloaded=n_ok, n_failed=n_fail,
        elapsed_sec=round(elapsed, 1), stopped_early=stopped_early,
        youtube_throttled=bool(throttled), block_streak=block_streak)
    log("=== SUMMARY === " + json.dumps(summary))
    json.dump(summary, open(os.path.join(BASE, 'results', 'technique_clips_summary.json'), 'w'), indent=1)

if __name__ == '__main__':
    main()
