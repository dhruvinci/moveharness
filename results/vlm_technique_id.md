# VLM Technique-ID Accuracy (V1)

- Model: `models/gemini-2.5-flash`
- Domain: MMA grappling label dataset (~12s clips)
- Matching: ontology-normalized semantic matching vs `gt_tags`
- Clips scored: **496** / 498 on disk (2 failed/skipped)
- Wall-clock: 1793.6s | API calls: 496 | approx cost: **$0.6127**

## Headline accuracy

| Metric | Value |
|---|---|
| Position top-1 accuracy | **58.9%** |
| Technique match rate (pred in GT set, semantic) | **48.0%** |
| Overall (position AND technique correct) | **35.9%** |

## Per-tag accuracy (top tags by clip count)

| Tag | Clips | Position acc | Technique match | Either match |
|---|---|---|---|---|
| standing | 44 | 43.2% | 77.3% | 86.4% |
| destabilization | 28 | 57.1% | 53.6% | 75.0% |
| guardpass | 27 | 66.7% | 40.7% | 66.7% |
| wrestling | 27 | 37.0% | 81.5% | 88.9% |
| legentanglement | 26 | 92.3% | 69.2% | 92.3% |
| turtleentry | 24 | 45.8% | 62.5% | 79.2% |
| singleleg | 23 | 47.8% | 73.9% | 82.6% |
| takedown | 21 | 42.9% | 95.2% | 95.2% |
| openguard | 21 | 81.0% | 47.6% | 85.7% |
| submission | 20 | 50.0% | 40.0% | 65.0% |
| underhook | 19 | 73.7% | 52.6% | 84.2% |
| gripfight | 19 | 68.4% | 52.6% | 73.7% |
| sweep | 17 | 82.4% | 41.2% | 82.4% |
| backtake | 16 | 56.2% | 31.2% | 56.2% |
| mount | 13 | 46.2% | 30.8% | 46.2% |

## Notes

- Anonymized aggregate stats only; no user IDs, paths, or raw transcripts.
- `gt_tags` are noisy multi-label hashtags (incl. non-technique tokens); scoring uses semantic family mapping so a prediction counts as correct if it names a position/technique actually present in the clip's GT tag set.
- Failures/skips: 2 (see JSON `failures`)
