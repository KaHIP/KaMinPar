# Reduced UFM Jet experiment

Completed 180/180; statuses: {'ok': 180}.

All ratios below are candidate/reference; smaller is better. Runtime is total partitioning time, excluding input/output; RSS is process-wide peak resident memory.

| Comparison | Matched runs | Runtime ratio | Cut ratio | RSS ratio | Jet time ratio |
|---|---:|---:|---:|---:|---:|
| cache_with_overload | 36 | 0.7511 | 1.0044 | 0.9825 | 0.6924 |
| cache_with_jet | 36 | 0.6473 | 1.0282 | 0.9085 | 0.5345 |
| balancer_with_sparse | 36 | 0.4467 | 0.8378 | 0.9764 | 0.3968 |
| balancer_with_jet | 36 | 0.3850 | 0.8577 | 0.9030 | 0.3063 |
| combined_vs_original | 36 | 0.2892 | 0.8615 | 0.8871 | 0.2121 |
| combined_vs_ufm | 36 | 1.7893 | 0.9981 | 0.9235 | — |

The study uses six selected UFM corpus graphs, k=16/1024 and three seeds, with 32 physical cores on one exclusive allocation. The eco arm is the repository's unconstrained LP + UFM preset. Graph selection is fixed before measurements. This screening subset does not establish full-corpus superiority.

`plan.json` records every command, source/executable/input hashes and arm order. `runs.jsonl` retains failed, timed-out and infeasible runs; `summary.json` lists missing/invalid comparison pairs. Every successful run has an independently checked exact block-size output. All four Jet arms use the same binary. The legacy arm uses explicit sparse cache + overload balancer within the candidate source, not a separately compiled pre-change program. All Jet arms also share the best-partition feasibility safety correction. Per-level Jet timers are summed without double-counting descendants; phase totals are included in summary.json.
