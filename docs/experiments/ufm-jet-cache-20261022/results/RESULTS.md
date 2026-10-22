# Reduced UFM adaptive-cache experiment

Completed 108/108; statuses: {'ok': 108}.

All ratios below are candidate/reference; smaller is better. Runtime is total partitioning time, excluding input/output; RSS is process-wide peak resident memory.

| Comparison | Matched runs | Runtime ratio | Cut ratio | RSS ratio | UFM time ratio |
|---|---:|---:|---:|---:|---:|
| adaptive_vs_corrected_compact | 36 | 1.0053 | 1.0018 | 1.0718 | 1.0078 |
| corrected_compact_vs_stock | 36 | 1.0724 | 0.9969 | 1.0059 | 1.2309 |
| adaptive_vs_stock | 36 | 1.0781 | 0.9987 | 1.0781 | 1.2405 |

The study uses six selected UFM corpus graphs, k=16/1024 and three seeds, with 32 physical cores on one exclusive allocation. All three arms use the repository's eco preset, unconstrained LP + UFM. Graph selection is fixed before measurements. This screening subset does not establish full-corpus superiority.

`plan.json` records every command, source/executable/input hashes and arm order. `runs.jsonl` retains failed, timed-out and infeasible runs; `summary.json` lists missing/invalid comparison pairs. Every successful run has an independently checked exact block-size output. Stock uses the frozen Jet PR head before the shared delta-cache corrections. Corrected compact hashing and Jet share one candidate executable with the same delta-cache corrections and scratch reuse; they differ only in FM cache selection. This separates the common correction/adapter effect from the adaptive representation effect. Rebalancing and preset parameters stay identical across arms. The existing UFM compact-hashing strategy uses the large-k cache path. Parallel scheduling and cache iteration order can change the search trajectory, so phase-time changes are not isolated per-operation costs. Per-level UFM roots are summed without double-counting descendants; direct phases and gain-cache initialization subphases are included in summary.json.
