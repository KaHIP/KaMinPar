# Unconstrained Refiner Scalability Experiments

Notes for mkexp2 runs that compare the unconstrained eco preset (`-P ueco`) against the
regular eco preset (`-P eco`).

## mkexp2 access

- The Codex mkexp2 entry is configured in `~/.codex/config.toml` as an MCP bridge to the
  mkexp2 web backend at `http://127.0.0.1:8766`.
- If Codex does not expose native `mcp__mkexp2__...` tools, use the web backend API with
  the `X-MKEXP2-Token` header from the local Codex config. Do not print the token in logs.
- The backend experiment repository is `/nfs/work/seemaier/experiments`.
- Stale submit locks can remain after canceled submissions. After confirming the jobs are
  canceled, clear only the stale lock before resubmitting.

## Cluster setup

- Use the server graph set `/nfs/work/graph_benchmark_sets/ufm_paper/small` for quick
  scalability experiments.
- Never use `Property slurm.partition all`.
- Use exactly one concrete compute partition from this preferred set:
  `diffie`, `hellman`, `naur`, `backus`, or `liskov`.
- Prefer an idle node. Use only physical cores:
  `diffie` and `hellman` have 96 physical cores, `naur` and `backus` have 64, and
  `liskov` has 128.
- For timing/scalability runs, keep `Property slurm.array.max_parallel 1` unless the
  experiment is explicitly designed to measure throughput rather than per-run time.

## Iteration policy

For fast iteration, only benchmark the selected node's maximum physical core count. Accept a
candidate if OptUeco is faster than BaseUeco at that max-core point while keeping quality sane. Run
the full `4/16/max` scalability comparison only at the end, after a candidate has been accepted.

Iteration experiments compare:

- `BaseUeco`: upstream `KaMinPar` with `-P ueco`
- `OptUeco`: `origin/codex/improve-ueco-scalability` with `-P ueco`

Final scalability experiments should add `BaseEco` and include the `4/16/max` thread sweep. The
baseline point should be 4 physical cores rather than 1 core.

## Local optimization context

The scalability work so far reduces shared synchronization and full-graph passes in the
unconstrained refiners:

- Unconstrained LP counts active/moved nodes with thread-local reductions instead of a
  contended global counter.
- Unconstrained LP computes round improvement from moved nodes and activates moved-node
  neighborhoods in parallel.
- Unconstrained FM avoids full partition copies for rollback and starts rollback from
  saved block weights.
- Unconstrained FM batches move-log appends per localized search batch, uses sparse
  rebalancing-move indexing, and replays rollback through the gain cache instead of
  rescanning adjacency for every move.

## Runs

### 2026.05.21-codex-ueco-scalability-small-hellman

- Partition: `hellman` (96 physical cores)
- Graph set: `/nfs/work/graph_benchmark_sets/ufm_paper/small`
- Created: `2026-05-21T14:02:34`
- Completed: `2026-05-21 17:23 CEST` (per `/api/.../progress`)
- Status: complete (`396/396`, `100%`)
  - BaseEco: `132/132`
  - BaseUeco: `132/132`
  - OptUeco: `132/132`

Geometric mean time (lower is better):

| Threads | BaseEco | BaseUeco | OptUeco | BaseUeco / OptUeco | OptUeco / BaseEco |
| --- | --- | --- | --- | --- | --- |
| `1x1x4` | `10.975s` | `10.181s` | `9.812s` | `1.038x` | `0.894x` |
| `1x1x16` | `3.525s` | `3.148s` | `2.999s` | `1.050x` | `0.851x` |
| `1x1x96` | `1.270s` | `1.644s` | `1.590s` | `1.033x` | `1.253x` |

Geometric mean cut (lower is better):

| Threads | BaseEco | BaseUeco | OptUeco | OptUeco / BaseUeco |
| --- | --- | --- | --- | --- |
| `1x1x4` | `1311331` | `1406716` | `1413194` | `1.0046x` |
| `1x1x16` | `1442155` | `1388866` | `1386910` | `0.9986x` |
| `1x1x96` | `1439627` | `1384600` | `1389794` | `1.0038x` |

Interpretation:

- OptUeco improves runtime vs BaseUeco by ~`3–5%` across `4/16/96` threads.
- Ueco scalability is still worse than Eco at 96 threads (OptUeco is ~`25%` slower than BaseEco at `1x1x96`).

### 2026.05.21-2026.05.21-codex-ueco-scalability-small-hellman-bwbatch

- Partition: `hellman` (idle at submit time, 96 physical cores)
- Graph set: `/nfs/work/graph_benchmark_sets/ufm_paper/small`
- Created: `2026-05-21T17:43:32` (submit action)
- OptUeco commit: `26003b679bd3d3f7a37f605162b7847ff4f123a0`
- Status: canceled after user changed iteration policy to max-core-only (`0/396`)
- Canceled jobs: `70719`, `70723`; dependent arrays `70720`..`70722` remained queued as
  `DependencyNeverSatisfied` and should not consume compute.
- Submit lock cleared via mkexp2 backend.

### 2026.05.21-codex-ueco-max96-bwbatch-hellman

- Partition: `hellman` (idle at submit time, 96 physical cores)
- Graph set: `/nfs/work/graph_benchmark_sets/ufm_paper/small`
- Created: `2026-05-21T17:53:41` (submit action)
- Algorithms: `BaseUeco`, `OptUeco`
- Threads: `1x1x96` only
- OptUeco commit: `14e0151b52c901464ad4d8671aec32dbb73b6d3f`
- Slurm jobs: `70724`, `70725`, `70726`
- Status: complete (last polled `2026-05-21 18:46 CEST`, `88/88`, `100%`)

Geometric mean time (lower is better):

| Threads | BaseUeco | OptUeco | BaseUeco / OptUeco |
| --- | --- | --- | --- |
| `1x1x96` | `1.644s` | `1.555s` | `1.058x` |

Geometric mean cut (lower is better):

| Threads | BaseUeco | OptUeco | OptUeco / BaseUeco |
| --- | --- | --- | --- |
| `1x1x96` | `1379891` | `1388780` | `1.0064x` |

### 2026.05.21-codex-ueco-rollback-4v96-hellman

- Partition: `hellman` (96 physical cores)
- Graph set: `/nfs/work/graph_benchmark_sets/ufm_paper/small`
- Created: `2026-05-21T19:20:59` (submit action)
- Algorithms: `BaseUeco`, `OptUeco`
- Threads: `1x1x4`, `1x1x96`
- OptUeco commit: `ec56814eba05c5de52a100e314268593d4a447f1` (rollback changes)
- Slurm jobs: `70816`, `70817`, `70818`, `70819`
- Status: complete (last polled `2026-05-21 20:39 CEST`, `176/176`, `100%`)

Geometric mean time (lower is better):

| Threads | BaseUeco | OptUeco | BaseUeco / OptUeco |
| --- | --- | --- | --- |
| `1x1x4` | `10.044s` | `11.658s` | `0.862x` |
| `1x1x96` | `1.622s` | `1.706s` | `0.951x` |

Geometric mean cut (lower is better):

| Threads | BaseUeco | OptUeco | OptUeco / BaseUeco |
| --- | --- | --- | --- |
| `1x1x4` | `1396294` | `1422538` | `1.0188x` |
| `1x1x96` | `1377314` | `1394810` | `1.0127x` |

Interpretation:

- Rejected. Rebuilding the gain cache during rollback regresses runtime at both 4 and 96
  cores and worsens cut quality at both points.
- The rollback change was reverted; keep the earlier accepted batching/rollback-through-cache
  changes as the current best branch state.

### 2026.05.21-codex-ueco-max96-ufm-tuning-hellman

- Partition: `hellman` (idle at submit time, 96 physical cores)
- Graph set: `/nfs/work/graph_benchmark_sets/ufm_paper/small`
- Created/submitted: `2026-05-21 23:41 CEST`
- Algorithms: `BaseUeco`, `OptUeco`
- Threads: `1x1x96` only
- OptUeco commit: `8f7db829d8b037b53c9471cea1008666f0e2fa59`
- Slurm install job: `70995`
- Status: complete (polled `2026-05-22 13:15 CEST`, `88/88`, `100%`)

Geometric mean time (lower is better):

| Threads | BaseUeco | OptUeco | BaseUeco / OptUeco |
| --- | --- | --- | --- |
| `1x1x96` | `1.638s` | `1.468s` | `1.116x` |

Geometric mean cut (lower is better):

| Threads | BaseUeco | OptUeco | OptUeco / BaseUeco |
| --- | --- | --- | --- |
| `1x1x96` | `1404198` | `1387912` | `0.9884x` |

Interpretation:

- Accepted as the new local/server baseline: the first UFM tuning/code cleanup improves
  max-core runtime by `11.6%` and slightly improves geomean cut.
- Arithmetic total time was neutral because `rmat_n25_m28` and `stokes` regressed sharply; future
  iterations should watch per-graph slowdowns, not only geomean speedup.

### 2026.05.22-codex-ueco-max96-seed400-diffie

- Partition: `diffie` (idle at submit time, 96 physical cores)
- Graph set: `/nfs/work/graph_benchmark_sets/ufm_paper/small`
- Created/submitted: `2026-05-22 13:25 CEST`
- Algorithms: `PrevOptUeco`, `OptUeco`
- Threads: `1x1x96` only
- PrevOptUeco commit: `a8f5ed9f26d7093a806ddd24a289db88b02449b6`
- OptUeco commit: `3babd60996b8c90fc8a812ec12f4a356d72f4c85`
- Slurm install job: `71092`
- Status: complete (polled `2026-05-22 13:45 CEST`, `88/88`, `100%`)

Geometric mean time (lower is better):

| Threads | PrevOptUeco | OptUeco | PrevOptUeco / OptUeco |
| --- | --- | --- | --- |
| `1x1x96` | `1.411s` | `1.325s` | `1.065x` |

Geometric mean cut (lower is better):

| Threads | PrevOptUeco | OptUeco | OptUeco / PrevOptUeco |
| --- | --- | --- | --- |
| `1x1x96` | `1396002` | `1414252` | `1.0131x` |

Interpretation:

- Accepted as a runtime-quality tradeoff: max-core geomean runtime improved by `6.5%`, arithmetic
  total runtime improved by `14.1%`, and geomean cut regressed by `1.3%`.
- Watch quality outliers in follow-up runs: `HV15R` cut regressed by `1.121x`; `kmer_V2a` runtime
  regressed sharply despite the overall speedup.

### 2026.05.22-codex-ueco-max96-reb10-hellman

- Partition: `hellman` (idle at submit time, 96 physical cores)
- Graph set: `/nfs/work/graph_benchmark_sets/ufm_paper/small`
- Created/submitted: `2026-05-22 13:46 CEST`
- Algorithms: `Seed400Ueco`, `OptUeco`
- Threads: `1x1x96` only
- Seed400Ueco commit: `3babd60996b8c90fc8a812ec12f4a356d72f4c85`
- OptUeco commit: `84dfbb40b5c403d3ea0985c227ebbe0b500fb8a7`
- Slurm install job: `71182`
- Status: complete (polled `2026-05-22 14:14 CEST`, `88/88`, `100%`)

Intent:

- Validate the follow-up `unconstrained_rebalancing_node_inclusion_threshold = 1.0` change against
  the accepted `num_seed_nodes = 400` candidate.
- Accept if it improves max-core runtime without worsening the seed400 cut regression.

Geometric mean time (lower is better):

| Threads | Seed400Ueco | OptUeco | Seed400Ueco / OptUeco |
| --- | --- | --- | --- |
| `1x1x96` | `1.397s` | `1.350s` | `1.034x` |

Geometric mean cut (lower is better):

| Threads | Seed400Ueco | OptUeco | OptUeco / Seed400Ueco |
| --- | --- | --- | --- |
| `1x1x96` | `1406454` | `1417538` | `1.0079x` |

Interpretation:

- Accepted as the current max-core candidate: geomean runtime improved by `3.4%`, arithmetic total
  runtime improved by `5.5%`, and geomean cut regressed by `0.8%`.
- Runtime improved on `26/44` graphs. Largest speedups were `arabic-2005` (`1.871x`),
  `webbase-2001` (`1.750x`), `Bump_2911` (`1.623x`), and `Hook_1498` (`1.457x`).
- Watch quality/runtime outliers in the final scalability run: `soc-flickr-und` cut regressed by
  `1.089x`; `HV15R` cut regressed by `1.073x`; `channel-500x100x100-b050` slowed to `0.604x`.

### 2026.05.22-codex-ueco-max96-rollback-state-diffie

- Partition: `diffie` (idle at submit time, 96 physical cores)
- Graph set: `/nfs/work/graph_benchmark_sets/ufm_paper/small`
- Created/submitted: `2026-05-22 14:21 CEST`
- Algorithms: `Reb10Ueco`, `OptUeco`
- Threads: `1x1x96` only
- Reb10Ueco commit: `5cc9f814c70ea44e28f1a6f181d309fa4046bab1`
- OptUeco commit: `b72976da9d44a10aba0e5edd57dc2702ea9ff558`
- Status: complete (polled `2026-05-22 15:05 CEST`, `88/88`, `100%`)

Intent:

- Validate the rollback-state code optimization against the accepted reb10 candidate.
- Accept only if server max-core runtime improves; it should not materially affect quality because
  the code only changes rollback scratch-state representation and reuses buffer capacity.

Geometric mean time (lower is better):

| Threads | Reb10Ueco | OptUeco | Reb10Ueco / OptUeco |
| --- | --- | --- | --- |
| `1x1x96` | `1.323s` | `1.381s` | `0.958x` |

Geometric mean cut (lower is better):

| Threads | Reb10Ueco | OptUeco | OptUeco / Reb10Ueco |
| --- | --- | --- | --- |
| `1x1x96` | `1413940` | `1412130` | `0.9987x` |

Interpretation:

- Rejected and reverted. Server max-core runtime regressed by `4.2%`, even though geomean cut
  improved slightly.
- Runtime improved on only `17/44` graphs. The largest slowdowns were `soc-sinaweibo` (`0.586x`),
  `vas_stokes_4M` (`0.637x`), `cage15` (`0.683x`), and
  `channel-500x100x100-b050` (`0.790x`).

### 2026.05.22-codex-ueco-max96-fine3x-diffie

- Partition: `diffie` (96 physical cores)
- Graph set: `/nfs/work/graph_benchmark_sets/ufm_paper/small`
- Created/submitted: `2026-05-22 15:25 CEST`
- Algorithms: `Reb10Ueco`, `OptUeco`
- Threads: `1x1x96` only
- Reb10Ueco commit: `5cc9f814c70ea44e28f1a6f181d309fa4046bab1`
- OptUeco commit: `522dc687dd27bb16f98bbc65bb724aac136e4d57`
- Slurm jobs: `71365`, `71366`, `71367`
- Status: complete (`88/88`)

Intent:

- Validate the code-level runtime-quality trade-off that uses `3 * num_seed_nodes` for finest-level
  UFM localized searches while keeping the accepted `num_seed_nodes = 400` on coarser levels.
- Accept only if max-core server runtime improves over reb10 and the cut remains sane; local
  validation showed a large FM speedup but noisy total runtime.

Result:

Geometric mean running time (lower is better):

| Threads | Reb10Ueco | OptUeco | Reb10Ueco / OptUeco |
| --- | --- | --- | --- |
| `1x1x96` | `1.37034s` | `1.31801s` | `1.0397x` |

Geometric mean cut (lower is better):

| Threads | Reb10Ueco | OptUeco | OptUeco / Reb10Ueco |
| --- | --- | --- | --- |
| `1x1x96` | `1414489` | `1452564` | `1.0269x` |

Interpretation:

- Rejected as the default despite the `3.97%` geomean runtime improvement. The quality trade-off is
  too large for the current accepted path: geomean cut regressed by `2.69%`, and `rhg` regressed
  from `20664` to `44381` cut (`2.1477x`) while also slowing from `0.674s` to `1.253s`.
- Runtime improved on `31/44` graphs and arithmetic total time improved by `1.165x` (`160.929s` to
  `138.152s`), so large fine-level batches remain useful as an optional speed-quality trade-off.
  The worst runtime slowdowns were `rhg` (`0.538x`), `stokes` (`0.546x`), `indochina-2004`
  (`0.702x`), and `channel-500x100x100-b050` (`0.801x`).
- Keep the current reb10 candidate as the accepted baseline, revert the fine-level multiplier, and
  look for code-level speedups that do not weaken the fine-level solution quality this much.

### 2026.05.22-codex-ueco-max96-constrained3x-diffie

- Partition: `diffie` (96 physical cores)
- Graph set: `/nfs/work/graph_benchmark_sets/ufm_paper/small`
- Created/submitted: `2026-05-22 16:07 CEST`
- Algorithms: `Reb10Ueco`, `OptUeco`
- Threads: `1x1x96` only
- Reb10Ueco commit: `5cc9f814c70ea44e28f1a6f181d309fa4046bab1`
- OptUeco commit: `438f10dae00e304e7406460e790587ac9fc4cf18`
- Slurm jobs: `71496` (install), `71497` (array), `71498` (submit-lock cleanup)
- Status: complete (`88/88`)

Intent:

- Validate the code-level compromise after rejecting unconditional fine-level `3x` batches: keep
  `num_seed_nodes = 400` during overload-permitting unconstrained FM iterations, then switch to
  `3 * num_seed_nodes` only for constrained FM iterations.
- Accept if server max-core runtime improves over reb10 without reproducing the fine-level `3x`
  quality outlier (`rhg` cut `2.1477x`).

Result:

Geometric mean running time (lower is better):

| Threads | Reb10Ueco | OptUeco | Reb10Ueco / OptUeco |
| --- | --- | --- | --- |
| `1x1x96` | `1.37942s` | `1.32564s` | `1.0406x` |

Geometric mean cut (lower is better):

| Threads | Reb10Ueco | OptUeco | OptUeco / Reb10Ueco |
| --- | --- | --- | --- |
| `1x1x96` | `1415425` | `1417920` | `1.0018x` |

Interpretation:

- Accepted as the current max-core candidate. It gives a `4.06%` geomean runtime improvement and a
  `3.96%` arithmetic total-time improvement (`147.535s` to `141.910s`) with only `0.18%` worse
  geomean cut.
- Runtime improved on `28/44` graphs. The largest wins were `arabic-2005` (`1.486x`),
  `kmer_V2a` (`1.470x`), `channel-500x100x100-b050` (`1.317x`), `Bump_2911` (`1.250x`), and
  `nlpkkt240` (`1.238x`).
- The largest runtime slowdowns were `Hook_1498` (`0.635x`), `nv2` (`0.854x`),
  `vas_stokes_4M` (`0.891x`), `stokes` (`0.903x`), and `soc-sinaweibo` (`0.905x`).
- Quality is much saner than the rejected fine-level `3x` run. The worst remaining cut outlier is
  `rhg` (`19509` to `22057`, `1.1306x`), followed by `Hook_1498` (`1.0400x`) and `rgg26`
  (`1.0339x`). This is worth accepting for now, but `rhg` remains the guard graph for future
  speed-quality trade-offs.

### 2026.05.22-codex-ueco-max128-mqgen-liskov

- Partition: `liskov` (idle at submit time, 128 physical cores)
- Graph set: `/nfs/work/graph_benchmark_sets/ufm_paper/small`
- Created/submitted: `2026-05-22 17:25 CEST`
- Algorithms: `Constrained3xUeco`, `OptUeco`
- Threads: `1x1x128` only
- Constrained3xUeco commit: `438f10dae00e304e7406460e790587ac9fc4cf18`
- OptUeco commit: `7dff9a08876c64c6ad3e1cba17e5c549f93485a2`
- Slurm jobs: `71667` (install), `71668` (array), `71669` (submit-lock cleanup)
- Status: complete (polled `2026-05-22 18:24 CEST`, `88/88`, `100%`)

Intent:

- Validate the code-level multi-queue overload-balancer implementation against the accepted
  constrained-only `3x` baseline. The local implementation target was met on the targeted
  balancer/rebalancing subphase (`1.217x` balancer speed, `1.202x` rebalancing speed), while
  end-to-end local runtime improved by `1.052x`.
- Accept only if server max-core runtime improves without a new `rhg` quality regression.

Result:

Geometric mean running time (lower is better):

| Threads | Constrained3xUeco | OptUeco | Constrained3xUeco / OptUeco |
| --- | --- | --- | --- |
| `1x1x128` | `2.04435s` | `2.08751s` | `0.9793x` |

Geometric mean cut (lower is better):

| Threads | Constrained3xUeco | OptUeco | OptUeco / Constrained3xUeco |
| --- | --- | --- | --- |
| `1x1x128` | `1423025` | `1421088` | `0.9986x` |

Interpretation:

- Rejected. The patch slightly improved geomean cut (`0.14%`) and the `rhg` guard graph improved
  in both time (`1.058x`) and cut (`0.9725x`), but server max-core runtime regressed:
  `0.9793x` geomean speed and `0.8590x` arithmetic total-time speed (`169.580s` to `197.424s`).
- Runtime improved on only `22/44` graphs. The largest slowdowns were `rmat_n25_m28` (`0.481x`,
  `31.080s` to `64.610s`), `arabic-2005` (`0.752x`), `Hook_1498` (`0.816x`), `Bump_2911`
  (`0.818x`), `channel-500x100x100-b050` (`0.820x`), and `com-lj.ungraph` (`0.831x`).
- The largest speedups were `nlpkkt240` (`1.425x`), `soc-sinaweibo` (`1.314x`), `Long_Coup_dt6`
  (`1.249x`), `Flan_1565` (`1.230x`), and `Queen_4147` (`1.192x`), but they did not compensate
  for the slowdowns.
- The balancer generation-stamp patch was reverted after this result. Keep `438f10da` as the
  accepted max-core baseline.

## Local runs

Local sanity-check runs (this machine) on `~/Graphs/coAuthorsDBLP.metis`, `k=16`, `eps=0.03`:

| Preset | Threads | Time (real) | Cut | Feasible |
| --- | --- | --- | --- | --- |
| `eco` | `4` | `0.240s` | `122499` | `yes` |
| `eco` | `16` | `0.130s` | `121851` | `yes` |
| `eco` | `32` | `0.130s` | `120959` | `yes` |
| `ueco` | `4` | `0.240s` | `124248` | `yes` |
| `ueco` | `16` | `0.140s` | `123320` | `yes` |
| `ueco` | `32` | `0.170s` | `124840` | `yes` |

Notes:

- `32` threads is oversubscribed locally (machine reports 18 CPUs), so only use it as a rough sanity check.

Local sanity-check runs after the FM rollback change (this worktree, `2026-05-21 18:46 CEST`):

| Preset | Threads | Time (real) | Cut | Feasible |
| --- | --- | --- | --- | --- |
| `eco` | `4` | `0.19s` | `123883` | `yes` |
| `eco` | `8` | `0.13s` | `123358` | `yes` |
| `eco` | `16` | `0.11s` | `122153` | `yes` |
| `ueco` | `4` | `0.23s` | `125096` | `yes` |
| `ueco` | `8` | `0.17s` | `124392` | `yes` |
| `ueco` | `16` | `0.15s` | `122870` | `yes` |

Local UFM parameter/code sweep (`2026-05-21 23:30 CEST`, `k=16`, `eps=0.03`, `t=16`):

| Graph | Baseline median | Tuned median | Speedup | Tuned / baseline cut |
| --- | --- | --- | --- | --- |
| `rmat_n16_m23` | `0.790s` | `0.640s` | `1.234x` | `1.0022x` |
| `kkt_power` | `0.430s` | `0.380s` | `1.132x` | `1.0624x` |
| `web-Google` | `0.250s` | `0.220s` | `1.136x` | `1.0180x` |

Geometric mean: `1.166x` faster with `1.027x` worse cut. The tuned version uses larger UFM
batches (`num_seed_nodes = 50`), earlier switch from unconstrained FM
(`unconstrained_min_improvement = 0.005`), an overload cap
(`unconstrained_upper_bound = 1.05`), and lower minimal search parallelism (`4`).

Rejected stronger local tradeoffs:

- `num_seed_nodes = 75` plus `unconstrained_min_improvement = 0.01`: repeat runs were slower on
  `kkt_power` and `web-Google` despite speeding up `rmat_n16_m23`.
- `num_seed_nodes = 100`: faster in one sweep, but quality degraded substantially on `kkt_power`.
- `unconstrained_upper_bound = 1.02`: not consistently faster than the selected `1.05` cap.

Additional max-local-core checks (`2026-05-21 23:45 CEST`, `t=18`, two repeats):

- Behavior-preserving generation stamps for UFM rebalancing-node flags avoided one full reset pass,
  but total runtime was neutral and noisy (`1.003x` geomean speedup, with `rmat_n16_m23` slower).
  Rejected for now to avoid extra state complexity without a clear local gain.
- Stronger runtime-quality tradeoffs did not produce a defensible local win at `t=18`. Relative to
  the current tuned preset, `num_seed_nodes = 100` was `0.990x` geomean speed with `1.038x` cut,
  `num_iterations = 6` was only `1.016x` speed with `1.018x` cut, and the combined fast variants
  were slower on geomean.

Local max-core UFM batch-size sweep (`2026-05-22 13:21 CEST`, `k=16`, `eps=0.03`, `t=18`):

| Graph set | Candidate | Total speedup | UFM speedup | Candidate / baseline cut |
| --- | --- | --- | --- | --- |
| 6 larger local graphs, one repeat | `num_seed_nodes = 400` | `1.329x` | `1.671x` | `0.9852x` |
| 6 larger local graphs, one repeat | `num_seed_nodes = 1200` | `1.286x` | `1.917x` | `1.0200x` |
| 9 local graphs, two repeats | `num_seed_nodes = 400` plus init cleanup | `1.114x` | `1.906x` | `1.0184x` |

The 9-graph validation compared the new candidate against the previous `num_seed_nodes = 50`
default via a CLI override. It used `com-lj.ungraph`, `as-skitter`, `coPapersDBLP`,
`web-BerkStan`, `rgg_n_2_19_s0`, `rmat_n16_m24`, `rmat_n16_m23`, `kkt_power`, and
`web-Google`. The candidate reaches the explicit `30%` UFM-time reduction target locally, with
about `1.8%` worse geomean cut. The largest observed quality regression was `kkt_power`
(`1.152x` cut), so server validation must check whether this remains acceptable on the small set.

Rejected follow-up:

- Skipping border-node shuffling when `num_seed_nodes >= 400` improved measured local UFM time by
  only `1.073x` and total time by `1.003x` versus the `num_seed_nodes = 400` candidate, with mixed
  per-graph runtime (`kkt_power` slowed down). Rejected as too noisy and not committed.

Additional local candidate:

- Raising `unconstrained_rebalancing_node_inclusion_threshold` from `0.7` to `1.0` on top of
  `num_seed_nodes = 400` was locally promising. A two-repeat 9-graph run at `t=18` showed `1.054x`
  total speedup and `1.205x` UFM speedup versus the `num_seed_nodes = 400` candidate, with
  `0.9959x` geomean cut.

Additional rejected local ideas (`2026-05-22 13:48–13:51 CEST`, `t=18`, 9 local graphs):

- Raising `unconstrained_min_improvement` to `0.02` looked good in a one-repeat sweep, but did not
  repeat. The two-repeat check was `0.950x` total speed, `0.942x` UFM speed, and `1.0027x` cut
  relative to the current seed400+reb10 candidate, so this is rejected.
- Increasing `num_seed_nodes` further to `1200` gave only a small and noisy total-speed gain
  (`1.013x` alone, `1.030x` combined with `unconstrained_min_improvement = 0.02`) despite UFM-only
  speedups. Do not submit this before the reb10 server result.
- A behavior-preserving NodeTracker generation-stamp implementation avoided full tracker resets and
  sped up the UFM border-initialization subphase (`1.102x`), but total runtime was slower
  (`0.953x`) and FM runtime was only neutral (`1.016x`). The patch was reverted; revisit only if a
  lower-overhead implementation removes the per-operation generation checks from hot paths.

Local code optimization validation (`2026-05-22 14:18 CEST`, `t=18`, 6 local graphs):

- Reused a single rollback move-state buffer and encoded reverted/applied flags in one byte array
  instead of allocating and clearing two rollback vectors per FM round.
- Compared against `/private/tmp/ueco/bin/KaMinPar_before_node_tracker_epoch` on `com-lj.ungraph`,
  `as-skitter`, `coPapersDBLP`, `web-BerkStan`, `rmat_n16_m24`, and `kkt_power`, two repeats.
- Local result: `1.057x` geomean total speedup, `1.180x` geomean FM speedup, `1.071x` rollback
  speedup, and `1.0054x` geomean cut ratio. This is code-level and behavior-preserving modulo
  normal parallel nondeterminism, so it is worth server validation.

Server follow-up rejected this rollback-state optimization; do not reintroduce it without a clearer
explanation for the server slowdown.

Local rejected code probes after the rollback-state rejection (`2026-05-22 15:10–15:17 CEST`,
`t=18`, 9 local graphs, two repeats against the reb10 baseline):

- Sparse-clearing UFM rebalancing-node flags and non-atomic tracker reset: `1.036x` total and
  `1.035x` FM geomean speed, but `com-lj.ungraph` regressed to `0.855x`; rejected as too risky.
- Non-atomic NodeTracker reset only: `0.973x` total geomean speed despite neutral FM; rejected.
- Lowering the high-degree no-improvement pruning threshold from `100000` to `10000`: `0.993x`
  total geomean speed and `1.0013x` cut; rejected.
- Bulk-copying interleaved rebalancing moves back into the concurrent round-move vector:
  `0.989x` total geomean speed; rejected.

Local runtime-quality trade-off selected for server validation:

- Code change: use `3 * num_seed_nodes` only for finest-level UFM localized searches, keeping the
  accepted `num_seed_nodes = 400` on coarser levels.
- Local result (`/private/tmp/ueco/ufm_fine_seed3x_t18_20260522_151902`, 9 graphs, two repeats):
  `1.012x` total geomean speedup, `1.268x` FM speedup, `1.324x` localized-search speedup, and
  `0.9994x` geomean cut ratio.
- Trade-off: total runtime is noisy and slower on `as-skitter`, `coPapersDBLP`, and `kkt_power`,
  but the FM-specific speedup is large enough to justify one max-core server validation. A milder
  `2x` fine-level multiplier was worse (`0.962x` total, `0.954x` FM, `1.0021x` cut), so the
  submitted candidate uses `3x`.

Local follow-up after the fine-level `3x` rejection (`2026-05-22 16:04 CEST`, `t=18`, ten local
graphs, two repeats against the reb10 baseline):

- Code change: keep `num_seed_nodes = 400` during unconstrained FM iterations, then use
  `3 * num_seed_nodes` only after UFM switches back to constrained FM. This keeps the speed-oriented
  larger batches out of the overload-permitting phase that caused the server quality regression.
- Local result (`/private/tmp/ueco/ufm_constrained_seed3x_t18_20260522_160403`):
  `1.083x` total geomean speedup, `1.106x` FM speedup, `1.217x` fine-level localized-search
  speedup, and `0.9971x` geomean cut ratio.
- Runtime improved on `7/10` graphs. The local `rhg18` guard graph improved from `84710` to
  `80439` cut (`0.9496x`) while speeding up from `0.090s` to `0.080s`, which addresses the most
  severe failure mode seen in the server fine-level `3x` run well enough to justify max-core server
  validation.

Rejected local code probe after accepting constrained-only `3x` batches (`2026-05-22 16:30 CEST`,
`t=18`, ten local graphs, two repeats against the accepted constrained-only `3x` baseline):

- Empty-batch fast path: return immediately from a UFM worker batch if polling the shared border-node
  cursor yields no seed nodes. This is behavior-preserving in intent, but the local result was not a
  useful FM speedup: `1.031x` total geomean speed, `0.979x` FM speed, `0.871x` fine-level
  localized-search speed, and `0.9949x` cut ratio
  (`/private/tmp/ueco/ufm_empty_batch_skip_t18_20260522_163034`).
- Rejected and reverted because the measured total-speed gain is likely phase noise, while the
  targeted FM/search timers got slower.

Local balancer implementation optimization after accepting constrained-only `3x` batches
(`2026-05-22 16:57–17:19 CEST`, `t=18`, ten local graphs, two repeats against accepted commit
`438f10da`):

- Rejected split constrained/unconstrained gain search in UFM: `1.006x` total speed, `1.046x` FM
  speed, `0.952x` fine-level localized-search speed, and `0.9980x` cut ratio
  (`/private/tmp/ueco/ufm_split_gain_t18_20260522_165719`).
- Rejected active-node reset alone in the multi-queue overload balancer: `1.003x` total speed,
  `1.065x` balancer speed, `1.067x` ULP speed, `1.070x` FM speed, and `0.9888x` cut ratio
  (`/private/tmp/ueco/mq_active_reset_t18_20260522_170400`).
- Fast per-worker xorshift RNG alone was useful but not enough: `1.033x` total speed,
  `1.167x` balancer speed, `1.143x` ULP speed, `0.997x` FM speed, and `1.0024x` cut ratio
  (`/private/tmp/ueco/mq_fast_rng_t18_20260522_170602`).
- Rejected reducing the number of priority queues from `2 * threads` to `threads`: `0.974x` total
  speed, `0.882x` balancer speed, `0.909x` ULP speed, and `0.9983x` cut ratio
  (`/private/tmp/ueco/mq_fast_rng_active_reset_p1_t18_20260522_171004`).
- Selected code change: replace `std::mt19937` queue sampling with a lightweight xorshift token,
  keep balancer priority queues/node-state arrays allocated across calls, use a per-refinement
  node generation stamp so stale node state is ignored without a full reset or active-node merge,
  and clear PQs without uncontended locks after worker threads have joined.
- Selected local result (`/private/tmp/ueco/mq_generation_stamp_t18_20260522_171905`):
  `1.052x` total speed, `1.217x` multi-queue overload-balancer speed, `1.191x` ULP speed,
  `1.202x` rebalancing speed, `1.047x` FM speed, and `0.9982x` cut ratio. This reaches the
  requested `20%` implementation speedup on the targeted balancer/rebalancing subphase, but it is
  not a `20%` end-to-end `-P ueco` speedup; server validation is still required.
  The local `rhg18` guard improved in both runtime (`1.125x`) and cut (`0.9888x`).
- Server validation rejected this patch at `1x1x128`; the local subphase gain did not carry over to
  end-to-end server runtime because large instances, especially `rmat_n25_m28`, regressed.

Additional local implementation probes after the MQ generation-stamp rejection (`2026-05-22
18:29–18:45 CEST`, `t=18`, ten local graphs including `rhg18`, two repeats against accepted
commit `438f10da`):

- Rejected sorting FM moves by node instead of using an `unordered_map` in
  `interleave_rebalancing_moves()`: `0.975x` total speed, `0.933x` FM speed, and `1.0173x` cut
  ratio (`/private/tmp/ueco/ufm_interleave_sorted_t18_20260522_182958`).
- Rejected removing duplicate `UnconstrainedFMData::initialize()` clearing: `1.008x` total speed
  but `0.985x` FM speed and `0.904x` init-timer speed
  (`/private/tmp/ueco/ufm_init_clear_t18_20260522_183126`).
- Rejected delaying virtual-weight-delta flushes until the end of a localized-search batch:
  `0.985x` total speed and `0.9993x` cut ratio despite a local `1.052x` FM timer
  (`/private/tmp/ueco/ufm_delayed_delta_flush_t18_20260522_183352`).
- Rejected sparse block-list flushing for virtual-weight deltas: `0.972x` total speed,
  `0.966x` FM speed, and `0.9872x` cut ratio
  (`/private/tmp/ueco/ufm_sparse_delta_flush_t18_20260522_183523`).
- Rejected the rollback tie-check shortcut that avoided scanning the heaviest block unless the cut
  could improve: `1.014x` geomean total speed but `0.990x` arithmetic total speed,
  `0.909x` FM speed, and `0.905x` rollback speed
  (`/private/tmp/ueco/ufm_rollback_tiecheck_t18_20260522_183706`).
- Rejected a conservative acquire-CAS NodeTracker lock variant: `1.001x` total speed,
  `1.120x` FM geomean speed, but `0.958x` arithmetic total speed and `1.0079x` cut ratio
  (`/private/tmp/ueco/ufm_acquire_tracker_lock_t18_20260522_184309`).
- Selected code change for validation: weaken `NodeTracker::lock()` from sequentially consistent
  compare-exchange to relaxed compare-exchange. The tracker state is used as an ownership token, and
  its loads/stores were already relaxed; the change removes the remaining hot-path seq-cst lock from
  seed and touched-node acquisition.
- Selected local result (`/private/tmp/ueco/ufm_relaxed_tracker_lock_t18_20260522_184105`):
  `1.078x` total geomean speed, `1.043x` arithmetic total speed, `1.230x` FM geomean speed,
  `1.265x` FM arithmetic speed, `1.148x` fine-level localized-search speed, `1.225x`
  rebalancing speed, and `0.9854x` cut ratio. This reaches the requested `20%` implementation
  speedup on the targeted FM phase locally, but end-to-end speed is still below `20%` and the result
  is scheduling-sensitive enough to require max-core server validation.
- Verification passed after the code change: `cmake --build build --target KaMinParApp -j 8` and
  `ctest --test-dir build -R '(ShmEndToEndTest|GainCacheTest)' --output-on-failure -j 8` (`70/70`).
- Submitted mkexp2 validation `2026.05.22-codex-ueco-max128-relaxed-lock-liskov` on idle
  `liskov`, graph set `/nfs/work/graph_benchmark_sets/ufm_paper/small`, algorithms
  `Constrained3xUeco` (`438f10da...`) and `OptUeco`
  (`50892619db5d3f817ca21de78ee2d0081a02b467`), threads `1x1x128`, 88 calls. Initial progress:
  `0/88`; Slurm jobs `71799` (install), `71800` (array), and `71801` (submit-lock cleanup).

## Automation

- `2026-05-22`: local LaunchAgent fallback configured because no Codex `automation_update` tool was
  available in this session. It runs `/Users/daniel/.codex/automations/iterate-ueco-scalability/continue.sh`
  every 30 minutes with a lock directory and writes logs to
  `/Users/daniel/.codex/automations/iterate-ueco-scalability/logs`.
- `2026-05-22 15:26 CEST`: Codex heartbeat automation
  `check-ueco-fine3x-validation` created to poll
  `2026.05.22-codex-ueco-max96-fine3x-diffie` every 30 minutes, parse completed results, document
  the interpretation, and continue with the next idea.
- `2026-05-22 16:08 CEST`: same heartbeat automation updated to poll
  `2026.05.22-codex-ueco-max96-constrained3x-diffie` every 30 minutes.
- `2026-05-22 17:26 CEST`: Codex heartbeat automation
  `check-ueco-mq-generation-validation` created to poll
  `2026.05.22-codex-ueco-max128-mqgen-liskov` every 30 minutes, parse completed results, document
  the interpretation, and continue with the next code-level idea.
- `2026-05-22 18:45 CEST`: next heartbeat should poll the NodeTracker relaxed-lock validation
  once submitted, then accept/reject it, update this document and the automation memory, and
  continue local-first implementation work with a new UFM idea instead of stopping at result
  interpretation.
- `2026-05-22 18:51 CEST`: Codex heartbeat automation
  `check-ueco-relaxed-lock-validation` created to poll
  `2026.05.22-codex-ueco-max128-relaxed-lock-liskov` every 30 minutes, parse completed results,
  document the interpretation, accept/reject `50892619...`, and continue with the next local-first
  code-level UFM idea.

## Next optimization idea

- Treat `438f10da` (constrained-only `3x` batches) as the accepted max-core baseline.
- The multi-queue overload-balancer generation-stamp implementation is rejected and reverted.
- Validate the relaxed NodeTracker lock against `438f10da` on a max-core mkexp2 run.
- If rejected, revert the lock change and continue with local-first code-level UFM work on
  search scheduling and rebalancing/interleaving data structures. Avoid the rejected
  tracker-generation/reset/rollback/balancer-generation micro-optimizations unless new profiling
  explains the previous slowdowns.
