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
- Status: running (initial poll `2026-05-22 14:22 CEST`, `0/88`, `0%`, install/build
  running)

Intent:

- Validate the rollback-state code optimization against the accepted reb10 candidate.
- Accept only if server max-core runtime improves; it should not materially affect quality because
  the code only changes rollback scratch-state representation and reuses buffer capacity.

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

## Automation

- `2026-05-22`: local LaunchAgent fallback configured because no Codex `automation_update` tool was
  available in this session. It runs `/Users/daniel/.codex/automations/iterate-ueco-scalability/continue.sh`
  every 30 minutes with a lock directory and writes logs to
  `/Users/daniel/.codex/automations/iterate-ueco-scalability/logs`.

## Next optimization idea

- Treat reb10 as the current accepted max-core candidate.
- Code-level next targets: reduce border-node initialization/shuffle overhead without changing
  search order, and inspect rollback/rebalancing data structures for avoidable per-move work. Avoid
  the rejected NodeTracker epoch approach unless it can be made cheaper in the hot path.
