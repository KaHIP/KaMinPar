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

## Next optimization idea

- Submit a max-core-only mkexp2 run for the `num_seed_nodes = 400` candidate and accept it only if
  it reduces `1x1x96` runtime without an unacceptable cut regression.
- If the server run rejects it, keep the unweighted incident-weight/bucket cleanup only if it is
  independently measurable; otherwise revert the whole candidate and inspect rebalancing hot spots.
