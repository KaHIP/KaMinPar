# Shared-memory library runtime ownership

`Context` contains configuration. `KaMinPar` takes its own copy and snapshots it for each call;
algorithms access that snapshot through a const reference. Settings remain editable between calls.
Partitioning never writes graph sizes, resolved block bounds, compression statistics, or effective
subproblem thread counts back into configuration.

`PartitionContext` describes a concrete partitioning problem: graph sizes and weights, its number of
blocks, and resolved minimum/maximum block weights. Each recursive subproblem owns its own value.
The original target constraints and current coarse/subgraph constraints remain distinct. Creating a
subproblem does not replace another branch's constraints.

The internal `ExecutionContext` combines a const configuration reference, a `PartitionContext` value,
an effective `ParallelContext`, and a reference to `RuntimeContext`. Copying this view copies problem
state and retains references to the same immutable settings and runtime services. Its references must
outlive every component using the view. A stage with different algorithm settings constructs a local
configuration value and a corresponding view, as the V-cycle partitioner does.

`RuntimeContext` owns one instance's timer, random-stream pool, output policy, and TBB task arena.
Separate library instances never share these services, including when constructed from the same
configuration. Pooled sequential bipartitioners own separate generators at stable addresses because
workers move while their components retain generator references.

## Concurrent calls

Use one `KaMinPar` object per concurrent operation. Operations on the same object, including seed or
configuration changes, must be serialized. Borrowed mutable graph arrays and output buffers must
belong to that operation. Read-only input arrays passed to `copy_graph()` may be shared.

Each instance's arena limits that instance's concurrency. The host application may additionally
limit the overall TBB worker budget. Constructing a partitioner does not change the process-wide
`global_control` value. Quiet mode affects only that instance's algorithm messages. Messages from
multiple verbose instances can interleave at message boundaries.

A call reapplies its configured seed to its own streams. Single-threaded runs can be compared with
identical isolated runs. Parallel stream assignment and algorithm scheduling can still change the
result; per-instance isolation does not promise scheduling-independent parallel reproducibility.
Stream allocation also changes, so a seed need not reproduce a partition from an earlier version.

Timer trees remain available after a successful call through `print_timers(out)`, including for quiet
instances. The next call resets completed timing data. Existing parallel branches still suppress
nested timing within that instance; this change does not introduce parallel timer-stack aggregation.
Compression measurements are collected in a separate `GraphCompressionStatistics` object.

The optional global heap profiler remains a process-wide debugging facility. Concurrent per-instance
profiling is supported for timers; concurrent runs should leave heap profiling disabled. The
distributed partitioner's global services are outside this shared-memory API change; its invocation
of the shared-memory algorithms uses a separate execution context and runtime.

## API migration

| Previous API | Replacement |
| --- | --- |
| `KaMinPar::reseed(seed)` | `partitioner.reseed(seed)` or `config.seed = seed` before construction |
| `KaMinPar::get_seed()` | `partitioner.get_seed()` |
| `partitioner.context().partition` | `partitioner.partition_context()` after a successful call |
| Internal algorithm constructor taking `Context` | An `ExecutionContext` with separate problem state and runtime |
| Python `kaminpar.seed()` / `kaminpar.reseed(seed)` | `partitioner.seed()` / `partitioner.reseed(seed)` |

C callers use `kaminpar_reseed(handle, seed)` and `kaminpar_get_seed(handle)`. Preset creation and
partitioning overloads remain available. These changes intentionally remove the process-wide seed
API from the shared-memory library.

## Validation

Initial validation on 2026-10-23 used macOS arm64 with normal assertions:

- Timer-enabled Debug C++ suite: 486 passed and one configuration-dependent test skipped.
- Nine runtime/concurrency tests passed under ThreadSanitizer with an instrumented oneTBB 2022.2.0
  runtime; no suppressions were used.
- The same nine tests passed with timer instrumentation disabled.
- Python 3.12 bindings with 64-bit IDs/weights: 13 tests passed, including concurrent calls matching
  independently seeded serial runs. Partitioning releases the Python GIL.
- Standalone C and C++ clients compiled and ran using only the installed-style public headers.
- Shared-memory CLI smoke test, distributed CLI smoke test with two MPI processes, and builds of both
  shared-memory benchmarks succeeded.

The MPI smoke test exited successfully; MPICH emitted yaksa leaked-handle warnings at shutdown.
No MPI datatype ownership code is changed by this refactor.

After rebasing onto main `295a6cc6` on 2026-10-26, 565 of 567 Debug tests passed, one
configuration-dependent test was skipped, and the ParHIP cross-chunk weight-total test aborted in
Homebrew oneTBB's `fold_tree` refcount assertion. The same abort was reproduced on unchanged main
with the same compiler and dependencies. All nine runtime/concurrency tests passed both normally
and under ThreadSanitizer with instrumented oneTBB 2022.2.0; the concurrent preset matrix now also
includes Jet's current sparse-cache and bucket-balancer implementation.
