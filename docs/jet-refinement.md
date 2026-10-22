# Jet connectivity cache and rebalancing

The shared-memory Jet refiner now uses adaptive connectivity tables and bucket-based
rebalancing by default. Both components can be selected independently to measure their
runtime and cut-quality effects. Distributed Jet is unchanged.

## Algorithm references

The design follows Sections 4.2 and 4.3 of
[Gilbert et al., *Jet: Multilevel Graph Partitioning on Graphics Processing Units*](https://arxiv.org/html/2304.13194v2).
The [authors' implementation](https://github.com/sandialabs/Jet-Partitioner/blob/main/src/jet_refiner.hpp)
resolves details that the paper leaves unspecified, particularly the destination-size
threshold and integer mean connectivity.

Jet's connectivity tables exploit the number of blocks actually adjacent to a vertex,
which can be much smaller than `min(degree(u), k)`. The rebalancer first attempts two weak
passes, selecting the best adjacent eligible destination, with arbitrary eligible targets
for disconnected vertices. If necessary, strong passes assign contiguous groups of evicted
vertices to destination-capacity intervals. Weak losses use best-target connectivity;
strong losses use mean connectivity over adjacent eligible targets. Evictions use the
paper's logarithmic loss buckets, with negative losses first, then zero, then positive
losses grouped by powers of two. Both variants restrict heavy-vertex evictions.

## Adaptive cache implementation

`kaminpar-shm/refinement/gains/jet_gain_cache.h` implements `JetGainCache<Graph>`.
For a row with `c` initially adjacent blocks, capacity is
`min(min(degree(u), k), c + 1 + floor(c / 4))`; isolated vertices have no entries.
Rows initially share one contiguous allocation. A full row grows geometrically, bounded
by `min(degree(u), k)`, and rehashes its existing entries into a separate allocation.
Queries and scans therefore depend on the current row capacity rather than on `k`.

The cache exposes `conn`, `gain`, `is_border_node`, `gains`, and `move`.
`gains(u, from, callback)` visits adjacent blocks other than `from`, passing a callable
that returns connectivity, not gain. `for_each_connection` also includes the source.
Callbacks must not reenter the same cache row. Jet's adaptive-cache move selection scans
these adjacent entries. The legacy sparse layout also enumerates adjacent targets,
although its dense rows still require scanning `k` entries. For unsorted graphs, or
when a row's weighted degree cannot fit the legacy packed value, the sparse option
uses on-the-fly adjacent-block aggregation to avoid `n*k` allocation or overflow.

Each row uses a lock for updates. The general `JetGainCache` also locks queries; Jet uses
`SynchronousJetGainCache`, whose queries require a quiescent cache. Jet's blocking parallel
phases separate queries from moves, the legacy balancer only sends move callbacks to this
cache, and the new balancer separates its query and update phases. Thus this specialization
avoids lock contention during read-only neighbor scans. A move subtracts its source
connection, removes zero entries with backward-shift deletion, and adds its destination connection.
This CPU implementation applies both deltas under one lock rather than reproducing the
GPU's separate subtraction, insertion, and reconstruction kernels. Growing from the
stored connections avoids reconstructing from a partition that concurrent moves are
still changing. Initialization, destruction, and aggregate diagnostics require quiescence.

`active_entries()` reports current row capacity. `allocated_entries()` additionally counts
initial slices abandoned after growth; neither statistic includes row metadata or allocator
overhead. Rows do not shrink between moves. Reinitialization after rollback rebuilds them.

## Rebalancer implementation and safeguards

`kaminpar-shm/refinement/jet/jet_balancer.h` implements `JetBalancer<Graph, GainCache>`.
Histograms are allocated only for overweight blocks, with up to eight minibuckets per
loss bucket. Exclusive weight prefixes select evictions, including the vertex crossing
the source's excess-weight threshold. The gain cache is updated after each applied move.

For an upper bound `L` and perfectly balanced weight `P`, weak destination thresholds are
`max(floor(0.99 * L), L - 100)`, while strong thresholds are
`max(P + 1, floor(0.99 * L))`. Thresholds are clamped to the actual upper bound for small
and nonuniform blocks. Only blocks strictly below their threshold are eligible.
The implementation retains the paper's base-two loss buckets and weight restriction;
the current upstream implementation instead uses normalized gains and base-1.5 buckets.

Strong moves that straddle destination-capacity intervals are deferred, ensuring they
cannot overfill a destination. Source lower bounds also constrain evictions. If strong
passes stop decreasing overload, a sequential pass tries adjacent feasible destinations
or a capacity-indexed nonadjacent target without the deadzone or heavy-vertex restriction.
This fallback respects source lower and destination upper bounds. It is a packing
heuristic, not a complete solver for indivisible weighted vertices.

The enclosing Jet refiner minimizes total bound violation before cut, using an
unsigned sum so combined over- and underload cannot overflow the signed weight range.
At equal violation it accepts a smaller cut. Thus an unsuccessful weighted repair does
not replace a known feasible partition. It retains upstream Jet's zero-gain candidate
and projected-move acceptance, reuses candidate gains in filtering, and updates cut
from the original and final blocks of moved vertices. Both balancers feed this move
tracking. Rollback clears LP locks and rebuilds the cache. Configured source minima
are protected by serial LP execution; the usual upper-bound-only path remains parallel.
The balancer exposes weak-round, strong-round, and fallback-move counts through
`statistics()`; phase timers distinguish weak, strong, and fallback work.

## Ablation switches

Use `-P jet` or `-P 4xjet` with the following combinations:

| Configuration | Gain-cache option | Rebalancer option |
| --- | --- | --- |
| Legacy components | `--r-jet-gain-cache sparse` | `--r-jet-balancer overload` |
| Adaptive cache only | `--r-jet-gain-cache jet` | `--r-jet-balancer overload` |
| Jet rebalancer only | `--r-jet-gain-cache sparse` | `--r-jet-balancer jet` |
| Both, the new default | `--r-jet-gain-cache jet` | `--r-jet-balancer jet` |

The corresponding context fields are `refinement.jet.gain_cache_strategy` and
`refinement.jet.balancing_algorithm`. Both default to `JET`. The legacy-component
combination includes the common upstream Jet correctness fixes and safe sparse fallback,
so a strict before/after comparison must additionally use an unchanged baseline executable.

Dedicated tests cover connectivity reconstruction, collision chains, cache growth,
concurrent moves, bucket boundaries, weak and strong passes, nonuniform bounds, source
lower bounds, weighted stalls, and Jet's component combinations. Run the registered
`test_shm_jet_gain_cache`, `test_shm_jet_balancer`, and `test_shm_jet_refiner` targets.

## Reduced UFM evaluation

The [completed six-graph cluster experiment](experiments/jet-adaptive-20261022/README.md)
compares all four component combinations and Eco/UFM at k=16 and k=1024 at frozen
runtime revision `4cf985f2`, before upstream Jet correctness fix #160. At that revision,
the rebalancer supplies the more consistent improvement; the adaptive cache helps mainly
at large k and regresses the small-k aggregate. Those measurements have not been rerun
on the rebased implementation. Raw measurements, balance audits, provenance, and
reproducible figures are retained with the report.
