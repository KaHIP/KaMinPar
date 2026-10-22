# Adaptive Jet gain cache in FM

`--r-fm-gc jet` selects the adaptive connectivity tables described in
[Jet refinement](jet-refinement.md) for shared-memory unconstrained FM and constrained
FM. The existing `compact-hashing` default is unchanged. For an Eco comparison, use
`-P eco --r-fm-gc jet` or `-P eco --r-fm-gc compact-hashing`.

`FMJetGainCache` uses the general Jet cache with query locking enabled. FM workers
read shared connectivity while other workers commit moves, so Jet refinement's
phase-separated, unlocked-query specialization is unsuitable here. Both compressed
and CSR graphs use the same adapter. Unlike the original Jet experiment's legacy
baseline, UFM's existing large-k compact cache already enumerates adjacent blocks.
Both strategies therefore present only adjacent move targets to UFM. Their internal
traversal still differs: compact hashing scans degree-bound hash rows or all `k`
entries of a dense row, whereas Jet scans its current adaptive row capacity.

The worker-local large-k delta cache is shared by both strategies. It now records
all touched block connections, including connections initially present in the shared
cache and deltas that cancel to zero. This preserves local adjacency when another
worker removes the last shared connection. Queries combine the shared row and local
deltas, deduplicate targets, and omit nonpositive connectivity. Delta-only targets
use connectivity when the base cache reports connectivity, and exact gains when it
reports gains.

A reusable sparse scratch map holds values copied from each shared row. Consumers
run after the shared row lock has been released, avoiding reentrant locking and
keeping consumer work outside that lock. The scratch allocation follows its largest
observed row; it is worker-local and queries are not reentrant. Concurrent shared
moves can still change observations between reads, as elsewhere in speculative FM;
this is not an atomic snapshot of the complete partition or cache.

The [completed reduced cluster experiment](experiments/ufm-jet-cache-20261022/README.md)
uses three arms: unchanged stock UFM, corrected compact hashing, and the Jet cache
with the same delta-cache corrections. Comparing the latter two isolates the cache
selection; the stock arm also reveals the cost and trajectory changes introduced
by the common corrections and scratch map. No UFM default change follows merely
from making the prototype selectable.

The measured prototype improves large-k runtime relative to corrected compact hashing,
but increases peak process memory and regresses small-k runtime. Compared with unchanged
stock UFM it is slower at both measured k values. Compact hashing remains the selected
default; the report contains all three comparisons and their limitations.
