/*******************************************************************************
 * Adaptive Jet gain cache adapter for asynchronous FM refinement.
 *
 * @file:   fm_jet_gain_cache.h
 * @author: Daniel Seemaier
 * @date:   22.10.2026
 ******************************************************************************/
#pragma once

#include "kaminpar-shm/refinement/gains/delta_gain_caches.h"
#include "kaminpar-shm/refinement/gains/jet_gain_cache.h"

namespace kaminpar::shm {

// FM workers query the shared cache while other workers commit moves. Keep query
// locking enabled; the phase-separated SynchronousJetGainCache is unsafe here.
template <typename GraphType> class FMJetGainCache : public JetGainCache<GraphType, true> {
public:
  using JetGainCache<GraphType, true>::JetGainCache;
  using DeltaGainCache = LargeKGenericDeltaGainCache<FMJetGainCache<GraphType>>;

  void print_statistics() const {}
};

} // namespace kaminpar::shm
