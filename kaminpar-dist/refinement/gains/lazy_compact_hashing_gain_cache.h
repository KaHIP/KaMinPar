/*******************************************************************************
 * @file:   lazy_compact_hashing_gain_cache.h
 * @author: Daniel Seemaier
 * @date:   19.12.2024
 ******************************************************************************/
#pragma once

#include "kaminpar-dist/refinement/gains/compact_hashing_gain_cache.h"

namespace kaminpar::dist {

template <typename DistributedGraph>
using LazyCompactHashingGainCache = CompactHashingGainCache<DistributedGraph, true>;

} // namespace kaminpar::dist
