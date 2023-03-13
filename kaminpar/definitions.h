/*******************************************************************************
 * @file:   definitions.h
 * @author: Daniel Seemaier
 * @date:   21.09.2021
 * @brief:  General type and macro definitions.
 ******************************************************************************/
#pragma once

#include <cstdint>
#include <limits>
#include <vector>

namespace kaminpar::shm {
#ifdef KAMINPAR_64BIT_NODE_IDS
using NodeID = std::uint64_t;
#else  // KAMINPAR_64BIT_NODE_IDS
using NodeID = std::uint32_t;
#endif // KAMINPAR_64BIT_NODE_IDS

#ifdef KAMINPAR_64BIT_EDGE_IDS
using EdgeID = std::uint64_t;
#else  // KAMINPAR_64BIT_EDGE_IDS
using EdgeID = std::uint32_t;
#endif // KAMINPAR_64BIT_EDGE_IDS

#ifdef KAMINPAR_64BIT_WEIGHTS
using NodeWeight = std::int64_t;
using EdgeWeight = std::int64_t;
#else  // KAMINPAR_64BIT_WEIGHTS
using NodeWeight = std::int32_t;
using EdgeWeight = std::int32_t;
#endif // KAMINPAR_64BIT_WEIGHTS

using BlockID = std::uint32_t;
using BlockWeight = NodeWeight;
using Gain = EdgeWeight;
using Degree = EdgeID;
using Clustering = std::vector<NodeID>;

constexpr BlockID kInvalidBlockID = std::numeric_limits<BlockID>::max();
constexpr NodeID kInvalidNodeID = std::numeric_limits<NodeID>::max();
constexpr EdgeID kInvalidEdgeID = std::numeric_limits<EdgeID>::max();
constexpr NodeWeight kInvalidNodeWeight =
    std::numeric_limits<NodeWeight>::max();
constexpr EdgeWeight kInvalidEdgeWeight =
    std::numeric_limits<EdgeWeight>::max();
constexpr BlockWeight kInvalidBlockWeight =
    std::numeric_limits<BlockWeight>::max();
constexpr Degree kMaxDegree = std::numeric_limits<Degree>::max();
} // namespace kaminpar::shm
