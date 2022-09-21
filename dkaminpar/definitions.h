/*******************************************************************************
 * @file:   definitions.h
 * @author: Daniel Seemaier
 * @date:   27.10.2021
 * @brief:  Basic data types used by the distributed graph partitioner.
 ******************************************************************************/
#pragma once

#include <cstdint>
#include <limits>

#include "kaminpar/definitions.h"

namespace kaminpar::mpi {
using PEID = int;
}

namespace kaminpar::dist {
using GlobalNodeID     = std::uint64_t;
using GlobalNodeWeight = std::int64_t;
using GlobalEdgeID     = std::uint64_t;
using GlobalEdgeWeight = std::int64_t;
using BlockWeight      = std::int64_t;

using mpi::PEID;

using shm::BlockID;
using shm::EdgeID;
using shm::NodeID;

#ifdef KAMINPAR_64BIT_LOCAL_WEIGHTS
using NodeWeight = std::int64_t;
using EdgeWeight = std::int64_t;
#else
using NodeWeight = std::int32_t;
using EdgeWeight = std::int32_t;
#endif

constexpr NodeID           kInvalidNodeID           = std::numeric_limits<NodeID>::max();
constexpr GlobalNodeID     kInvalidGlobalNodeID     = std::numeric_limits<GlobalNodeID>::max();
constexpr NodeWeight       kInvalidNodeWeight       = std::numeric_limits<NodeWeight>::max();
constexpr GlobalNodeWeight kInvalidGlobalNodeWeight = std::numeric_limits<GlobalNodeWeight>::max();
constexpr EdgeID           kInvalidEdgeID           = std::numeric_limits<EdgeID>::max();
constexpr GlobalEdgeID     kInvalidGlobalEdgeID     = std::numeric_limits<GlobalEdgeID>::max();
constexpr EdgeWeight       kInvalidEdgeWeight       = std::numeric_limits<EdgeWeight>::max();
constexpr GlobalEdgeWeight kInvalidGlobalEdgeWeight = std::numeric_limits<GlobalEdgeWeight>::max();
constexpr BlockID          kInvalidBlockID          = std::numeric_limits<BlockID>::max();
constexpr BlockWeight      kInvalidBlockWeight      = std::numeric_limits<BlockWeight>::max();
} // namespace kaminpar::dist
