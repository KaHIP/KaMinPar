/*******************************************************************************
 * @file:   definitions.h
 * @author: Daniel Seemaier
 * @date:   27.10.2021
 * @brief:
 ******************************************************************************/
#pragma once

#include <cstdint>
#include <limits>

#include "kaminpar/definitions.h"

namespace kaminpar::dist {
using shm::NodeID;
using GlobalNodeID = std::uint64_t;
using shm::NodeWeight;
using GlobalNodeWeight = std::int64_t;
using shm::EdgeID;
using GlobalEdgeID = std::uint64_t;
using shm::EdgeWeight;
using GlobalEdgeWeight = std::int64_t;
using shm::BlockID;
using BlockWeight = std::int64_t;
using PEID        = int;

using shm::kInvalidNodeID;
constexpr GlobalNodeID kInvalidGlobalNodeID = std::numeric_limits<GlobalNodeID>::max();
using shm::kInvalidNodeWeight;
constexpr GlobalNodeWeight kInvalidGlobalNodeWeight = std::numeric_limits<GlobalNodeWeight>::max();
using shm::kInvalidEdgeID;
constexpr GlobalEdgeID kInvalidGlobalEdgeID = std::numeric_limits<GlobalEdgeID>::max();
using shm::kInvalidEdgeWeight;
constexpr GlobalEdgeWeight kInvalidGlobalEdgeWeight = std::numeric_limits<GlobalEdgeWeight>::max();
using shm::kInvalidBlockID;
using shm::kInvalidBlockWeight;
} // namespace kaminpar::dist
