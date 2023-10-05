/*******************************************************************************
 * IO utilities for the shared-memory partitioner.
 *
 * @file:   shm_io.h
 * @author: Daniel Seemaier
 * @date:   21.09.2021
 ******************************************************************************/
#pragma once

#include <string>
#include <vector>

#include "kaminpar-shm/definitions.h"

#include "kaminpar-common/datastructures/static_array.h"

namespace kaminpar::shm::io {
namespace metis {
template <bool checked>
void read(
    const std::string &filename,
    StaticArray<EdgeID> &nodes,
    StaticArray<NodeID> &edges,
    StaticArray<NodeWeight> &node_weights,
    StaticArray<EdgeWeight> &edge_weights
);
} // namespace metis

namespace partition {
std::vector<BlockID> read(const std::string &filename);
void write(const std::string &filename, const std::vector<BlockID> &partition);
} // namespace partition
} // namespace kaminpar::shm::io
