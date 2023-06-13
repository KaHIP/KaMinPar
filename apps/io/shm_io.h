/*******************************************************************************
 * @file:   shm_io.h
 * @author: Daniel Seemaier
 * @date:   21.09.2021
 * @brief:  Graph and partition IO functions.
 ******************************************************************************/
#pragma once

#include "kaminpar/datastructures/graph.h"
#include "kaminpar/definitions.h"

#include "common/assertion_levels.h"

#include "apps/io/metis_parser.h"
#include "apps/io/mmap_toker.h"

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
