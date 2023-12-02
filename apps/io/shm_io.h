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

#include "kaminpar-shm/datastructures/compressed_graph.h"
#include "kaminpar-shm/kaminpar.h"

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

/**
 * Reads a graph that is stored in a file in METIS format.
 *
 * @param filename The name of the file to read.
 * @tparam checked Whether to validate the read graph.
 *
 * @return The graph in compressed sparse row format stored in the file.
 */
template <bool checked> CSRGraph csr_read(const std::string &filename);

/*!
 * Reads and compresses a graph that is stored in a file in METIS format.
 *
 * @param filename The name of the file to read.
 * @tparam checked Whether to validate the read graph.
 *
 * @return The graph in compressed form stored in the file.
 */
template <bool checked> CompressedGraph compress_read(const std::string &filename);

} // namespace metis

namespace partition {
std::vector<BlockID> read(const std::string &filename);
void write(const std::string &filename, const std::vector<BlockID> &partition);
} // namespace partition
} // namespace kaminpar::shm::io
