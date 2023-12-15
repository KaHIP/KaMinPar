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

#include "kaminpar-shm/datastructures/graph.h"
#include "kaminpar-shm/kaminpar.h"

namespace kaminpar::shm::io {
namespace metis {

/**
 * Reads a graph that is stored in a file in METIS format.
 *
 * @param filename The name of the file to read.
 * @tparam checked Whether to validate the read graph.
 * @return The graph in compressed sparse row format stored in the file.
 */
template <bool checked> CSRGraph csr_read(const std::string &filename);

/*!
 * Reads and compresses a graph that is stored in a file in METIS format.
 *
 * @param filename The name of the file to read.
 * @tparam checked Whether to validate the read graph.
 * @return The graph in compressed form stored in the file.
 */
template <bool checked> CompressedGraph compress_read(const std::string &filename);

} // namespace metis

/*!
 * Reads a graph that is either stored in METIS or compressed format.
 *
 * @param filename The name of the file to read.
 * @param compress Whether to compress the graph.
 * @param validate Whether to validate the graph.
 * @return The graph to read.
 */
Graph read(const std::string &filename, bool compress, bool validate);

namespace partition {
std::vector<BlockID> read(const std::string &filename);
void write(const std::string &filename, const std::vector<BlockID> &partition);
} // namespace partition
} // namespace kaminpar::shm::io
