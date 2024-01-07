/*******************************************************************************
 * IO utilities for the shared-memory partitioner.
 *
 * @file:   shm_io.h
 * @author: Daniel Seemaier
 * @date:   21.09.2021
 ******************************************************************************/
#pragma once

#include <optional>
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
 * @param sorted Whether the nodes of the graph to read are stored in deg-buckets order.
 * @tparam checked Whether to validate the read graph.
 * @return The graph in compressed sparse row format stored in the file.
 */
template <bool checked> CSRGraph csr_read(const std::string &filename, const bool sorted = false);

/*!
 * Reads and compresses a graph that is stored in a file in METIS format.
 *
 * @param filename The name of the file to read.
 * @param sorted Whether the nodes of the graph to read are stored in deg-buckets order.
 * @param may_dismiss Whether the reading process is aborted when the compressed graph uses more
 * memory than the uncompressed graph.
 * @tparam checked Whether to validate the read graph.
 * @return The graph in compressed form stored in the file.
 */
template <bool checked>
std::optional<CompressedGraph> compress_read(
    const std::string &filename, const bool sorted = false, const bool may_dismiss = false
);

/*!
 * Writes a graph to a file in METIS format.
 *
 * @param filename The name of the file for saving the graph.
 * @param graph The graph to save.
 */
void write(const std::string &filename, const Graph &graph);

} // namespace metis

/*!
 * Reads a graph that is either stored in METIS or compressed format.
 *
 * @param filename The name of the file to read.
 * @param compress Whether to compress the graph.
 * @param may_dismiss Whether the compressed graph is only returned when it uses less memory than
 * the uncompressed graph.
 * @param sorted Whether the nodes of the graph to read are stored in deg-buckets order.
 * @param validate Whether to validate the graph.
 * @return The graph to read.
 */
Graph read(
    const std::string &filename,
    const bool compress,
    const bool may_dismiss,
    const bool sorted,
    const bool validate
);

namespace partition {
std::vector<BlockID> read(const std::string &filename);
void write(const std::string &filename, const std::vector<BlockID> &partition);
} // namespace partition
} // namespace kaminpar::shm::io
