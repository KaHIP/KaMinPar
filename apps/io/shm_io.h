/*******************************************************************************
 * IO utilities for the shared-memory partitioner.
 *
 * @file:   shm_io.h
 * @author: Daniel Seemaier
 * @date:   21.09.2021
 ******************************************************************************/
#pragma once

#include <string>
#include <unordered_map>
#include <vector>

#include "kaminpar-shm/datastructures/graph.h"

namespace kaminpar::shm::io {

/*!
 * All graph file formats that can be parsed.
 */
enum class GraphFileFormat {
  METIS,
  PARHIP
};

/*!
 * Returns a table that maps identifiers to their corresponding graph file format.
 *
 * @return A table that maps identifiers to their corresponding graph file format.
 */
[[nodiscard]] std::unordered_map<std::string, GraphFileFormat> get_graph_file_formats();

/*!
 * Reads a graph that is either stored in METIS, ParHiP or compressed format.
 *
 * @param filename The name of the file to read.
 * @param file_format The format of the file used to store the graph.
 * @param compress Whether to compress the graph.
 * @param may_dismiss Whether the compressed graph is only returned when it uses less memory than
 * the uncompressed graph.
 * @param sorted Whether the nodes of the graph to read are stored in degree-buckets order.
 * @param validate Whether to validate the graph.
 * @return The graph to read.
 */
Graph read(
    const std::string &filename,
    const GraphFileFormat file_format,
    const bool compress,
    const bool may_dismiss,
    const bool sorted
);

namespace partition {

std::vector<BlockID> read(const std::string &filename);

void write(const std::string &filename, const std::vector<BlockID> &partition);

} // namespace partition

} // namespace kaminpar::shm::io
