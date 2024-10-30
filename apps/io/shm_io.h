/*******************************************************************************
 * IO utilities for the shared-memory partitioner.
 *
 * @file:   shm_io.h
 * @author: Daniel Seemaier
 * @date:   21.09.2021
 ******************************************************************************/
#pragma once

#include <span>
#include <string>
#include <unordered_map>
#include <vector>

#include "kaminpar-shm/datastructures/compressed_graph.h"
#include "kaminpar-shm/datastructures/csr_graph.h"
#include "kaminpar-shm/datastructures/graph.h"
#include "kaminpar-shm/kaminpar.h"

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

/**
 * Reads a graph that is stored in METIS or ParHip format.
 *
 * @param filename The name of the file to read.
 * @param file_format The format of the file used to store the graph.
 * @param ordering The node ordering of the graph to read.
 * @return The graph that is stored in the file.
 */
CSRGraph csr_read(
    const std::string &filename, const GraphFileFormat file_format, const NodeOrdering ordering
);

/**
 * Reads a graph that is stored in METIS or ParHip format.
 *
 * @param filename The name of the file to read.
 * @param file_format The format of the file used to store the graph.
 * @param ordering The node ordering of the graph to read.
 * @return The graph that is stored in the file.
 */
CompressedGraph compressed_read(
    const std::string &filename, const GraphFileFormat file_format, const NodeOrdering ordering
);

/*!
 * Reads a graph that is either stored in METIS, ParHiP or compressed format.
 *
 * @param filename The name of the file to read.
 * @param file_format The format of the file used to store the graph.
 * @param ordering The node ordering of the graph to read.
 * @param compress Whether to compress the graph.
 * @return The graph that is stored in the file.
 */
Graph read(
    const std::string &filename,
    const GraphFileFormat file_format,
    const NodeOrdering ordering,
    const bool compress
);

namespace partition {

StaticArray<BlockID> read(const std::string &filename);

StaticArray<BlockID> read_block_sizes(const std::string &filename);

void write(const std::string &filename, std::span<const BlockID> partition);

void write_block_sizes(const std::string &filename, BlockID k, std::span<const BlockID> partition);

} // namespace partition

} // namespace kaminpar::shm::io
