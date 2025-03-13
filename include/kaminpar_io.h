/*******************************************************************************
 * IO utilities for the shared-memory partitioner.
 *
 * @file:   kaminpar_io.h
 * @author: Daniel Seemaier
 * @date:   21.09.2021
 ******************************************************************************/
#ifndef KAMINPAR_IO_H
#define KAMINPAR_IO_H

#include <optional>
#include <span>
#include <string>

#include <kaminpar.h>

namespace kaminpar::shm::io {

/*!
 * All graph file formats that can be parsed.
 */
enum class GraphFileFormat {
  METIS,
  PARHIP,
  COMPRESSED,
};

/*!
 * Reads a graph that is either stored in METIS, ParHiP or compressed format.
 *
 * @param filename The name of the file to read.
 * @param file_format The format of the file used to store the graph.
 * @param ordering The node ordering of the graph to read.
 * @param compress Whether to compress the graph.
 * @return The graph that is stored in the file.
 */
[[nodiscard]] std::optional<Graph> read_graph(
    const std::string &filename,
    GraphFileFormat file_format,
    bool compress = false,
    NodeOrdering ordering = NodeOrdering::NATURAL
);

void write_graph(const std::string &filename, GraphFileFormat file_format, const Graph &graph);

std::vector<BlockID> read_partition(const std::string &filename);

void write_partition(const std::string &filename, std::span<const BlockID> partition);

std::vector<BlockID> read_block_sizes(const std::string &filename);

void write_block_sizes(const std::string &filename, BlockID k, std::span<const BlockID> partition);

void write_remapping(const std::string &filename, std::span<const NodeID> partition);

} // namespace kaminpar::shm::io

#endif
