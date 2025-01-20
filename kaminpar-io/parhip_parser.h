/*******************************************************************************
 * Sequential and parallel ParHiP parser.
 *
 * @file:   parhip_parser.h
 * @author: Daniel Salwasser
 * @date:   15.02.2024
 ******************************************************************************/
#pragma once

#include <optional>
#include <string>

#include "kaminpar-shm/datastructures/compressed_graph.h"
#include "kaminpar-shm/datastructures/csr_graph.h"
#include "kaminpar-shm/kaminpar.h"

namespace kaminpar::shm::io::parhip {

/**
 * Reads a graph that is stored in a file with ParHIP format.
 *
 * @param filename The name of the file to read.
 * @param sorted Whether the nodes of the graph to read are stored in degree-buckets order.
 * @return The graph that is stored in the file.
 */
[[nodiscard]] std::optional<CSRGraph>
csr_read(const std::string &filename, const NodeOrdering ordering = NodeOrdering::NATURAL);

/*!
 * Reads and compresses a graph that is stored in a file with ParHiP format.
 *
 * @param filename The name of the file to read.
 * @param sorted Whether the nodes of the graph to read are stored in degree-buckets order.
 * @return The graph that is stored in the file.
 */
[[nodiscard]] std::optional<CompressedGraph>
compressed_read(const std::string &filename, const NodeOrdering ordering = NodeOrdering::NATURAL);

/*!
 * Writes a graph to a file in ParHIP format.
 *
 * @param filename The name of the file in which to store the graph.
 * @param graph The graph to store.
 */
void write(const std::string &filename, const CSRGraph &graph);

} // namespace kaminpar::shm::io::parhip
