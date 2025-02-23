/*******************************************************************************
 * IO utilities for the compressed graph binary.
 *
 * @file:   graph_compression_binary.h
 * @author: Daniel Salwasser
 * @date:   12.12.2023
 ******************************************************************************/
#pragma once

#include <cstdint>
#include <optional>
#include <string>

#include "kaminpar-shm/datastructures/compressed_graph.h"

namespace kaminpar::shm::io::compressed_binary {
//! Magic number to identify a compressed graph binary file.
constexpr std::uint64_t kMagicNumber = 0x434F4D5052455353;

void write(const std::string &filename, const CompressedGraph &graph);

[[nodiscard]] std::optional<Graph> read(const std::string &filename);

bool is_compressed(const std::string &filename);

} // namespace kaminpar::shm::io::compressed_binary
