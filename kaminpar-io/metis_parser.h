/*******************************************************************************
 * Sequential METIS parser.
 *
 * @file:   metis_parser.h
 * @author: Daniel Seemaier
 * @date:   26.10.2022
 ******************************************************************************/
#pragma once

#include <optional>
#include <string>

#include "kaminpar.h"

namespace kaminpar::shm::io::metis {

[[nodiscard]] std::optional<Graph> read_graph(
    const std::string &filename,
    bool compress = false,
    NodeOrdering ordering = NodeOrdering::NATURAL
);

void write_graph(const std::string &filename, const Graph &graph);

} // namespace kaminpar::shm::io::metis
