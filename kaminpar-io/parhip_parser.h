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

#include "kaminpar-shm/kaminpar.h"

namespace kaminpar::shm::io::parhip {

[[nodiscard]] std::optional<Graph> read_graph(
    const std::string &filename,
    bool compress = false,
    NodeOrdering ordering = NodeOrdering::NATURAL
);

void write_graph(const std::string &filename, const Graph &graph);

} // namespace kaminpar::shm::io::parhip
