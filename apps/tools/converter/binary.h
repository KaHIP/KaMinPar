/*******************************************************************************
 * This file is part of KaMinPar.
 *
 * Copyright (C) 2021 Daniel Seemaier <daniel.seemaier@kit.edu>
 *
 * KaMinPar is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * KaMinPar is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with KaMinPar.  If not, see <http://www.gnu.org/licenses/>.
 *
******************************************************************************/
#pragma once

#include <fstream>

#define HAS_NODE_WEIGHTS(version) (((version) & 0b10) == 1)
#define HAS_EDGE_WEIGHTS(version) (((version) & 0b01) == 1)

namespace kaminpar::tool::converter {

class BinaryReader : public GraphReader {
public:
  SimpleGraph read(const std::string &filename) override {
    std::FILE *fd = fopen(filename.c_str(), "r");
    if (!fd) { FATAL_PERROR << "cannot read from " << filename; }

    std::uint64_t version;
    std::uint64_t n;
    std::uint64_t m;
    ALWAYS_ASSERT(std::fread(&version, sizeof(std::uint64_t), 1, fd) == 1);
    ALWAYS_ASSERT(std::fread(&n, sizeof(std::uint64_t), 1, fd) == 1);
    ALWAYS_ASSERT(std::fread(&m, sizeof(std::uint64_t), 1, fd) == 1);

    const bool has_node_weights = HAS_NODE_WEIGHTS(version);
    const bool has_edge_weights = HAS_EDGE_WEIGHTS(version);

    SimpleGraph graph;
    {
      std::vector<std::uint64_t> nodes(n + 1);
      ALWAYS_ASSERT(std::fread(nodes.data(), sizeof(std::uint64_t), n + 1, fd) == n + 1);
      graph.nodes.resize(n + 1);
      std::ranges::copy(nodes, graph.nodes.begin());
    }
    if (has_node_weights) {
      std::vector<std::int64_t> node_weights(n);
      ALWAYS_ASSERT(std::fread(node_weights.data(), sizeof(std::int64_t), n, fd) == n);
      graph.node_weights.resize(n);
      std::ranges::copy(node_weights, graph.node_weights.begin());
    }
    {
      std::vector<std::uint64_t> edges(m);
      ALWAYS_ASSERT(std::fread(edges.data(), sizeof(std::uint64_t), m, fd) == m);
      graph.edges.resize(m);
      std::ranges::copy(edges, graph.edges.begin());
    }
    if (has_edge_weights) {
      std::vector<std::int64_t> edge_weights(m);
      ALWAYS_ASSERT(std::fread(edge_weights.data(), sizeof(std::int64_t), m, fd) == m);
      graph.edge_weights.resize(m);
      std::ranges::copy(edge_weights, graph.edge_weights.begin());
    }

    std::fclose(fd);
    return {};
  }

  [[nodiscard]] std::string description() const override { return "Binary KaMinPar graph format"; }

  [[nodiscard]] std::string default_extension() const override { return "bin"; }
};

class BinaryWriter : public GraphWriter {
public:
  void write(const std::string &filename, SimpleGraph graph, const std::string &) override {
    std::FILE *fd = std::fopen(filename.c_str(), "w");
    if (!fd) { FATAL_PERROR << "cannot write to " << filename; }

    std::uint64_t version;
    if (graph.has_node_weights() && graph.has_edge_weights()) {
      version = 3;
    } else if (graph.has_node_weights()) {
      version = 2;
    } else if (graph.has_edge_weights()) {
      version = 1;
    } else {
      version = 0;
    }

    const std::uint64_t n = graph.n();
    const std::uint64_t m = graph.m();

    ALWAYS_ASSERT(std::fwrite(&version, sizeof(std::uint64_t), 1, fd) == 1);
    ALWAYS_ASSERT(std::fwrite(&n, sizeof(std::uint64_t), 1, fd) == 1);
    ALWAYS_ASSERT(std::fwrite(&m, sizeof(std::uint64_t), 1, fd) == 1);

    {
      std::vector<std::uint64_t> nodes(graph.n() + 1);
      std::ranges::copy(graph.nodes, nodes.begin());
      ALWAYS_ASSERT(std::fwrite(nodes.data(), sizeof(std::uint64_t), graph.n() + 1, fd) == graph.n() + 1);
    }
    if (graph.has_node_weights()) {
      std::vector<std::int64_t> node_weights(graph.n());
      std::ranges::copy(graph.node_weights, node_weights.begin());
      ALWAYS_ASSERT(std::fwrite(node_weights.data(), sizeof(std::int64_t), graph.n(), fd) == graph.n());
    }
    {
      std::vector<std::uint64_t> edges(graph.m());
      std::ranges::copy(graph.edges, edges.begin());
      ALWAYS_ASSERT(std::fwrite(edges.data(), sizeof(std::uint64_t), graph.m(), fd) == graph.m());
    }
    if (graph.has_edge_weights()) {
      std::vector<std::int64_t> edge_weights(graph.m());
      std::ranges::copy(graph.edge_weights, edge_weights.begin());
      ALWAYS_ASSERT(std::fwrite(edge_weights.data(), sizeof(std::int64_t), graph.m(), fd) == graph.m());
    }

    std::fclose(fd);
  }

  [[nodiscard]] std::string description() const override { return "Binary KaMinPar graph format"; }

  [[nodiscard]] std::string default_extension() const override { return "bin"; }
};
} // namespace kaminpar::tool::converter