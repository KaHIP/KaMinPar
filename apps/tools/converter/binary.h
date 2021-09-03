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

#define HAS_NODE_WEIGHTS(version) (((version) & 0b10) == 0)
#define HAS_EDGE_WEIGHTS(version) (((version) & 0b01) == 0)

namespace kaminpar::tool::converter {
using parhip_ulong_t = unsigned long long;
using parhip_long_t = signed long long;

class BinaryReader : public GraphReader {
public:
  SimpleGraph read(const std::string &filename) override {
    std::FILE *fd = fopen(filename.c_str(), "r");
    if (!fd) { FATAL_PERROR << "cannot read from " << filename; }

    parhip_ulong_t version;
    parhip_ulong_t n;
    parhip_ulong_t m;
    ALWAYS_ASSERT(std::fread(&version, sizeof(parhip_ulong_t), 1, fd) == 1);
    ALWAYS_ASSERT(std::fread(&n, sizeof(parhip_ulong_t), 1, fd) == 1);
    ALWAYS_ASSERT(std::fread(&m, sizeof(parhip_ulong_t), 1, fd) == 1);

    const bool has_node_weights = HAS_NODE_WEIGHTS(version);
    const bool has_edge_weights = HAS_EDGE_WEIGHTS(version);

    SimpleGraph graph;
    {
      std::vector<parhip_ulong_t> nodes(n + 1);
      ALWAYS_ASSERT(std::fread(nodes.data(), sizeof(parhip_ulong_t), n + 1, fd) == n + 1);
      graph.nodes.resize(n + 1);
      std::ranges::copy(nodes, graph.nodes.begin());
    }
    if (has_node_weights) {
      std::vector<parhip_long_t> node_weights(n);
      ALWAYS_ASSERT(std::fread(node_weights.data(), sizeof(parhip_long_t), n, fd) == n);
      graph.node_weights.resize(n);
      std::ranges::copy(node_weights, graph.node_weights.begin());
    }
    {
      std::vector<parhip_ulong_t> edges(m);
      ALWAYS_ASSERT(std::fread(edges.data(), sizeof(parhip_ulong_t), m, fd) == m);
      graph.edges.resize(m);
      std::ranges::copy(edges, graph.edges.begin());
    }
    if (has_edge_weights) {
      std::vector<parhip_long_t> edge_weights(m);
      ALWAYS_ASSERT(std::fread(edge_weights.data(), sizeof(parhip_long_t), m, fd) == m);
      graph.edge_weights.resize(m);
      std::ranges::copy(edge_weights, graph.edge_weights.begin());
    }

    std::fclose(fd);
    return {};
  }

  [[nodiscard]] std::string description() const override { return "METIS graph format"; }
};

class BinaryWriter : public GraphWriter {
public:
  void write(const std::string &filename, SimpleGraph graph, const std::string &) override {
    std::FILE *fd = std::fopen(filename.c_str(), "w");
    if (!fd) { FATAL_PERROR << "cannot write to " << filename; }

    parhip_ulong_t version;
    if (graph.has_node_weights() && graph.has_edge_weights()) {
      version = 0;
    } else if (graph.has_node_weights()) {
      version = 1;
    } else if (graph.has_edge_weights()) {
      version = 2;
    } else {
      version = 3;
    }

    const parhip_ulong_t n = graph.n();
    const parhip_ulong_t m = graph.m();

    ALWAYS_ASSERT(std::fwrite(&version, sizeof(parhip_ulong_t), 1, fd) == 1);
    ALWAYS_ASSERT(std::fwrite(&n, sizeof(parhip_ulong_t), 1, fd) == 1);
    ALWAYS_ASSERT(std::fwrite(&m, sizeof(parhip_ulong_t), 1, fd) == 1);

    {
      std::vector<parhip_ulong_t> nodes(graph.n() + 1);
      std::ranges::copy(graph.nodes, nodes.begin());
      ALWAYS_ASSERT(std::fwrite(nodes.data(), sizeof(parhip_ulong_t), graph.n() + 1, fd) == graph.n() + 1);
    }
    if (graph.has_node_weights()) {
      std::vector<parhip_long_t> node_weights(graph.n());
      std::ranges::copy(graph.node_weights, node_weights.begin());
      ALWAYS_ASSERT(std::fwrite(node_weights.data(), sizeof(parhip_long_t), graph.n(), fd) == graph.n());
    }
    {
      std::vector<parhip_ulong_t> edges(graph.m());
      std::ranges::copy(graph.edges, edges.begin());
      ALWAYS_ASSERT(std::fwrite(edges.data(), sizeof(parhip_ulong_t), graph.m(), fd) == graph.m());
    }
    if (graph.has_edge_weights()) {
      std::vector<parhip_long_t> edge_weights(graph.m());
      std::ranges::copy(graph.edge_weights, edge_weights.begin());
      ALWAYS_ASSERT(std::fwrite(edge_weights.data(), sizeof(parhip_long_t), graph.m(), fd) == graph.m());
    }

    std::fclose(fd);
  }

  [[nodiscard]] std::string description() const override { return "METIS graph format"; }
};
} // namespace kaminpar::tool::converter