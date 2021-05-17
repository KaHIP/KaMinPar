#pragma once

#include "datastructure/graph.h"
#include "definitions.h"
#include "utility/console_io.h"
#include "twopass_graph_builder.h"

#include <ranges>

namespace kaminpar::tool::converter {
struct Edge {
  NodeID u;
  NodeID v;
  EdgeWeight weight;
};

class EdgeListBuilder {
  static constexpr auto kDebug = false;

public:
  explicit EdgeListBuilder(const NodeID n) : _n{n} {}

  EdgeListBuilder(const EdgeListBuilder &) = delete;
  EdgeListBuilder &operator=(const EdgeListBuilder &) = delete;

  EdgeListBuilder(EdgeListBuilder &&) noexcept = default;
  EdgeListBuilder &operator=(EdgeListBuilder &&) noexcept = default;

  void add_edge(const NodeID u, const NodeID v, const EdgeWeight weight) {
    if (u < v) {
      _edge_list.push_back({u, v, weight});
    } else if (v < u) {
      _edge_list.push_back({v, u, weight});
    }
  }

  SimpleGraph build() {
    cio::ProgressBar progress(4, "Edgelist builder");
    progress.step("sorting edges");
    std::ranges::sort(_edge_list, [](const auto &a, const auto &b) { return a.u < b.u || (a.u == b.u && a.v < b.v); });

    progress.step("computing node degrees");
    TwoPassGraphBuilder builder(_n);
    if (!_edge_list.empty()) { builder.pass1_add_edge(_edge_list.front().u, _edge_list.front().v); }
    for (std::size_t i = 1; i < _edge_list.size(); ++i) {
      const Edge &previous = _edge_list[i - 1];
      const Edge &current = _edge_list[i];

      if (current.u == previous.u && current.v == previous.v) {
        ASSERT(current.weight == previous.weight)
            << "duplicate edge weights mismatch on edge " << current.u << " --> " << current.v << ": weight "
            << current.weight << " vs " << previous.weight;
      } else {
        builder.pass1_add_edge(current.u, current.v);
      }
    }

    progress.step("computing prefix sum");
    builder.pass1_finish();

    progress.step("adjacency array");
    if (!_edge_list.empty()) { builder.pass2_add_edge(_edge_list.front().u, _edge_list.front().v); }
    for (std::size_t i = 1; i < _edge_list.size(); ++i) {
      const Edge &previous = _edge_list[i - 1];
      const Edge &current = _edge_list[i];

      if (current.u != previous.u || current.v != previous.v) {
        builder.pass2_add_edge(current.u, current.v, current.weight);
      }
    }

    progress.stop();
    return builder.pass2_finish();
  }

private:
  const NodeID _n;
  std::vector<Edge> _edge_list{};
};
} // namespace kaminpar::tool::converter