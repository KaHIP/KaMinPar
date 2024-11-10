/*******************************************************************************
 * Static uncompressed CSR graph data structure.
 *
 * @file:   csr_graph.cc
 * @author: Daniel Seemaier
 * @date:   21.09.2021
 ******************************************************************************/
#include "kaminpar-shm/datastructures/csr_graph.h"

#include <numeric>

#include <tbb/enumerable_thread_specific.h>
#include <tbb/parallel_reduce.h>

#include "kaminpar-shm/datastructures/graph.h"

#include "kaminpar-common/logger.h"
#include "kaminpar-common/parallel/algorithm.h"

namespace kaminpar::shm {

CSRGraph::CSRGraph(const Graph &graph)
    : _nodes(graph.n() + 1),
      _edges(graph.m()),
      _node_weights(graph.n()),
      _edge_weights(graph.m()),
      _buckets(kNumberOfDegreeBuckets<NodeID> + 1) {
  graph.reified([&](const auto &graph) {
    _nodes.front() = 0;
    graph.pfor_nodes([&](const NodeID u) {
      _nodes[u + 1] = graph.degree(u);
      _node_weights[u] = graph.node_weight(u);
    });
    parallel::prefix_sum(_nodes.begin(), _nodes.end(), _nodes.begin());

    graph.pfor_nodes([&](const NodeID u) {
      graph.neighbors(u, [&](const EdgeID e, const NodeID v, const EdgeWeight w) {
        _edges[e] = v;
        _edge_weights[e] = w;
      });
    });

    _total_node_weight = graph.total_node_weight();
    _total_edge_weight = graph.total_edge_weight();
    _max_degree = graph.max_degree();
    init_degree_buckets();
  });
}

CSRGraph::CSRGraph(
    StaticArray<EdgeID> nodes,
    StaticArray<NodeID> edges,
    StaticArray<NodeWeight> node_weights,
    StaticArray<EdgeWeight> edge_weights,
    bool sorted
)
    : _nodes(std::move(nodes)),
      _edges(std::move(edges)),
      _node_weights(std::move(node_weights)),
      _edge_weights(std::move(edge_weights)),
      _sorted(sorted),
      _buckets(kNumberOfDegreeBuckets<NodeID> + 1) {
  if (_node_weights.empty()) {
    _total_node_weight = static_cast<NodeWeight>(n());
    _max_node_weight = 1;
  } else {
    _total_node_weight = parallel::accumulate(_node_weights, static_cast<NodeWeight>(0));
    _max_node_weight = parallel::max_element(_node_weights);
  }

  if (_edge_weights.empty()) {
    _total_edge_weight = static_cast<EdgeWeight>(m());
  } else {
    _total_edge_weight = parallel::accumulate(_edge_weights, static_cast<EdgeWeight>(0));
  }

  _max_degree = parallel::max_difference(_nodes.begin(), _nodes.end());

  init_degree_buckets();
}

CSRGraph::CSRGraph(
    seq,
    StaticArray<EdgeID> nodes,
    StaticArray<NodeID> edges,
    StaticArray<NodeWeight> node_weights,
    StaticArray<EdgeWeight> edge_weights,
    bool sorted,
    std::vector<NodeID> buckets
)
    : _nodes(std::move(nodes)),
      _edges(std::move(edges)),
      _node_weights(std::move(node_weights)),
      _edge_weights(std::move(edge_weights)),
      _sorted(sorted),
      _buckets(std::move(buckets)) {
  if (_node_weights.empty()) {
    _total_node_weight = static_cast<NodeWeight>(n());
    _max_node_weight = 1;
  } else {
    _total_node_weight =
        std::accumulate(_node_weights.begin(), _node_weights.end(), static_cast<NodeWeight>(0));
    _max_node_weight = *std::max_element(_node_weights.begin(), _node_weights.end());
  }

  if (_edge_weights.empty()) {
    _total_edge_weight = static_cast<EdgeWeight>(m());
  } else {
    _total_edge_weight =
        std::accumulate(_edge_weights.begin(), _edge_weights.end(), static_cast<EdgeWeight>(0));
  }

  // TODO: Use a sequential routine to initialize degree buckets since work isolation can otherwise
  // be violated. However, this does not currently pose a problem as it is not called in parallel.
  init_degree_buckets();
}

void CSRGraph::update_total_node_weight() {
  if (_node_weights.empty()) {
    _total_node_weight = n();
    _max_node_weight = 1;
  } else {
    _total_node_weight =
        std::accumulate(_node_weights.begin(), _node_weights.end(), static_cast<NodeWeight>(0));
    _max_node_weight = *std::max_element(_node_weights.begin(), _node_weights.end());
  }
}

void CSRGraph::remove_isolated_nodes(const NodeID num_isolated_nodes) {
  KASSERT(sorted());

  if (num_isolated_nodes == 0) {
    return;
  }

  const NodeID new_n = n() - num_isolated_nodes;
  _nodes.restrict(new_n + 1);
  if (!_node_weights.empty()) {
    _node_weights.restrict(new_n);
  }

  update_total_node_weight();

  // Update degree buckets
  for (std::size_t i = 0; i < _buckets.size() - 1; ++i) {
    _buckets[1 + i] -= num_isolated_nodes;
  }

  // If the graph has only isolated nodes then there are no buckets afterwards
  if (_number_of_buckets == 1) {
    _number_of_buckets = 0;
  }
}

void CSRGraph::integrate_isolated_nodes() {
  KASSERT(sorted());

  const NodeID nonisolated_nodes = n();
  _nodes.unrestrict();
  _node_weights.unrestrict();

  const NodeID isolated_nodes = n() - nonisolated_nodes;
  update_total_node_weight();

  // Update degree buckets
  for (std::size_t i = 0; i < _buckets.size() - 1; ++i) {
    _buckets[1 + i] += isolated_nodes;
  }

  // If the graph has only isolated nodes then there is one afterwards
  if (_number_of_buckets == 0) {
    _number_of_buckets = 1;
  }
}

void CSRGraph::init_degree_buckets() {
  KASSERT(std::all_of(_buckets.begin(), _buckets.end(), [](const auto n) { return n == 0; }));

  constexpr std::size_t kNumBuckets = kNumberOfDegreeBuckets<NodeID> + 1;

  if (_sorted) {
    tbb::enumerable_thread_specific<std::array<NodeID, kNumBuckets>> buckets_ets([&] {
      return std::array<NodeID, kNumBuckets>{};
    });

    tbb::parallel_for(tbb::blocked_range<NodeID>(0, n()), [&](const tbb::blocked_range<NodeID> r) {
      auto &buckets = buckets_ets.local();
      for (NodeID u = r.begin(); u != r.end(); ++u) {
        ++buckets[degree_bucket(degree(u)) + 1];
      }
    });

    std::fill(_buckets.begin(), _buckets.end(), 0);
    for (auto &local_buckets : buckets_ets) {
      for (std::size_t i = 0; i < kNumBuckets; ++i) {
        _buckets[i] += local_buckets[i];
      }
    }

    KASSERT(
        [&] {
          std::vector<NodeID> buckets2(_buckets.size());
          for (const NodeID u : nodes()) {
            ++buckets2[degree_bucket(degree(u)) + 1];
          }
          for (std::size_t i = 0; i < _buckets.size(); ++i) {
            if (_buckets[i] != buckets2[i]) {
              return false;
            }
          }
          return true;
        }(),
        "",
        assert::heavy
    );
    auto last_nonempty_bucket =
        std::find_if(_buckets.rbegin(), _buckets.rend(), [](const auto n) { return n > 0; });
    _number_of_buckets = std::distance(_buckets.begin(), (last_nonempty_bucket + 1).base());
  } else {
    _buckets[1] = n();
    _number_of_buckets = 1;
  }

  std::partial_sum(_buckets.begin(), _buckets.end(), _buckets.begin());
}

namespace debug {
bool validate_graph(
    const CSRGraph &graph, const bool check_undirected, const NodeID num_pseudo_nodes
) {
  return validate_graph(
      graph.n(),
      graph.raw_nodes(),
      graph.raw_edges(),
      graph.raw_node_weights(),
      graph.raw_edge_weights(),
      check_undirected,
      num_pseudo_nodes
  );
}

bool validate_graph(
    const NodeID n,
    const StaticArray<EdgeID> &xadj,
    const StaticArray<NodeID> &adjncy,
    const StaticArray<NodeWeight> &vwgt,
    const StaticArray<EdgeWeight> &adjwgt,
    const bool check_undirected,
    const NodeID num_pseudo_nodes
) {
  if (xadj.size() < n + 1) {
    LOG_WARNING << "xadj is not large enough: " << xadj.size() << " < " << n + 1;
    return false;
  }

  if (!vwgt.empty() && vwgt.size() < n) {
    LOG_WARNING << "vwgt is not large enough: " << vwgt.size() << " < " << n;
    return false;
  }

  const EdgeID m = xadj[n];

  if (!adjwgt.empty() && adjwgt.size() < m) {
    LOG_WARNING << "adjwgt is not large enough: " << adjwgt.size() << " < " << xadj[n];
    return false;
  }

  for (NodeID u = 0; u < n; ++u) {
    if (xadj[u] > xadj[u + 1]) {
      LOG_WARNING << "Bad node array at position " << u;
      return false;
    }
  }

  for (NodeID u = 0; u < n; ++u) {
    for (EdgeID e = xadj[u]; e < xadj[u + 1]; ++e) {
      if (e >= m) {
        LOG_WARNING << "Edge " << e << " of " << u << " is out-of-graph";
        return false;
      }

      const NodeID v = adjncy[e];

      if (v >= n) {
        LOG_WARNING << "Neighbor " << v << " of " << u << " is out-of-graph";
        return false;
      }

      if (u == v) {
        LOG_WARNING << "Self-loop at " << u << ": " << e << " --> " << v;
        return false;
      }

      bool found_reverse = false;
      for (EdgeID e_prime = xadj[v]; e_prime < xadj[v + 1]; ++e_prime) {
        if (e_prime >= m) {
          LOG_WARNING << "Edge " << e_prime << " of " << v << " is out-of-graph";
          return false;
        }

        const NodeID u_prime = adjncy[e_prime];

        if (u_prime >= n) {
          LOG_WARNING << "Neighbor " << u_prime << " of neighbor " << v << " of " << u
                      << " is out-of-graph";
          return false;
        }

        if (u != u_prime) {
          continue;
        }

        if (!adjwgt.empty() && adjwgt[e] != adjwgt[e_prime]) {
          LOG_WARNING << "Weight of edge " << e << " (" << adjwgt[e] << ") "              //
                      << "differs from the weight of its reverse edge " << e_prime << " " //
                      << "(" << adjwgt[e_prime] << ")";                                   //
          return false;
        }

        found_reverse = true;
        break;
      }

      if (check_undirected && v < n - num_pseudo_nodes && !found_reverse) {
        LOG_WARNING << "Edge " << u << " --> " << v << " exists with edge " << e
                    << ", but the reverse edges does not exist";
        return false;
      }
    }
  }

  return true;
}

CSRGraph sort_neighbors(CSRGraph graph) {
  const bool sorted = graph.sorted();
  const bool edge_weighted = graph.is_edge_weighted();

  StaticArray<EdgeID> nodes = graph.take_raw_nodes();
  StaticArray<NodeID> edges = graph.take_raw_edges();
  StaticArray<NodeWeight> node_weights = graph.take_raw_node_weights();
  StaticArray<EdgeWeight> edge_weights = graph.take_raw_edge_weights();

  if (edge_weighted) {
    StaticArray<std::pair<NodeID, EdgeWeight>> zipped(edges.size());
    tbb::parallel_for<EdgeID>(static_cast<EdgeID>(0), edges.size(), [&](const EdgeID e) {
      zipped[e] = {edges[e], edge_weights[e]};
    });

    tbb::parallel_for<NodeID>(0, nodes.size() - 1, [&](const NodeID u) {
      std::sort(
          zipped.begin() + nodes[u],
          zipped.begin() + nodes[u + 1],
          [](const auto &a, const auto &b) { return a.first < b.first; }
      );
    });

    tbb::parallel_for<EdgeID>(static_cast<EdgeID>(0), edges.size(), [&](const EdgeID e) {
      std::tie(edges[e], edge_weights[e]) = zipped[e];
    });
  } else {
    tbb::parallel_for<NodeID>(0, nodes.size() - 1, [&](const NodeID u) {
      std::sort(edges.begin() + nodes[u], edges.begin() + nodes[u + 1]);
    });
  }

  CSRGraph sorted_graph(
      std::move(nodes), std::move(edges), std::move(node_weights), std::move(edge_weights), sorted
  );
  sorted_graph.set_permutation(graph.take_raw_permutation());

  return sorted_graph;
}
} // namespace debug
} // namespace kaminpar::shm
