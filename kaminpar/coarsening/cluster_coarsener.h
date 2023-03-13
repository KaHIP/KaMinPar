/*******************************************************************************
 * @file:   coarsener.h
 * @author: Daniel Seemaier
 * @date:   29.09.2021
 * @brief:  Coarsener that uses a clustering graphutils to coarsen the graph.
 ******************************************************************************/
#pragma once

#include "kaminpar/coarsening/i_coarsener.h"
#include "kaminpar/context.h"
#include "kaminpar/datastructures/graph.h"
#include "kaminpar/definitions.h"
#include "kaminpar/graphutils/graph_contraction.h"

namespace kaminpar::shm {
class ClusteringCoarsener : public ICoarsener {
public:
  ClusteringCoarsener(std::unique_ptr<IClustering> clustering_algorithm,
                      const Graph &input_graph, const CoarseningContext &c_ctx)
      : _input_graph{input_graph}, _current_graph{&input_graph},
        _clustering_algorithm{std::move(clustering_algorithm)}, _c_ctx{c_ctx} {}

  ClusteringCoarsener(const ClusteringCoarsener &) = delete;
  ClusteringCoarsener &operator=(const ClusteringCoarsener) = delete;
  ClusteringCoarsener(ClusteringCoarsener &&) = delete;
  ClusteringCoarsener &operator=(ClusteringCoarsener &&) = delete;

  std::pair<const Graph *, bool>
  compute_coarse_graph(NodeWeight max_cluster_weight, NodeID to_size) final;
  PartitionedGraph uncoarsen(PartitionedGraph &&p_graph) final;

  [[nodiscard]] const Graph *coarsest_graph() const final {
    return _current_graph;
  }
  [[nodiscard]] std::size_t size() const final { return _hierarchy.size(); }
  void initialize(const Graph *) final {}
  [[nodiscard]] const CoarseningContext &context() const { return _c_ctx; }

private:
  const Graph &_input_graph;
  const Graph *_current_graph;
  std::vector<Graph> _hierarchy;
  std::vector<scalable_vector<NodeID>> _mapping;

  std::unique_ptr<IClustering> _clustering_algorithm;

  const CoarseningContext &_c_ctx;
  graph::contraction::MemoryContext _contraction_m_ctx{};
};
} // namespace kaminpar::shm
