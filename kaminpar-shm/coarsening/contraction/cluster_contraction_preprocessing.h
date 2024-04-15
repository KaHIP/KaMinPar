/*******************************************************************************
 * Common preprocessing utilities for cluster contraction implementations.
 *
 * @file:   cluster_contraction_preprocessing.h
 * @author: Daniel Seemaier
 * @author: Daniel Salwasser
 * @date:   21.09.2021
 ******************************************************************************/
#pragma once

#include "kaminpar-shm/coarsening/contraction/cluster_contraction.h"
#include "kaminpar-shm/datastructures/graph.h"

#include "kaminpar-common/datastructures/static_array.h"

namespace kaminpar::shm::contraction {
template <template <typename> typename Mapping> class CoarseGraphImpl : public CoarseGraph {
public:
  CoarseGraphImpl(Graph graph, Mapping<NodeID> mapping)
      : _graph(std::move(graph)),
        _mapping(std::move(mapping)) {}

  const Graph &get() const final {
    return _graph;
  }

  Graph &get() final {
    return _graph;
  }

  void project(const StaticArray<BlockID> &array, StaticArray<BlockID> &onto) final {
    tbb::parallel_for<std::size_t>(0, onto.size(), [&](const std::size_t i) {
      onto[i] = array[_mapping[i]];
    });
  }

private:
  Graph _graph;
  Mapping<NodeID> _mapping;
};

template <template <typename> typename Mapping>
std::pair<NodeID, Mapping<NodeID>>
preprocess(const Graph &graph, StaticArray<NodeID> &clustering, MemoryContext &m_ctx);
} // namespace kaminpar::shm::contraction
