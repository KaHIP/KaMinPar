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

#include "datastructure/graph.h"
#include "parallel.h"

namespace kaminpar::graph {
// TODO make local memory stuff more generic and move to its own file
struct Edge {
  NodeID target;
  EdgeWeight weight;
};

static constexpr std::size_t kChunkSize = (1 << 15);

using LocalEdgeMemoryChunk = scalable_vector<Edge>;

struct LocalEdgeMemory {
  LocalEdgeMemory() { current_chunk.reserve(kChunkSize); }

  scalable_vector<LocalEdgeMemoryChunk> chunks;
  LocalEdgeMemoryChunk current_chunk;

  std::size_t get_current_position() const { return chunks.size() * kChunkSize + current_chunk.size(); }

  void push(const NodeID c_v, const EdgeWeight weight) {
    if (current_chunk.size() == kChunkSize) { flush(); }
    current_chunk.emplace_back(c_v, weight);
  }

  const auto &get(const std::size_t position) const { return chunks[position / kChunkSize][position % kChunkSize]; }
  auto &get(const std::size_t position) {
    ASSERT(position / kChunkSize < chunks.size()) << V(position) << V(kChunkSize) << V(chunks.size());
    ASSERT(position % kChunkSize < chunks[position / kChunkSize].size())
    << V(position) << V(kChunkSize) << V(chunks[position / kChunkSize].size());
    return chunks[position / kChunkSize][position % kChunkSize];
  }

  void flush() {
    chunks.push_back(std::move(current_chunk));
    current_chunk.clear();
    current_chunk.reserve(kChunkSize);
  }
};

struct BufferNode {
  NodeID c_u;
  std::size_t position;
  LocalEdgeMemory *chunks;
};

struct ContractionMemoryContext {
  scalable_vector<NodeID> buckets;
  scalable_vector<parallel::IntegralAtomicWrapper<NodeID>> buckets_index;
  scalable_vector<parallel::IntegralAtomicWrapper<NodeID>> leader_mapping;
  scalable_vector<BufferNode> all_buffered_nodes;
};

struct ContractionResult {
  Graph graph;
  scalable_vector<NodeID> mapping;
  ContractionMemoryContext m_ctx;
};

ContractionResult contract(const Graph &r, const scalable_vector<NodeID> &clustering,
                           ContractionMemoryContext m_ctx = {});
} // namespace kaminpar::graph