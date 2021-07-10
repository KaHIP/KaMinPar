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
#include "dkaminpar/datastructure/distributed_graph.h"

namespace dkaminpar::graph {
struct Edge {
  DNodeID target;
  DEdgeWeight weight;
};

static constexpr std::size_t kChunkSize = (1 << 15);

using LocalEdgeMemoryChunk = scalable_vector<Edge>;

struct LocalEdgeMemory {
  LocalEdgeMemory() { current_chunk.reserve(kChunkSize); }

  scalable_vector<LocalEdgeMemoryChunk> chunks;
  LocalEdgeMemoryChunk current_chunk;

  [[nodiscard]] std::size_t get_current_position() const { return chunks.size() * kChunkSize + current_chunk.size(); }

  void push(const DNodeID c_v, const DEdgeWeight weight) {
    if (current_chunk.size() == kChunkSize) { flush(); }
    current_chunk.emplace_back(c_v, weight);
  }

  [[nodiscard]] const auto &get(const std::size_t position) const {
    return chunks[position / kChunkSize][position % kChunkSize];
  }

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
  DNodeID c_u;
  std::size_t position;
  LocalEdgeMemory *chunks;
};

struct ContractionMemoryContext {
  scalable_vector<DNodeID> buckets;
  scalable_vector<shm::parallel::IntegralAtomicWrapper<DNodeID>> buckets_index;
  scalable_vector<shm::parallel::IntegralAtomicWrapper<DNodeID>> leader_mapping;
  scalable_vector<BufferNode> all_buffered_nodes;
};

struct ContractionResult {
  DistributedGraph graph;
  scalable_vector<DNodeID> mapping;
  ContractionMemoryContext m_ctx;
};

ContractionResult contract_locally(const DistributedGraph &graph, const scalable_vector<DNodeID> &clustering,
                           ContractionMemoryContext m_ctx = {});
} // namespace dkaminpar::graph