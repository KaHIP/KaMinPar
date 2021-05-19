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

#include <memory>
#include <string_view>

namespace libkaminpar {
struct PartitionerPrivate;


using NodeID = uint32_t;
#ifdef KAMINPAR_64BIT_EDGE_IDS
using EdgeID = uint64_t;
#else  // KAMINPAR_64BIT_EDGE_IDS
using EdgeID = uint32_t;
#endif // KAMINPAR_64BIT_EDGE_IDS
using BlockID = uint32_t;
using NodeWeight = int32_t;
using EdgeWeight = int32_t;
using BlockWeight = NodeWeight;

class Partitioner {
  friend class PartitionerBuilder;

public:
  Partitioner();
  ~Partitioner();

  void set_option(const std::string &name, const std::string &value);
  std::unique_ptr<BlockID[]> partition(BlockID k);
  std::size_t partition_size() const;

private:
  PartitionerPrivate *_pimpl;
};

struct PartitionerBuilderPrivate;

class PartitionerBuilder {
public:
  PartitionerBuilder(const PartitionerBuilder &) = delete;
  PartitionerBuilder &operator=(const PartitionerBuilder &) = delete;
  PartitionerBuilder(PartitionerBuilder &&) noexcept = default;
  PartitionerBuilder &operator=(PartitionerBuilder &&) noexcept = default;
  ~PartitionerBuilder();

  static PartitionerBuilder from_graph_file(const std::string &filename);
  static PartitionerBuilder from_adjacency_array(NodeID n, EdgeID *nodes, NodeID *edges);

  void with_node_weights(NodeWeight *node_weights);
  void with_edge_weights(EdgeWeight *edge_weights);
  Partitioner create();
  Partitioner rearrange_and_create();

private:
  PartitionerBuilder();

private:
  PartitionerBuilderPrivate *_pimpl;
};
} // namespace libkaminpar