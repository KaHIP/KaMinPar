/*******************************************************************************
 * @file:   kaminpar.h
 *
 * @author: Daniel Seemaier
 * @date:   21.09.21
 * @brief:  KaMinPar library.
 ******************************************************************************/
#pragma once

#include <memory>
#include <string_view>

#include "kaminpar_export.h"

namespace libkaminpar {
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

class KAMINPAR_EXPORT Partitioner {
  friend class PartitionerBuilder;

public:
  Partitioner();
  ~Partitioner();

  Partitioner &set_option(const std::string &name, const std::string &value);
  std::unique_ptr<BlockID[]> partition(BlockID k) const;
  std::unique_ptr<BlockID[]> partition(BlockID k, EdgeWeight &edge_cut) const;
  std::size_t partition_size() const;

private:
  struct PartitionerPrivate *_pimpl;
};

class KAMINPAR_EXPORT PartitionerBuilder {
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
  struct PartitionerBuilderPrivate *_pimpl;
};
} // namespace libkaminpar