/*******************************************************************************
 * @file:   kaminpar.h
 * @author: Daniel Seemaier
 * @date:   21.09.21
 * @brief:  KaMinPar library.
 ******************************************************************************/
#pragma once

#include <memory>
#include <string_view>

namespace libkaminpar {
#ifdef KAMINPAR_64BIT_NODE_IDS
using NodeID = uint64_t;
#else  // KAMINPAR_64BIT_NODE_IDS
using NodeID     = uint32_t;
#endif // KAMINPAR_64BIT_NODE_IDS

#ifdef KAMINPAR_64BIT_EDGE_IDS
using EdgeID = uint64_t;
#else  // KAMINPAR_64BIT_EDGE_IDS
using EdgeID     = uint32_t;
#endif // KAMINPAR_64BIT_EDGE_IDS

#ifdef KAMINPAR_64BIT_WEIGHTS
using NodeWeight = int64_t;
using EdgeWeight = int64_t;
#else  // KAMINPAR_64BIT_WEIGHTS
using NodeWeight = int32_t;
using EdgeWeight = int32_t;
#endif // KAMINPAR_64BIT_WEIGHTS

using BlockID     = uint32_t;
using BlockWeight = NodeWeight;
using Degree      = EdgeID;

class Partitioner {
    friend class PartitionerBuilder;

public:
    Partitioner();
    ~Partitioner();

    Partitioner&               set_option(const std::string& name, const std::string& value);
    void                       set_quiet(bool quiet);
    void                       set_num_threads(int num_threads);
    void                       set_seed(int seed);
    void                       set_preset(const std::string& name);
    std::unique_ptr<BlockID[]> partition(BlockID k) const;
    std::unique_ptr<BlockID[]> partition(BlockID k, EdgeWeight& edge_cut) const;
    std::size_t                partition_size() const;

private:
    struct PartitionerPrivate* _pimpl = nullptr;

    bool _quiet       = true;
    int  _num_threads = 0;
    int  _seed        = 0;
};

class PartitionerBuilder {
public:
    PartitionerBuilder(const PartitionerBuilder&)                = delete;
    PartitionerBuilder& operator=(const PartitionerBuilder&)     = delete;
    PartitionerBuilder(PartitionerBuilder&&) noexcept            = default;
    PartitionerBuilder& operator=(PartitionerBuilder&&) noexcept = default;
    ~PartitionerBuilder();

    static PartitionerBuilder from_graph_file(const std::string& filename);
    static PartitionerBuilder from_adjacency_array(NodeID n, EdgeID* nodes, NodeID* edges);

    void        with_node_weights(NodeWeight* node_weights);
    void        with_edge_weights(EdgeWeight* edge_weights);
    Partitioner create();
    Partitioner rearrange_and_create();

private:
    PartitionerBuilder();

private:
    struct PartitionerBuilderPrivate* _pimpl;
};
} // namespace libkaminpar
