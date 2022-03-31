/*******************************************************************************
 * @file:   greedy_graph_growing_bipartitioner.h
 *
 * @author: Daniel Seemaier
 * @date:   21.09.21
 * @brief:  Initial partitioner based on greedy graph growing.
 ******************************************************************************/
#pragma once

#include "kaminpar/datastructure/binary_heap.h"
#include "kaminpar/datastructure/marker.h"
#include "kaminpar/initial_partitioning/i_bipartitioner.h"
#include "kaminpar/utils/random.h"

namespace kaminpar {
class GreedyGraphGrowingBipartitioner : public Bipartitioner {
public:
    struct MemoryContext {
        BinaryMinHeap<Gain> queue{0};
        Marker<>            marker{0};

        std::size_t memory_in_kb() const {
            return queue.memory_in_kb() + marker.memory_in_kb();
        }
    };

    GreedyGraphGrowingBipartitioner(
        const Graph& graph, const PartitionContext& p_ctx, const InitialPartitioningContext& i_ctx,
        MemoryContext& m_ctx)
        : Bipartitioner(graph, p_ctx, i_ctx),
          _queue{m_ctx.queue},
          _marker{m_ctx.marker} {
        if (_queue.capacity() < _graph.n()) {
            _queue.resize(_graph.n());
        }
        if (_marker.size() < _graph.n()) {
            _marker.resize(_graph.n());
        }
    }

protected:
    void bipartition_impl() override;

private:
    [[nodiscard]] Gain compute_negative_gain(NodeID u) const;

    BinaryMinHeap<Gain>& _queue;
    Marker<>&            _marker;
};
} // namespace kaminpar