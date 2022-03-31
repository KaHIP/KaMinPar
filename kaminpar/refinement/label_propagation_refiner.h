/*******************************************************************************
 * @file:   parallel_label_propagation_refiner.h
 *
 * @author: Daniel Seemaier
 * @date:   21.09.21
 * @brief:
 ******************************************************************************/
#pragma once

#include "kaminpar/refinement/i_refiner.h"

namespace kaminpar {
class LabelPropagationRefiner : public IRefiner {
public:
    LabelPropagationRefiner(const Context& ctx);

    ~LabelPropagationRefiner();

    void initialize(const Graph& graph) override;

    bool refine(PartitionedGraph& p_graph, const PartitionContext& p_ctx) override;

    [[nodiscard]] EdgeWeight expected_total_gain() const override;

private:
    class LabelPropagationRefinerImpl* _impl;
};
} // namespace kaminpar
