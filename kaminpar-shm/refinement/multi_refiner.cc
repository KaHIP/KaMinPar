/*******************************************************************************
 * Pseudo-refiner that runs multiple refiners in sequence.
 *
 * @file:   multi_refiner.cc
 * @author: Daniel Seemaier
 * @date:   14.03.2023
 ******************************************************************************/
#include "kaminpar-shm/refinement/multi_refiner.h"

#include "kaminpar-shm/kaminpar.h"
#include "kaminpar-shm/metrics.h"
#include "kaminpar-shm/refinement/refiner.h"

#include "kaminpar-common/logger.h"

namespace kaminpar::shm {

MultiRefiner::MultiRefiner(
    std::unordered_map<RefinementAlgorithm, std::unique_ptr<Refiner>> refiners,
    std::vector<RefinementAlgorithm> order
)
    : _refiners(std::move(refiners)),
      _order(std::move(order)) {}

std::string MultiRefiner::name() const {
  return "Multi Refiner";
}

void MultiRefiner::initialize([[maybe_unused]] const PartitionedGraph &p_graph) {}

bool MultiRefiner::refine(PartitionedGraph &p_graph, const PartitionContext &p_ctx) {
  bool found_improvement = false;

  double imbalance_before = _output_level >= OutputLevel::DEBUG ? metrics::imbalance(p_graph) : 0.0;
  EdgeWeight cut_before = _output_level >= OutputLevel::DEBUG ? metrics::edge_cut(p_graph) : 0;
  bool feasible_before =
      _output_level >= OutputLevel::DEBUG ? metrics::is_feasible(p_graph, p_ctx) : true;

  for (std::size_t i = 0; i < _order.size(); ++i) {
    const RefinementAlgorithm algorithm = _order[i];

    if (_output_level >= OutputLevel::INFO) {
      LLOG << _output_prefix << i + 1 << ". " << _refiners[algorithm]->name();
    }

    _refiners[algorithm]->initialize(p_graph);

    const bool current_refiner_found_improvement = _refiners[algorithm]->refine(p_graph, p_ctx);
    found_improvement |= current_refiner_found_improvement;

    if (_output_level >= OutputLevel::DEBUG) {
      const double imbalance_after = metrics::imbalance(p_graph);
      const EdgeWeight cut_after = metrics::edge_cut(p_graph);
      const bool feasible_after = metrics::is_feasible(p_graph, p_ctx);

      const bool unchanged = cut_before == cut_after && imbalance_before == imbalance_after;

      LOG << ": Cut[" << cut_before << " -> " << cut_after << "], "
          << "Imbalance[" << imbalance_before << " -> " << imbalance_after << "], Feasible["
          << feasible_before << " -> " << feasible_after << "] "
          << (unchanged ? "(no effect)" : "");

      imbalance_before = imbalance_after;
      cut_before = cut_after;
      feasible_before = feasible_after;
    } else if (_output_level >= OutputLevel::INFO) {
      if (!current_refiner_found_improvement) {
        LOG << " (no effect)";
      } else {
        LOG;
      }
    }
  }

  return found_improvement;
}

void MultiRefiner::set_communities(std::span<const NodeID> communities) {
  for (auto &[_, refiner] : _refiners) {
    refiner->set_communities(communities);
  }
}

} // namespace kaminpar::shm
