#pragma once

#include "kaminpar-shm/datastructures/csr_graph.h"
#include "kaminpar-shm/datastructures/partitioned_graph.h"
#include "kaminpar-shm/refinement/refiner.h"

namespace kaminpar::shm {

struct LHopTable {
  BlockID block;
  std::vector<unsigned long> pathLength;

  LHopTable(BlockID b, const std::size_t size) : block(b), pathLength(size) {}
};

//TODO opt: remove BlockID src
struct LHopNodeGain {
  NodeID node;
  BlockID src;
  BlockID dest;
  unsigned long gain;

  LHopNodeGain(NodeID n, BlockID s, BlockID d, unsigned long value) : node(n), src(s), dest(d), gain(value) {}

  bool operator<(const LHopNodeGain& rhs) const { return gain > rhs.gain; }
};

struct LHopPartitionGain {
  BlockID src;
  BlockID dest;
  unsigned long gain;

  LHopPartitionGain(BlockID s, BlockID d, unsigned long value) : src(s), dest(d), gain(value) {}

  bool operator<(const LHopPartitionGain& rhs) const { return gain > rhs.gain; }
};

class LHopRefiner : public Refiner {
public:
  LHopRefiner(const Context &ctx);

  LHopRefiner(const LHopRefiner &) = delete;
  LHopRefiner &operator=(const LHopRefiner &) = delete;

  LHopRefiner(LHopRefiner &&) noexcept = default;
  LHopRefiner &operator=(LHopRefiner &&) noexcept = delete;

  [[nodiscard]] std::string name() const override;

  void initialize(const PartitionedGraph &p_graph) override;

  bool refine(PartitionedGraph &p_graph, const PartitionContext &p_ctx) override;

private:
  const Context &_ctx;

  const CSRGraph *_graph = nullptr;

  //is lHop 2-3 better than edge-cut?
  const std::size_t l = 3; // SET L used for LHop -> TODO replace

  const std::vector<int> lweights = {100, 10, 1};

  void initializeLHopModel(PartitionedGraph &p_graph, std::vector<std::vector<LHopTable>> &lhopModel);

  void lhopPathFinder(PartitionedGraph &p_graph, std::vector<std::vector<LHopTable>> &lhopModel, std::vector<NodeID> &startgroup);

  void calculateGains(PartitionedGraph &p_graph, std::vector<std::vector<LHopTable>> &lhopModel, std::vector<LHopNodeGain> &nodeGains, 
                      std::vector<LHopPartitionGain> &partitionGains);

  unsigned long tableToGain(LHopTable &gain);

  bool moveAndUpdate(PartitionedGraph &p_graph, const PartitionContext &p_ctx, std::vector<std::vector<LHopTable>> &lhopModel, 
                      std::vector<LHopNodeGain> &nodeGains, BlockID src, BlockID dest);

  void subtractLHopTable(LHopTable &minuend, LHopTable &subtrahend);

  void addLHopTable(LHopTable &result, LHopTable &addends);
};

} // namespace kaminpar::shm
