#include "kaminpar-shm/refinement/lhop/lhop_refiner.h"

#include "kaminpar-shm/datastructures/graph.h"

#include <vector>

namespace kaminpar::shm {

LHopRefiner::LHopRefiner(const Context &ctx) : _ctx(ctx) {}

std::string LHopRefiner::name() const {
  return "L-hop";
}

void LHopRefiner::initialize(const PartitionedGraph &p_graph) {
  _graph = &concretize<CSRGraph>(p_graph.graph());
}



void LHopRefiner::initializeLHopModel(PartitionedGraph &p_graph, std::vector<std::vector<LHopTable>> &lhopModel) {
  //TODO opt: better way to find partition?
  for(BlockID block = 0; block < p_graph.k(); ++block) {
    //find Partition
    std::vector<NodeID> partition; 
    for (NodeID u = 0; u < _graph->n(); ++u) {
      if(p_graph.block(u) == block) {
        partition.push_back(u);
      }
    }
    //run lhopPathFinder for partition
    lhopPathFinder(p_graph, lhopModel, partition);
  }
}


//TODO opt: make this parallel?
void LHopRefiner::lhopPathFinder(PartitionedGraph &p_graph, std::vector<std::vector<LHopTable>> &lhopModel, std::vector<NodeID> &startgroup) {
  BlockID identifier = p_graph.block(startgroup[0]);
  //TODO opt: activeGroup to vector<bool>?
  std::vector<NodeID> activeGroup;
  std::vector<bool> activeGroupVisited(_graph->n(), false);
  //TODO opt: calculate parSum?
  std::vector<std::vector<unsigned long>> parSum(_graph->n(), {0,0});
  //Range == 1
  for (auto activeNode = startgroup.begin(); activeNode != startgroup.end(); ++activeNode) {
    // Visit activeNode's neighbors:
    parSum[*activeNode][0] = 1;
    _graph->neighbors(*activeNode, [&](const EdgeID e, const NodeID neighbor, const EdgeWeight ew) {
      activeGroup.push_back(neighbor);
      if(!lhopModel[neighbor].empty() && lhopModel[neighbor].back().block == identifier) {
        lhopModel[neighbor].back().pathLength[0]++;
      } else {
        LHopTable newEntry(identifier, l);
        newEntry.pathLength[0]++;
        lhopModel[neighbor].push_back(newEntry);
      }
      ((void)e);
      ((void)ew);
    });
  }
  //Range > 1
  for(unsigned int len = 1; len < l; ++len) {
    std::vector<NodeID> nextActiveGroup;
    for (auto activeNode = activeGroup.begin(); activeNode != activeGroup.end(); ++activeNode) {
      if(activeGroupVisited[*activeNode]) {
        continue;
      }
      activeGroupVisited[*activeNode] = true;
      parSum[*activeNode][len % 2] += lhopModel[*activeNode].back().pathLength[(len-1)];
      // Visit activeNode's neighbors:
      _graph->neighbors(*activeNode, [&](const EdgeID e, const NodeID neighbor, const EdgeWeight ew) {
        nextActiveGroup.push_back(neighbor);
        if(!lhopModel[neighbor].empty() && lhopModel[neighbor].back().block == identifier) {
          lhopModel[neighbor].back().pathLength[len] += parSum[*activeNode][len % 2] - parSum[neighbor][(len+1) % 2];
        } else {
          LHopTable newEntry(identifier, l);
          newEntry.pathLength[len] += parSum[*activeNode][len % 2] - parSum[neighbor][(len+1) % 2];
          lhopModel[neighbor].push_back(newEntry);
        }
        ((void)e);
        ((void)ew);
      });
    }
    activeGroup = nextActiveGroup;
    nextActiveGroup = {};
    activeGroupVisited.assign(_graph->n(), false);
  }
}

void LHopRefiner::calculateGains(PartitionedGraph &p_graph, std::vector<std::vector<LHopTable>> &lhopModel, std::vector<LHopNodeGain> &nodeGains, 
                                  std::vector<LHopPartitionGain> &partitionGains) {
  //TODO opt: make it parallel
  for (NodeID node = 0; node < _graph->n(); ++node) { 
    BlockID srcBlock = p_graph.block(node);
    unsigned long gainForStay = 0;
    //TODO opt: iterate only once to generate gains
    for(LHopTable& table : lhopModel[node]) {
      if(table.block == srcBlock) {
        gainForStay = tableToGain(table);
      }
    }
    for(LHopTable& table : lhopModel[node]) {
      if(gainForStay < tableToGain(table)){
        //NodeGains
        nodeGains.push_back(LHopNodeGain(node, srcBlock, table.block, (tableToGain(table) - gainForStay)));

        //PartitionGains
        bool isIn = false;
        //TODO opt: is this the best data structure for partitionGains
        for(LHopPartitionGain& partition : partitionGains) {
          if(partition.src == srcBlock && partition.dest == table.block) {
            partition.gain += (tableToGain(table) - gainForStay);
            isIn = true;
            break;
          }
        }
        if(!isIn) {
          partitionGains.push_back(LHopPartitionGain(srcBlock, table.block, (tableToGain(table) - gainForStay)));
        }
      }
    }
  }
  //TODO opt: is sorting nessesary?
  std::sort(nodeGains.begin(), nodeGains.end());
  std::sort(partitionGains.begin(), partitionGains.end());

  /*for(LHopNodeGain& node : nodeGains) {
    LOG << "Gain: " << node.node << " :: " << node.src << " :: " << node.dest << " :: " << node.gain;
  }

  for(LHopPartitionGain& partition : partitionGains) {
    LOG << "PartitionGain: " << partition.src << " :: " << partition.dest << " :: " << partition.gain;
  }*/
}

unsigned long LHopRefiner::tableToGain(LHopTable &gain) {
  unsigned long result = 0;
  for(unsigned int len = 1; len < l; ++len) {
    result += gain.pathLength[len] * lweights[len];
  }
  return result;
}

bool LHopRefiner::moveAndUpdate(PartitionedGraph &p_graph, const PartitionContext &p_ctx, std::vector<std::vector<LHopTable>> &lhopModel, 
                                std::vector<LHopNodeGain> &nodeGains, BlockID src, BlockID dest) {
  //Move nodes
  std::vector<NodeID> movedNodes;
  for(LHopNodeGain& node : nodeGains) {
    if(node.src == src && node.dest == dest) {
      if(p_graph.move(node.node, node.src, node.dest, p_ctx.max_block_weight(node.dest))) {
        movedNodes.push_back(node.node);
      } else { break; }
    }
  }

  if(movedNodes.empty()) {
    return false;
  }
  //update Model
  //TODO opt: build smaller data structure
  std::vector<std::vector<LHopTable>> updateLHopModel(p_graph.n());
  lhopPathFinder(p_graph, updateLHopModel, movedNodes);

  //merge updateModel into lHopModel
  for (NodeID node = 0; node < _graph->n(); ++node) {
    std::vector<LHopTable>& nodeUpdate = updateLHopModel[node];
    if(nodeUpdate.empty()) {
      continue;
    }
    for(LHopTable& lHopModelEntry : lhopModel[node]) {
      if(lHopModelEntry.block == src) {
        //subtract
        subtractLHopTable(lHopModelEntry, nodeUpdate[0]);
      } else if (lHopModelEntry.block == dest) {
        //add
        addLHopTable(lHopModelEntry, nodeUpdate[0]);
      }
    }
  }
  LOG << "Moved and updated: " << src << " -> " << dest << " : " << movedNodes.size();
  return true;
}

void LHopRefiner::subtractLHopTable(LHopTable &minuend, LHopTable &subtrahend) {
  for(unsigned int len = 1; len < l; ++len) {
    minuend.pathLength[len] -= subtrahend.pathLength[len];
  }
}

void LHopRefiner::addLHopTable(LHopTable &result, LHopTable &addends) {
  for(unsigned int len = 1; len < l; ++len) {
    result.pathLength[len] += addends.pathLength[len];
  }
}

bool LHopRefiner::refine(PartitionedGraph &p_graph, const PartitionContext &p_ctx) {
  // Do nothing on coarse levels:
  if (p_graph.graph().level() > 0) {
    return false;
  }
  LOG << "Build LHop Model";
  std::vector<std::vector<LHopTable>> lhopModel(p_graph.n());
  initializeLHopModel(p_graph, lhopModel);

  std::vector<LHopNodeGain> nodeGains;
  std::vector<LHopPartitionGain> partitionGains;
  
  LOG << "START BATCH";
  bool movedANode = false;
  bool moving = true;
  while(moving) {
    LOG << "Generate Gains";
    nodeGains.clear();
    partitionGains.clear();
    calculateGains(p_graph, lhopModel, nodeGains, partitionGains);
    if(partitionGains.empty()) {
      break;
    }
    LOG << "Move and update";
    for(LHopPartitionGain& moveBlock : partitionGains) {
      if(moveAndUpdate(p_graph, p_ctx, lhopModel, nodeGains, moveBlock.src, moveBlock.dest)) {
        movedANode = true;
        moving = true;
        break;
      } else {
        LOG << "Full Partition: " << moveBlock.dest;
        moving = false;
      }
    }
  }
  LOG << "END BATCH";
  /*
  // E.g.,
  // Move vertex 0 from block 1 to block 0, but only if the resulting weight of block 0 does not
  // exceed p_ctx.max_block_weight(0)
  const bool success = p_graph.move(0, 1, 0, p_ctx.max_block_weight(0));
  ((void)success);

  // Iterate over vertices (in parallel):
  _graph->pfor_nodes([&](const NodeID u) {
    // ... do stuff with u ...

    // Visit u's neighbors:
    _graph->neighbors(u, [&](const EdgeID e, const NodeID v, const EdgeWeight ew) {
      // Edge e, with weight ew, connects u to neighbor v
      ((void)e);
      ((void)v);
      ((void)ew);
    });
  });

  // Iterate over vertices sequentially:
  for (NodeID u = 0; u < _graph->n(); ++u) {
    // ... do stuff with u ...
  }

  // Indicate whether refinement improved the partition:
  // (mostly ignored)*/
  return movedANode;
}

} // namespace kaminpar::shm
