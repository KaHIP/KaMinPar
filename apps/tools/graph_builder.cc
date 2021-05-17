#include "graph_builder.h"

namespace kaminpar::tool {
NodeID GraphBuilder::new_node(const NodeWeight weight) {
  _nodes.push_back(_edges.size());
  _node_weights.push_back(weight);
  return _nodes.size() - 1;
}

EdgeID GraphBuilder::new_edge(const NodeID v, const EdgeID weight) {
  _edges.push_back(v);
  _edge_weights.push_back(weight);
  return _edges.size() - 1;
}
} // namespace kaminpar::tool