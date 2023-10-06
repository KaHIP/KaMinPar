#include <vector>

#include "kaminpar-shm/kaminpar.h"

namespace kaminpar::shm::data {
static std::vector<EdgeID> xadj = {
#include "data.graph.xadj"
};
static std::vector<NodeID> adjncy = {
#include "data.graph.adjncy"
};
} // namespace kaminpar::shm::data
