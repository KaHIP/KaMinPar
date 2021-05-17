#include "initial_partitioning/initial_refiner.h"

namespace kaminpar::ip {
template class InitialTwoWayFMRefiner<fm::MaxOverloadSelectionPolicy, fm::BalancedMinCutAcceptancePolicy,
                                      fm::SimpleStoppingPolicy>;
template class InitialTwoWayFMRefiner<fm::MaxOverloadSelectionPolicy, fm::BalancedMinCutAcceptancePolicy,
                                      fm::AdaptiveStoppingPolicy>;
} // namespace kaminpar::ip