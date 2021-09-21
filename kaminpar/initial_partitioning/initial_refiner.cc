/*******************************************************************************
 * @file:   initial_refiner.cc
 *
 * @author: Daniel Seemaier
 * @date:   21.09.21
 * @brief:  Sequential local improvement algorithm used to improve an initial
 * partition.
 ******************************************************************************/
#include "kaminpar/initial_partitioning/initial_refiner.h"

namespace kaminpar::ip {
template class InitialTwoWayFMRefiner<fm::MaxOverloadSelectionPolicy, fm::BalancedMinCutAcceptancePolicy,
                                      fm::SimpleStoppingPolicy>;
template class InitialTwoWayFMRefiner<fm::MaxOverloadSelectionPolicy, fm::BalancedMinCutAcceptancePolicy,
                                      fm::AdaptiveStoppingPolicy>;
} // namespace kaminpar::ip