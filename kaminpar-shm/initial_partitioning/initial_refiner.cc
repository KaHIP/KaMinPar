/*******************************************************************************
 * @file:   initial_refiner.cc
 * @author: Daniel Seemaier
 * @date:   21.09.2021
 * @brief:  Sequential local improvement graphutils used to improve an initial
 * partition.
 ******************************************************************************/
#include "kaminpar-shm/initial_partitioning/initial_refiner.h"

namespace kaminpar::shm::ip {
template class InitialTwoWayFMRefiner<
    fm::MaxOverloadSelectionPolicy,
    fm::BalancedMinCutAcceptancePolicy,
    fm::SimpleStoppingPolicy>;
template class InitialTwoWayFMRefiner<
    fm::MaxOverloadSelectionPolicy,
    fm::BalancedMinCutAcceptancePolicy,
    fm::AdaptiveStoppingPolicy>;
} // namespace kaminpar::shm::ip
