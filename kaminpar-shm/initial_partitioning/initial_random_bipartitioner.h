/*******************************************************************************
 * Random initial bipartitioner that uses actual PRNG.
 *
 * @file:   initial_random_bipartitioner.h
 * @author: Daniel Seemaier
 * @date:   21.09.2021
 ******************************************************************************/
#pragma once

#include "kaminpar-shm/initial_partitioning/initial_flat_bipartitioner.h"

#include "kaminpar-common/random.h"

namespace kaminpar::shm {
class InitialRandomBipartitioner : public InitialFlatBipartitioner {
public:
  explicit InitialRandomBipartitioner(const InitialPoolPartitionerContext &pool_ctx);

protected:
  void fill_bipartition() final;

  Random &_rand = Random::instance();
};
} // namespace kaminpar::shm
