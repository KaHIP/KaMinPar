/*******************************************************************************
 * Random initial bipartitioner that uses actual PRNG.
 *
 * @file:   random_bipartitioner.h
 * @author: Daniel Seemaier
 * @date:   21.09.2021
 ******************************************************************************/
#pragma once

#include "kaminpar-shm/initial_partitioning/bipartitioner.h"

#include "kaminpar-common/random.h"

namespace kaminpar::shm::ip {
class RandomBipartitioner : public Bipartitioner {
public:
  explicit RandomBipartitioner(const InitialPoolPartitionerContext &pool_ctx);

protected:
  void fill_bipartition() final;

  Random &_rand = Random::instance();
};
} // namespace kaminpar::shm::ip
