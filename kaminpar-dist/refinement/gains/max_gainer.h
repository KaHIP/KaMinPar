/*******************************************************************************
 * Struct for the max gainer of a node.
 *
 * @file:   max_gainer.h
 * @author: Daniel Seemaier
 * @date:   03.07.2023
 ******************************************************************************/
#pragma once

#include "kaminpar-dist/dkaminpar.h"

namespace kaminpar::dist {

struct MaxGainer {
  EdgeWeight int_degree;
  EdgeWeight ext_degree;
  BlockID block;
  NodeWeight weight;

  [[nodiscard]] EdgeWeight absolute_gain() const {
    return ext_degree - int_degree;
  }

  [[nodiscard]] double relative_gain() const {
    if (ext_degree >= int_degree) {
      return 1.0 * absolute_gain() * weight;
    } else {
      return 1.0 * absolute_gain() / weight;
    }
  }
};

} // namespace kaminpar::dist
