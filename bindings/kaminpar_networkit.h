/*******************************************************************************
 * NetworKit bindings for shared-memory KaMinPar.
 *
 * @file:   kaminpar_networkit.h
 * @author: Daniel Seemaier
 * @date:   09.12.2024
 ******************************************************************************/
#pragma once

#include <kaminpar-shm/kaminpar.h>
#include <networkit/graph/Graph.hpp>

namespace kaminpar {

class KaMinParNetworKit : public KaMinPar {
public:
  KaMinParNetworKit(int num_threads, const kaminpar::shm::Context &ctx);

  KaMinParNetworKit(const KaMinParNetworKit &) = delete;
  KaMinParNetworKit &operator=(const KaMinParNetworKit &) = delete;

  KaMinParNetworKit(KaMinParNetworKit &&) noexcept = default;
  KaMinParNetworKit &operator=(KaMinParNetworKit &&) noexcept = default;

  void copy_graph(const NetworKit::Graph &graph);
};

} // namespace kaminpar
