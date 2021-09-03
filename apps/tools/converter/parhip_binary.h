/*******************************************************************************
 * This file is part of KaMinPar.
 *
 * Copyright (C) 2021 Daniel Seemaier <daniel.seemaier@kit.edu>
 *
 * KaMinPar is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * KaMinPar is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with KaMinPar.  If not, see <http://www.gnu.org/licenses/>.
 *
******************************************************************************/
#pragma once

/*******************************************************************************
 * This file is part of KaMinPar.
 *
 * Copyright (C) 2021 Daniel Seemaier <daniel.seemaier@kit.edu>
 *
 * KaMinPar is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * KaMinPar is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with KaMinPar.  If not, see <http://www.gnu.org/licenses/>.
 *
******************************************************************************/
#pragma once

#include <fstream>

#define HAS_NODE_WEIGHTS(version) (((version) &0b10) == 1)
#define HAS_EDGE_WEIGHTS(version) (((version) &0b01) == 1)

namespace kaminpar::tool::converter {
using parhip_ulong_t = unsigned long long;

class ParhipBinaryWriter : public GraphWriter {
public:
  void write(const std::string &filename, SimpleGraph graph, const std::string &) override {
    std::FILE *fd = std::fopen(filename.c_str(), "w");
    if (!fd) { FATAL_PERROR << "cannot write to " << filename; }

    const parhip_ulong_t version = 3;
    const parhip_ulong_t n = graph.n();
    const parhip_ulong_t m = graph.m();

    ALWAYS_ASSERT(std::fwrite(&version, sizeof(parhip_ulong_t), 1, fd) == 1);
    ALWAYS_ASSERT(std::fwrite(&n, sizeof(parhip_ulong_t), 1, fd) == 1);
    ALWAYS_ASSERT(std::fwrite(&m, sizeof(parhip_ulong_t), 1, fd) == 1);

    {
      parhip_ulong_t offset = (4 + graph.n()) * sizeof(parhip_ulong_t);

      std::vector<parhip_ulong_t> nodes(graph.n() + 1);
      for (NodeID u = 0; u < n; ++u) {
        nodes[u] = offset;
        offset += graph.degree(u) * sizeof(parhip_ulong_t);
      }
      nodes[n] = offset;

      ALWAYS_ASSERT(std::fwrite(nodes.data(), sizeof(parhip_ulong_t), n + 1, fd) == n + 1);
    }
    {
      std::vector<parhip_ulong_t> edges(graph.m());
      std::ranges::copy(graph.edges, edges.begin());
      ALWAYS_ASSERT(std::fwrite(edges.data(), sizeof(parhip_ulong_t), m, fd) == m);
    }

    std::fclose(fd);
  }

  [[nodiscard]] std::string description() const override { return "METIS graph format"; }
};
} // namespace kaminpar::tool::converter