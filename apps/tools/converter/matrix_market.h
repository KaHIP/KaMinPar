/* Adapted from here: https://math.nist.gov/MatrixMarket/mmio/c/example_read.c */
#pragma once

#include "dynamic_graph_builder.h"
#include "edgelist_builder.h"
#include "graph_converter.h"
#include "mmio.h"
#include "utility/console_io.h"

#include <cstdlib>

namespace kaminpar::tool::converter {
class MatrixMarketReader : public GraphReader {
public:
  SimpleGraph read(const std::string &filename) override {
    std::vector<EdgeID> nodes;
    std::vector<NodeID> edges;
    std::vector<NodeWeight> node_weight;
    std::vector<EdgeWeight> edge_weight;
    bool read_edge_weights = true;

    FILE *f = fopen(filename.c_str(), "r");
    if (f == nullptr) { FATAL_PERROR << "Cannot open file " << filename; }

    MM_typecode matcode;
    if (mm_read_banner(f, &matcode) != 0) { FATAL_ERROR << "Could not process Matrix Market banner."; }
    if (!mm_is_valid(matcode)) { FATAL_ERROR << "Bad file format."; }
    if (mm_is_complex(matcode)) { read_edge_weights = false; WARNING << "Complex edge weights are not supported; won't read edge weights"; }

    int M, N, nz;
    if (mm_read_mtx_crd_size(f, &M, &N, &nz) != 0) { FATAL_ERROR << "Bad file format."; }
    ASSERT(M == N);

    const bool is_weighted = mm_is_real(matcode);

    LOG << "Reading " << M << "x" << N << " matrix with " << nz << " nonzeros";
    LOG << "If the matrix is asymmetric, missing edges are added until the resulting graph is undirected";
    cio::ProgressBar progress(nz, "Reading");

    EdgeListBuilder builder(M);
    for (int i = 0; i < nz; ++i) {
      int u, v;
      double val;
      if (is_weighted) {
        std::fscanf(f, "%d %d %lg\n", &u, &v, &val);
      } else {
        std::fscanf(f, "%d %d\n", &u, &v);
        val = 1.0;
      }
      builder.add_edge(u - 1, v - 1, (read_edge_weights) ? static_cast<EdgeWeight>(val) : 1);

      progress.step();
    }
    fclose(f);
    progress.stop();

    return builder.build();
  }

  [[nodiscard]] std::string description() const override { return "MatrixMarket format"; }
};
} // namespace kaminpar::tool::converter