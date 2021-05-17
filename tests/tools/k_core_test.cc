#include "tests/tests.h"
#include "tools/graph_tools.h"

#include <ranges>

using ::testing::Each;
using ::testing::ElementsAre;
using ::testing::Eq;
using ::testing::IsEmpty;
using ::testing::Ne;

namespace kamipar {
TEST(KCoreTest, ComputesCorrectKCores) {
  Graph graph({0, 2, 5, 11, 15, 19, 23, 27, 30, 32}, {
                                                         1, 2,             //
                                                         0, 2, 5,          //
                                                         0, 1, 5, 6, 3, 4, //
                                                         4, 2, 8, 7,       //
                                                         2, 6, 3, 7,       //
                                                         1, 8, 6, 2,       //
                                                         2, 5, 7, 4,       //
                                                         4, 3, 6,          //
                                                         5, 3              //
                                                     });

  std::vector<std::vector<EdgeWeight>> k_cores{};
  for (EdgeWeight k = 1;; ++k) {
    if (k_cores.empty()) {
      k_cores.push_back(compute_k_core(graph, k));
    } else {
      k_cores.push_back(compute_k_core(graph, k, k_cores.back()));
    }
    if (std::ranges::all_of(k_cores.back(), [](const auto arg) { return arg == 0; })) { break; }
  }

  // degeneracy should be 3 -> 4 graphs with the last one being empty
  EXPECT_THAT(k_cores.size(), 4);

  // 1-core, 2-core are entire graph
  EXPECT_THAT(k_cores[0], Each(Ne(0)));
  EXPECT_THAT(k_cores[1], Each(Ne(0)));

  // 3-core misses some nodes
  EXPECT_THAT(k_cores[2], ElementsAre(Eq(0), Eq(0), Ne(0), Ne(0), Ne(0), Eq(0), Ne(0), Ne(0), Eq(0)));

  // 4-core is empty
  EXPECT_THAT(k_cores[3], Each(Eq(0)));
}

TEST(KCoreTest, WorksOnEmptyGraph) {
  Graph graph({0}, {});

  std::vector<EdgeWeight> core1{compute_k_core(graph, 1)};
  EXPECT_THAT(core1, IsEmpty());
}

TEST(KCoreTest, WorksWithIsolatedNodes) {
  // *--*  *
  Graph graph({0, 1, 2, 2}, {1, 0});

  // core 1
  std::vector<EdgeWeight> core1{compute_k_core(graph, 1)};
  EXPECT_THAT(core1, ElementsAre(Ne(0), Ne(0), Eq(0)));
}


TEST(KCoreTest, WorksWithCliqueAndLooselyConnectedNodes) {
  // clang-format off
  // K5 with some loosely connected nodes
  Graph graph({0, 5, 10, 14, 19, 24, 25, 26, 27, 29, 30}, {
      7, 4, 1, 2, 3,
      8, 3, 4, 2, 0,
      0, 1, 3, 4,
      0, 2, 4, 1, 5,
      3, 2, 1, 0, 6,
      3,
      4,
      0,
      1, 9,
      8
  });
  // clang-format on

  std::vector<EdgeWeight> core1{compute_k_core(graph, 1)};
  EXPECT_THAT(core1, ElementsAre(Ne(0), Ne(0), Ne(0), Ne(0), Ne(0), Ne(0), Ne(0), Ne(0), Ne(0), Ne(0)));

  std::vector<EdgeWeight> core2{compute_k_core(graph, 2, core1)};
  EXPECT_THAT(core2, ElementsAre(Ne(0), Ne(0), Ne(0), Ne(0), Ne(0), Eq(0), Eq(0), Eq(0), Eq(0), Eq(0)));

  std::vector<EdgeWeight> core3{compute_k_core(graph, 3, core2)};
  EXPECT_THAT(core3, ElementsAre(Ne(0), Ne(0), Ne(0), Ne(0), Ne(0), Eq(0), Eq(0), Eq(0), Eq(0), Eq(0)));

  std::vector<EdgeWeight> core4{compute_k_core(graph, 4, core3)};
  EXPECT_THAT(core4, ElementsAre(Ne(0), Ne(0), Ne(0), Ne(0), Ne(0), Eq(0), Eq(0), Eq(0), Eq(0), Eq(0)));

  std::vector<EdgeWeight> core5{compute_k_core(graph, 5, core4)};
  EXPECT_THAT(core5, ElementsAre(Eq(0), Eq(0), Eq(0), Eq(0), Eq(0), Eq(0), Eq(0), Eq(0), Eq(0), Eq(0)));
}

TEST(KCoreTest, WorksWithCliqueAndLooselyConnectedNodesWeighted) {
  // clang-format off
  // K5 with some loosely connected nodes
  Graph graph({0, 5, 10, 14, 19, 24, 25, 26, 27, 29, 30}, {
                  7, 4, 1, 2, 3,
                  8, 3, 4, 2, 0,
                  0, 1, 3, 4,
                  0, 2, 4, 1, 5,
                  3, 2, 1, 0, 6,
                  3,
                  4,
                  0,
                  1, 9,
                  8
              },
              {1, 1, 1, 1, 1, 1, 1, 1, 1, 1},
              {
                  1, 2, 2, 2, 2,
                  1, 2, 2, 2, 2,
                  2, 2, 2, 2,
                  3, 2, 2, 2, 2,
                  4, 2, 2, 2, 2,
                  3,
                  4,
                  1,
                  1, 6,
                  6
              });
  // clang-format on

  std::vector<EdgeWeight> core1{compute_k_core(graph, 1)};
  EXPECT_THAT(core1, ElementsAre(Ne(0), Ne(0), Ne(0), Ne(0), Ne(0), Ne(0), Ne(0), Ne(0), Ne(0), Ne(0)));

  std::vector<EdgeWeight> core2{compute_k_core(graph, 2, core1)};
  EXPECT_THAT(core2, ElementsAre(Ne(0), Ne(0), Ne(0), Ne(0), Ne(0), Ne(0), Ne(0), Eq(0), Ne(0), Ne(0)));

  std::vector<EdgeWeight> core3{compute_k_core(graph, 3, core2)};
  EXPECT_THAT(core3, ElementsAre(Ne(0), Ne(0), Ne(0), Ne(0), Ne(0), Ne(0), Ne(0), Eq(0), Ne(0), Ne(0)));

  std::vector<EdgeWeight> core4{compute_k_core(graph, 4, core3)};
  EXPECT_THAT(core4, ElementsAre(Ne(0), Ne(0), Ne(0), Ne(0), Ne(0), Eq(0), Ne(0), Eq(0), Ne(0), Ne(0)));

  std::vector<EdgeWeight> core5{compute_k_core(graph, 5, core4)};
  EXPECT_THAT(core5, ElementsAre(Ne(0), Ne(0), Ne(0), Ne(0), Ne(0), Eq(0), Eq(0), Eq(0), Ne(0), Ne(0)));

  std::vector<EdgeWeight> core6{compute_k_core(graph, 6, core5)};
  EXPECT_THAT(core6, ElementsAre(Ne(0), Ne(0), Ne(0), Ne(0), Ne(0), Eq(0), Eq(0), Eq(0), Ne(0), Ne(0)));

  std::vector<EdgeWeight> core7{compute_k_core(graph, 7, core6)};
  EXPECT_THAT(core7, ElementsAre(Ne(0), Ne(0), Ne(0), Ne(0), Ne(0), Eq(0), Eq(0), Eq(0), Eq(0), Eq(0)));

  std::vector<EdgeWeight> core8{compute_k_core(graph, 8, core7)};
  EXPECT_THAT(core8, ElementsAre(Ne(0), Ne(0), Ne(0), Ne(0), Ne(0), Eq(0), Eq(0), Eq(0), Eq(0), Eq(0)));

  std::vector<EdgeWeight> core9{compute_k_core(graph, 9, core8)};
  EXPECT_THAT(core9, ElementsAre(Eq(0), Eq(0), Eq(0), Eq(0), Eq(0), Eq(0), Eq(0), Eq(0), Eq(0), Eq(0)));
}
} // namespace kaminpar