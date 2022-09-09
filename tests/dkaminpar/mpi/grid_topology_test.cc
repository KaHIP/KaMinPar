/*******************************************************************************
 * @file:   grid_topology_test.cc
 * @author: Daniel Seemaier
 * @date:   09.09.2022
 * @brief:
 ******************************************************************************/
#include <gmock/gmock.h>

#include "dkaminpar/mpi/grid_topology.h"

namespace kaminpar::dist::mpi {
TEST(GridTopologyTest, size_1_rows) {
    GridTopology topo(1);
    EXPECT_EQ(topo.row(0), 0);
}

TEST(GridTopologyTest, size_1_cols) {
    GridTopology topo(1);
    EXPECT_EQ(topo.column(0), 0);
}

TEST(GridTopologyTest, size_1_virtual_cols) {
    GridTopology topo(1);
    EXPECT_EQ(topo.virtual_column(0), 0);
}

TEST(GridTopologyTest, size_4_rows) {
    GridTopology topo(4);
    EXPECT_EQ(topo.row(0), 0);
    EXPECT_EQ(topo.row(1), 0);
    EXPECT_EQ(topo.row(2), 1);
    EXPECT_EQ(topo.row(3), 1);
}

TEST(GridTopologyTest, size_4_cols) {
    GridTopology topo(4);
    EXPECT_EQ(topo.column(0), 0);
    EXPECT_EQ(topo.column(1), 1);
    EXPECT_EQ(topo.column(2), 0);
    EXPECT_EQ(topo.column(3), 1);
}

TEST(GridTopologyTest, size_4_virtual_cols) {
    GridTopology topo(4);
    EXPECT_EQ(topo.virtual_column(0), 0);
    EXPECT_EQ(topo.virtual_column(1), 1);
    EXPECT_EQ(topo.virtual_column(2), 0);
    EXPECT_EQ(topo.virtual_column(3), 1);
}

TEST(GridTopologyTest, size_8_rows) {
    GridTopology topo(8);
    EXPECT_EQ(topo.row(0), 0);
    EXPECT_EQ(topo.row(1), 0);
    EXPECT_EQ(topo.row(2), 0);
    EXPECT_EQ(topo.row(3), 0);
    EXPECT_EQ(topo.row(4), 1);
    EXPECT_EQ(topo.row(5), 1);
    EXPECT_EQ(topo.row(6), 1);
    EXPECT_EQ(topo.row(7), 1);
}

TEST(GridTopologyTest, size_8_cols) {
    GridTopology topo(8);
    EXPECT_EQ(topo.column(0), 0);
    EXPECT_EQ(topo.column(1), 1);
    EXPECT_EQ(topo.column(2), 2);
    EXPECT_EQ(topo.column(3), 3);
    EXPECT_EQ(topo.column(4), 0);
    EXPECT_EQ(topo.column(5), 1);
    EXPECT_EQ(topo.column(6), 2);
    EXPECT_EQ(topo.column(7), 3);
}

TEST(GridTopologyTest, size_8_virtual_cols) {
    GridTopology topo(8);
    EXPECT_EQ(topo.virtual_column(0), 0);
    EXPECT_EQ(topo.virtual_column(1), 1);
    EXPECT_EQ(topo.virtual_column(2), 2);
    EXPECT_EQ(topo.virtual_column(3), 3);
    EXPECT_EQ(topo.virtual_column(4), 0);
    EXPECT_EQ(topo.virtual_column(5), 1);
    EXPECT_EQ(topo.virtual_column(6), 2);
    EXPECT_EQ(topo.virtual_column(7), 3);
}

TEST(GridTopologyTest, size_21_rows) {
    GridTopology topo(21);
    EXPECT_EQ(topo.row(0), 0);
    EXPECT_EQ(topo.row(1), 0);
    EXPECT_EQ(topo.row(2), 0);
    EXPECT_EQ(topo.row(3), 0);
    EXPECT_EQ(topo.row(4), 0);
    EXPECT_EQ(topo.row(5), 0);
    EXPECT_EQ(topo.row(6), 1);
    EXPECT_EQ(topo.row(7), 1);
    EXPECT_EQ(topo.row(8), 1);
    EXPECT_EQ(topo.row(9), 1);
    EXPECT_EQ(topo.row(10), 1);
    EXPECT_EQ(topo.row(11), 2);
    EXPECT_EQ(topo.row(12), 2);
    EXPECT_EQ(topo.row(13), 2);
    EXPECT_EQ(topo.row(14), 2);
    EXPECT_EQ(topo.row(15), 2);
    EXPECT_EQ(topo.row(16), 3);
    EXPECT_EQ(topo.row(17), 3);
    EXPECT_EQ(topo.row(18), 3);
    EXPECT_EQ(topo.row(19), 3);
    EXPECT_EQ(topo.row(20), 3);
}

TEST(GridTopologyTest, size_21_cols) {
    GridTopology topo(21);
    EXPECT_EQ(topo.column(0), 0);
    EXPECT_EQ(topo.column(1), 1);
    EXPECT_EQ(topo.column(2), 2);
    EXPECT_EQ(topo.column(3), 3);
    EXPECT_EQ(topo.column(4), 4);
    EXPECT_EQ(topo.column(5), 5);
    EXPECT_EQ(topo.column(6), 0);
    EXPECT_EQ(topo.column(7), 1);
    EXPECT_EQ(topo.column(8), 2);
    EXPECT_EQ(topo.column(9), 3);
    EXPECT_EQ(topo.column(10), 4);
    EXPECT_EQ(topo.column(11), 0);
    EXPECT_EQ(topo.column(12), 1);
    EXPECT_EQ(topo.column(13), 2);
    EXPECT_EQ(topo.column(14), 3);
    EXPECT_EQ(topo.column(15), 4);
    EXPECT_EQ(topo.column(16), 0);
    EXPECT_EQ(topo.column(17), 1);
    EXPECT_EQ(topo.column(18), 2);
    EXPECT_EQ(topo.column(19), 3);
    EXPECT_EQ(topo.column(20), 4);
}

TEST(GridTopologyTest, size_21_virtual_cols) {
    GridTopology topo(21);
    EXPECT_EQ(topo.virtual_column(0), 0);
    EXPECT_EQ(topo.virtual_column(1), 1);
    EXPECT_EQ(topo.virtual_column(2), 2);
    EXPECT_EQ(topo.virtual_column(3), 3);
    EXPECT_EQ(topo.virtual_column(4), 4);
    EXPECT_EQ(topo.virtual_column(5), 0);
    EXPECT_EQ(topo.virtual_column(6), 0);
    EXPECT_EQ(topo.virtual_column(7), 1);
    EXPECT_EQ(topo.virtual_column(8), 2);
    EXPECT_EQ(topo.virtual_column(9), 3);
    EXPECT_EQ(topo.virtual_column(10), 4);
    EXPECT_EQ(topo.virtual_column(11), 0);
    EXPECT_EQ(topo.virtual_column(12), 1);
    EXPECT_EQ(topo.virtual_column(13), 2);
    EXPECT_EQ(topo.virtual_column(14), 3);
    EXPECT_EQ(topo.virtual_column(15), 4);
    EXPECT_EQ(topo.virtual_column(16), 0);
    EXPECT_EQ(topo.virtual_column(17), 1);
    EXPECT_EQ(topo.virtual_column(18), 2);
    EXPECT_EQ(topo.virtual_column(19), 3);
    EXPECT_EQ(topo.virtual_column(20), 4);
}
} // namespace kaminpar::dist::mpi
