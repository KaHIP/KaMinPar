/*******************************************************************************
 * @file:   grid_topology_test.cc
 * @author: Daniel Seemaier
 * @date:   09.09.2022
 * @brief:
 ******************************************************************************/
#include <gmock/gmock.h>

#include "kaminpar-mpi/grid_topology.h"

namespace kaminpar::mpi {
TEST(GridTopologyTest, size_1_rows) {
  GridTopology topo(1);
  EXPECT_EQ(topo.row(0), 0);
}

TEST(GridTopologyTest, size_1_cols) {
  GridTopology topo(1);
  EXPECT_EQ(topo.col(0), 0);
}

TEST(GridTopologyTest, size_1_virtual_cols) {
  GridTopology topo(1);
  EXPECT_EQ(topo.virtual_col(0), 0);
}

TEST(GridTopologyTest, size_1_sizes) {
  GridTopology topo(1);
  EXPECT_EQ(topo.row_size(0), 1);
  EXPECT_EQ(topo.col_size(0), 1);
  EXPECT_EQ(topo.virtual_col_size(0), 1);
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
  EXPECT_EQ(topo.col(0), 0);
  EXPECT_EQ(topo.col(1), 1);
  EXPECT_EQ(topo.col(2), 0);
  EXPECT_EQ(topo.col(3), 1);
}

TEST(GridTopologyTest, size_4_sizes) {
  GridTopology topo(4);
  EXPECT_EQ(topo.row_size(0), 2);
  EXPECT_EQ(topo.row_size(1), 2);
  EXPECT_EQ(topo.col_size(0), 2);
  EXPECT_EQ(topo.col_size(1), 2);
  EXPECT_EQ(topo.virtual_col_size(0), 2);
  EXPECT_EQ(topo.virtual_col_size(1), 2);
}

TEST(GridTopologyTest, size_4_virtual_cols) {
  GridTopology topo(4);
  EXPECT_EQ(topo.virtual_col(0), 0);
  EXPECT_EQ(topo.virtual_col(1), 1);
  EXPECT_EQ(topo.virtual_col(2), 0);
  EXPECT_EQ(topo.virtual_col(3), 1);
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
  EXPECT_EQ(topo.col(0), 0);
  EXPECT_EQ(topo.col(1), 1);
  EXPECT_EQ(topo.col(2), 2);
  EXPECT_EQ(topo.col(3), 3);
  EXPECT_EQ(topo.col(4), 0);
  EXPECT_EQ(topo.col(5), 1);
  EXPECT_EQ(topo.col(6), 2);
  EXPECT_EQ(topo.col(7), 3);
}

TEST(GridTopologyTest, size_8_virtual_cols) {
  GridTopology topo(8);
  EXPECT_EQ(topo.virtual_col(0), 0);
  EXPECT_EQ(topo.virtual_col(1), 1);
  EXPECT_EQ(topo.virtual_col(2), 2);
  EXPECT_EQ(topo.virtual_col(3), 3);
  EXPECT_EQ(topo.virtual_col(4), 0);
  EXPECT_EQ(topo.virtual_col(5), 1);
  EXPECT_EQ(topo.virtual_col(6), 2);
  EXPECT_EQ(topo.virtual_col(7), 3);
}

TEST(GridTopologyTest, size_8_virtual_elements) {
  GridTopology topo(8);
  EXPECT_EQ(topo.virtual_element(0, 0), 0);
  EXPECT_EQ(topo.virtual_element(0, 1), 1);
  EXPECT_EQ(topo.virtual_element(0, 2), 2);
  EXPECT_EQ(topo.virtual_element(0, 3), 3);
  EXPECT_EQ(topo.virtual_element(1, 0), 4);
  EXPECT_EQ(topo.virtual_element(1, 1), 5);
  EXPECT_EQ(topo.virtual_element(1, 2), 6);
  EXPECT_EQ(topo.virtual_element(1, 3), 7);
}

TEST(GridTopologyTest, size_8_sizes) {
  GridTopology topo(8);
  EXPECT_EQ(topo.row_size(0), 4);
  EXPECT_EQ(topo.row_size(1), 4);
  EXPECT_EQ(topo.col_size(0), 2);
  EXPECT_EQ(topo.col_size(1), 2);
  EXPECT_EQ(topo.col_size(2), 2);
  EXPECT_EQ(topo.col_size(3), 2);
  EXPECT_EQ(topo.virtual_col_size(0), 2);
  EXPECT_EQ(topo.virtual_col_size(1), 2);
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
  EXPECT_EQ(topo.col(0), 0);
  EXPECT_EQ(topo.col(1), 1);
  EXPECT_EQ(topo.col(2), 2);
  EXPECT_EQ(topo.col(3), 3);
  EXPECT_EQ(topo.col(4), 4);
  EXPECT_EQ(topo.col(5), 5);
  EXPECT_EQ(topo.col(6), 0);
  EXPECT_EQ(topo.col(7), 1);
  EXPECT_EQ(topo.col(8), 2);
  EXPECT_EQ(topo.col(9), 3);
  EXPECT_EQ(topo.col(10), 4);
  EXPECT_EQ(topo.col(11), 0);
  EXPECT_EQ(topo.col(12), 1);
  EXPECT_EQ(topo.col(13), 2);
  EXPECT_EQ(topo.col(14), 3);
  EXPECT_EQ(topo.col(15), 4);
  EXPECT_EQ(topo.col(16), 0);
  EXPECT_EQ(topo.col(17), 1);
  EXPECT_EQ(topo.col(18), 2);
  EXPECT_EQ(topo.col(19), 3);
  EXPECT_EQ(topo.col(20), 4);
}

TEST(GridTopologyTest, size_21_virtual_cols) {
  GridTopology topo(21);
  EXPECT_EQ(topo.virtual_col(0), 0);
  EXPECT_EQ(topo.virtual_col(1), 1);
  EXPECT_EQ(topo.virtual_col(2), 2);
  EXPECT_EQ(topo.virtual_col(3), 3);
  EXPECT_EQ(topo.virtual_col(4), 4);
  EXPECT_EQ(topo.virtual_col(5), 0);
  EXPECT_EQ(topo.virtual_col(6), 0);
  EXPECT_EQ(topo.virtual_col(7), 1);
  EXPECT_EQ(topo.virtual_col(8), 2);
  EXPECT_EQ(topo.virtual_col(9), 3);
  EXPECT_EQ(topo.virtual_col(10), 4);
  EXPECT_EQ(topo.virtual_col(11), 0);
  EXPECT_EQ(topo.virtual_col(12), 1);
  EXPECT_EQ(topo.virtual_col(13), 2);
  EXPECT_EQ(topo.virtual_col(14), 3);
  EXPECT_EQ(topo.virtual_col(15), 4);
  EXPECT_EQ(topo.virtual_col(16), 0);
  EXPECT_EQ(topo.virtual_col(17), 1);
  EXPECT_EQ(topo.virtual_col(18), 2);
  EXPECT_EQ(topo.virtual_col(19), 3);
  EXPECT_EQ(topo.virtual_col(20), 4);
}

TEST(GridTopologyTest, size_21_sizes) {
  GridTopology topo(21);
  EXPECT_EQ(topo.row_size(0), 6);
  EXPECT_EQ(topo.row_size(1), 5);
  EXPECT_EQ(topo.row_size(2), 5);
  EXPECT_EQ(topo.row_size(3), 5);
  EXPECT_EQ(topo.col_size(0), 4);
  EXPECT_EQ(topo.col_size(1), 4);
  EXPECT_EQ(topo.col_size(2), 4);
  EXPECT_EQ(topo.col_size(3), 4);
  EXPECT_EQ(topo.col_size(4), 4);
  EXPECT_EQ(topo.col_size(5), 1);
  EXPECT_EQ(topo.virtual_col_size(0), 5);
  EXPECT_EQ(topo.virtual_col_size(1), 4);
  EXPECT_EQ(topo.virtual_col_size(2), 4);
  EXPECT_EQ(topo.virtual_col_size(3), 4);
  EXPECT_EQ(topo.virtual_col_size(4), 4);
}

TEST(GridTopologyTest, size_21_virtual_elements) {
  GridTopology topo(21);
  EXPECT_EQ(topo.virtual_element(0, 0), 0);
  EXPECT_EQ(topo.virtual_element(0, 1), 1);
  EXPECT_EQ(topo.virtual_element(0, 2), 2);
  EXPECT_EQ(topo.virtual_element(0, 3), 3);
  EXPECT_EQ(topo.virtual_element(0, 4), 4);
  EXPECT_EQ(topo.virtual_element(1, 0), 6);
  EXPECT_EQ(topo.virtual_element(1, 1), 7);
  EXPECT_EQ(topo.virtual_element(1, 2), 8);
  EXPECT_EQ(topo.virtual_element(1, 3), 9);
  EXPECT_EQ(topo.virtual_element(1, 4), 10);
  EXPECT_EQ(topo.virtual_element(2, 0), 11);
  EXPECT_EQ(topo.virtual_element(2, 1), 12);
  EXPECT_EQ(topo.virtual_element(2, 2), 13);
  EXPECT_EQ(topo.virtual_element(2, 3), 14);
  EXPECT_EQ(topo.virtual_element(2, 4), 15);
  EXPECT_EQ(topo.virtual_element(3, 0), 16);
  EXPECT_EQ(topo.virtual_element(3, 1), 17);
  EXPECT_EQ(topo.virtual_element(3, 2), 18);
  EXPECT_EQ(topo.virtual_element(3, 3), 19);
  EXPECT_EQ(topo.virtual_element(3, 4), 20);
  EXPECT_EQ(topo.virtual_element(4, 0), 5);
}
} // namespace kaminpar::mpi
