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
#include "kaminpar/datastructure/marker.h"

#include "gmock/gmock.h"

namespace kaminpar {
TEST(MarkerTests, ConcurrentMarkersWork) {
  Marker<3> marker(1);
  EXPECT_FALSE(marker.get(0, 0));
  EXPECT_FALSE(marker.get(0, 1));
  EXPECT_FALSE(marker.get(0, 2));
  marker.set(0, 0);
  EXPECT_TRUE(marker.get(0, 0));
  EXPECT_FALSE(marker.get(0, 1));
  EXPECT_FALSE(marker.get(0, 2));
  marker.set(0, 1);
  EXPECT_TRUE(marker.get(0, 0));
  EXPECT_TRUE(marker.get(0, 1));
  EXPECT_FALSE(marker.get(0, 2));
  marker.set(0, 2);
  EXPECT_TRUE(marker.get(0, 0));
  EXPECT_TRUE(marker.get(0, 1));
  EXPECT_TRUE(marker.get(0, 2));
}

TEST(MarkerTests, ResetWorks) {
  Marker<2> marker(1);
  marker.set(0, 0);
  marker.set(0, 1);
  marker.reset();
  EXPECT_FALSE(marker.get(0, 0));
  EXPECT_FALSE(marker.get(0, 1));
}

TEST(MarkerTests, MarkingOutOfOrderWorks) {
  Marker<2> marker(1);
  marker.set(0, 1);
  EXPECT_FALSE(marker.get(0, 0));
  EXPECT_TRUE(marker.get(0, 1));
  marker.set(0, 0);
  EXPECT_TRUE(marker.get(0, 0));
  EXPECT_TRUE(marker.get(0, 1));
}

TEST(MarkerTests, MarkingOutOfOrderAfterResetWorks) {
  Marker<2> marker(1);
  marker.set(0, 0);
  marker.reset();
  marker.set(0, 1);
  EXPECT_FALSE(marker.get(0, 0));
  EXPECT_TRUE(marker.get(0, 1));
}

TEST(MarkerTests, MarkingAfterResetsWorks) {
  Marker<2> marker(1);
  marker.reset();
  marker.set(0, 0);
  EXPECT_TRUE(marker.get(0, 0));
  EXPECT_FALSE(marker.get(0, 1));
  marker.reset();
  EXPECT_FALSE(marker.get(0, 0));
  EXPECT_FALSE(marker.get(0, 1));
  marker.set(0, 1);
  EXPECT_FALSE(marker.get(0, 0));
  EXPECT_TRUE(marker.get(0, 1));
  marker.set(0, 0);
  EXPECT_TRUE(marker.get(0, 0));
  EXPECT_TRUE(marker.get(0, 1));
  marker.reset();
  EXPECT_FALSE(marker.get(0, 0));
  EXPECT_FALSE(marker.get(0, 1));
}
} // namespace kaminpar