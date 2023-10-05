#include <gmock/gmock.h>

#include "kaminpar-common/datastructures/marker.h"

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
