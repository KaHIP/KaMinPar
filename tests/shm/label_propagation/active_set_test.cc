/*******************************************************************************
 * Unit tests for the ActiveSet building block.
 *
 * @file:   active_set_test.cc
 * @author: Daniel Seemaier
 * @date:   21.09.2021
 ******************************************************************************/
#include <gmock/gmock.h>

#include "kaminpar-shm/label_propagation/active_set.h"

namespace kaminpar::testing {

TEST(ActiveSetEnabledTest, InitiallyAllActiveAfterReset) {
  ActiveSet<true> as;
  as.reset(5);
  for (std::size_t i = 0; i < 5; ++i) {
    EXPECT_TRUE(as.is_active(i));
  }
}

TEST(ActiveSetEnabledTest, MarkInactiveAndBackActive) {
  ActiveSet<true> as;
  as.reset(5);
  as.mark_inactive(2);
  EXPECT_FALSE(as.is_active(2));
  EXPECT_TRUE(as.is_active(1));
  EXPECT_TRUE(as.is_active(3));
  as.mark_active(2);
  EXPECT_TRUE(as.is_active(2));
}

TEST(ActiveSetEnabledTest, ResetRestoresAllActive) {
  ActiveSet<true> as;
  as.reset(4);
  as.mark_inactive(0);
  as.mark_inactive(1);
  as.mark_inactive(3);
  as.reset(4);
  for (std::size_t i = 0; i < 4; ++i) {
    EXPECT_TRUE(as.is_active(i));
  }
}

TEST(ActiveSetEnabledTest, AllocateGrowsCapacity) {
  ActiveSet<true> as;
  as.allocate(3);
  as.reset(3);
  // Grow to larger size: should not crash
  as.allocate(10);
  as.reset(10);
  for (std::size_t i = 0; i < 10; ++i) {
    EXPECT_TRUE(as.is_active(i));
  }
}

TEST(ActiveSetEnabledTest, TakeAndSetRoundTrip) {
  ActiveSet<true> as;
  as.reset(3);
  as.mark_inactive(1);

  auto data = as.take();
  EXPECT_EQ(data[0], 1);
  EXPECT_EQ(data[1], 0);
  EXPECT_EQ(data[2], 1);

  ActiveSet<true> as2;
  as2.set(std::move(data));
  EXPECT_TRUE(as2.is_active(0));
  EXPECT_FALSE(as2.is_active(1));
  EXPECT_TRUE(as2.is_active(2));
}

TEST(ActiveSetDisabledTest, IsActiveAlwaysReturnsTrue) {
  ActiveSet<false> as;
  as.allocate(5);
  as.reset(5);
  for (std::size_t i = 0; i < 5; ++i) {
    EXPECT_TRUE(as.is_active(i));
  }
  as.mark_inactive(2);
  EXPECT_TRUE(as.is_active(2)); // no-op
}

TEST(ActiveSetDisabledTest, TakeReturnsEmptyArray) {
  ActiveSet<false> as;
  as.reset(5);
  auto data = as.take();
  EXPECT_EQ(data.size(), 0);
}

} // namespace kaminpar::testing
