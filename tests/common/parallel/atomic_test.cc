#include <gmock/gmock.h>

#include "kaminpar-common/parallel/atomic.h"

namespace kaminpar::parallel {

TEST(AtomicTest, PostfixDecrementReturnsOldValueAndDecrements) {
  Atomic<int> value(5);

  EXPECT_EQ(value--, 5);
  EXPECT_EQ(value.load(), 4);
}

} // namespace kaminpar::parallel
