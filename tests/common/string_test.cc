#include <gmock/gmock.h>

#include "common/utils/strings.h"

namespace kaminpar {
TEST(UtilityTest, extract_basename) {
    EXPECT_EQ(str::extract_basename("test.graph"), "test");
    EXPECT_EQ(str::extract_basename("/test.graph"), "test");
    EXPECT_EQ(str::extract_basename("//test.graph"), "test");
    EXPECT_EQ(str::extract_basename("/test.graph/"), "");
    EXPECT_EQ(str::extract_basename("test"), "test");
    EXPECT_EQ(str::extract_basename("/test"), "test");
    EXPECT_EQ(str::extract_basename("/home/dummy/graphs/europe.osm.graph"), "europe.osm");
}
} // namespace kaminpar
