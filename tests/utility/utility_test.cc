#include "gmock/gmock.h"
#include "kaminpar/utils/strings.h"

namespace kaminpar::utility::str {
TEST(UtilityTest, ExtractBasename) {
    EXPECT_EQ(extract_basename("test.graph"), "test");
    EXPECT_EQ(extract_basename("/test.graph"), "test");
    EXPECT_EQ(extract_basename("//test.graph"), "test");
    EXPECT_EQ(extract_basename("/test.graph/"), "");
    EXPECT_EQ(extract_basename("test"), "test");
    EXPECT_EQ(extract_basename("/test"), "test");
    EXPECT_EQ(extract_basename("/home/dummy/graphs/europe.osm.graph"), "europe.osm");
}
} // namespace kaminpar::utility::str