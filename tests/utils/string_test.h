#include <gmock/gmock.h>

#include "kaminpar/utils/strings.h"

using namespace kaminpar;

TEST(UtilityTest, extract_basename) {
    EXPECT_EQ(utility::str::extract_basename("test.graph"), "test");
    EXPECT_EQ(utility::str::extract_basename("/test.graph"), "test");
    EXPECT_EQ(utility::str::extract_basename("//test.graph"), "test");
    EXPECT_EQ(utility::str::extract_basename("/test.graph/"), "");
    EXPECT_EQ(utility::str::extract_basename("test"), "test");
    EXPECT_EQ(utility::str::extract_basename("/test"), "test");
    EXPECT_EQ(utility::str::extract_basename("/home/dummy/graphs/europe.osm.graph"), "europe.osm");
}
