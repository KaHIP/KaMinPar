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
#include "kaminpar/utility/strings.h"

#include "gmock/gmock.h"

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