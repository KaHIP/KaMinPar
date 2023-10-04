#include <gmock/gmock.h>

#include "kaminpar-common/datastructures/binary_heap.h"

using ::testing::ElementsAre;

namespace kaminpar {
TEST(BinaryHeapTest, SizeAndEmptyWork) {
  BinaryMinHeap<int> heap(10);
  EXPECT_TRUE(heap.empty());
  EXPECT_EQ(heap.size(), 0);

  heap.push(0, 20);
  EXPECT_EQ(heap.size(), 1);

  heap.push(1, 30);
  EXPECT_EQ(heap.size(), 2);

  heap.pop();
  EXPECT_EQ(heap.size(), 1);

  heap.pop();
  EXPECT_TRUE(heap.empty());
  EXPECT_EQ(heap.size(), 0);
}

TEST(BinaryHeapTest, MinElementInSequenceOfPushesWorks) {
  BinaryMinHeap<int> heap(10);

  heap.push(0, 10);
  EXPECT_EQ(heap.peek_id(), 0);
  EXPECT_EQ(heap.peek_key(), 10);

  heap.push(1, 5);
  EXPECT_EQ(heap.peek_id(), 1);
  EXPECT_EQ(heap.peek_key(), 5);

  heap.push(2, 7);
  EXPECT_EQ(heap.peek_id(), 1);
  EXPECT_EQ(heap.peek_key(), 5);

  heap.push(3, -100);
  EXPECT_EQ(heap.peek_id(), 3);
  EXPECT_EQ(heap.peek_key(), -100);
}

TEST(BinaryHeapTest, MinElementInSequenceOfPushesAndPopsWorks) {
  BinaryMinHeap<int> heap(10);

  heap.push(0, 10);
  heap.push(1, 5);
  heap.push(2, 7);
  heap.push(3, 1);

  EXPECT_EQ(heap.peek_key(), 1);
  heap.pop();
  EXPECT_EQ(heap.peek_key(), 5);
  heap.pop();
  EXPECT_EQ(heap.peek_key(), 7);
  heap.pop();
  EXPECT_EQ(heap.peek_key(), 10);
}

TEST(BinaryHeapTest, DecreaseKeyWorks) {
  BinaryMinHeap<int> heap(10);

  heap.push(0, 10);
  heap.push(1, 20);

  EXPECT_EQ(heap.peek_key(), 10);
  heap.decrease_priority(1, 5);
  EXPECT_EQ(heap.peek_id(), 1);
  EXPECT_EQ(heap.peek_key(), 5);
  heap.pop();
  EXPECT_EQ(heap.peek_key(), 10);
}

TEST(BinaryHeapTest, MaxHeapWorksWithPush) {
  BinaryMaxHeap<int> heap(10);

  heap.push(0, 15);
  EXPECT_EQ(heap.peek_key(), 15);
  heap.push(1, 10);
  EXPECT_EQ(heap.peek_key(), 15);
  heap.push(2, 20);
  EXPECT_EQ(heap.peek_key(), 20);
}

TEST(BinaryHeapTest, MaxHeapWorksWithPushAndPop) {
  BinaryMaxHeap<int> heap(10);

  heap.push(0, 15);
  heap.push(1, 10);
  heap.push(2, 20);

  EXPECT_EQ(heap.peek_key(), 20);
  EXPECT_EQ(heap.peek_id(), 2);
  heap.pop();
  EXPECT_EQ(heap.peek_key(), 15);
  EXPECT_EQ(heap.peek_id(), 0);
  heap.pop();
  EXPECT_EQ(heap.peek_key(), 10);
  EXPECT_EQ(heap.peek_id(), 1);
  heap.pop();
  EXPECT_TRUE(heap.empty());
}

TEST(BinaryHeapTest, MaxHeapWorksWithChangeKey) {
  BinaryMaxHeap<int> heap(10);

  heap.push(0, 15);
  heap.push(1, 10);
  heap.push(2, 20);

  heap.change_priority(1, 30);
  EXPECT_EQ(heap.peek_key(), 30);
  EXPECT_EQ(heap.peek_id(), 1);

  heap.change_priority(0, 40);
  EXPECT_EQ(heap.peek_key(), 40);
  EXPECT_EQ(heap.peek_id(), 0);

  heap.pop();
  EXPECT_EQ(heap.peek_key(), 30);
  EXPECT_EQ(heap.peek_id(), 1);

  heap.change_priority(2, 31);
  EXPECT_EQ(heap.peek_key(), 31);
  EXPECT_EQ(heap.peek_id(), 2);

  heap.pop();
  EXPECT_EQ(heap.peek_key(), 30);
  EXPECT_EQ(heap.peek_id(), 1);

  heap.pop();
  EXPECT_TRUE(heap.empty());
}

TEST(BinaryHeapTest, RemoveWorks) {
  BinaryMaxHeap<int> heap(10);
  heap.push(0, 15);
  heap.push(1, 10);
  heap.push(2, 20);

  heap.remove(1);
  EXPECT_EQ(heap.size(), 2);
  EXPECT_EQ(heap.peek_id(), 2);
  EXPECT_EQ(heap.peek_key(), 20);

  heap.remove(2);
  EXPECT_EQ(heap.size(), 1);
  EXPECT_EQ(heap.peek_id(), 0);
  EXPECT_EQ(heap.peek_key(), 15);

  heap.remove(0);
  EXPECT_TRUE(heap.empty());
}

TEST(BinaryHeapTest, RemoveWorksFromMinHeap) {
  BinaryMinHeap<int> heap(10);
  heap.push(0, 0);
  heap.push(1, 1);
  heap.push(2, 2);
  heap.push(3, 3);

  heap.remove(3);
  EXPECT_EQ(heap.peek_id(), 0);
  heap.remove(2);
  EXPECT_EQ(heap.peek_id(), 0);
  heap.remove(1);
  EXPECT_EQ(heap.peek_id(), 0);
  heap.remove(0);
  EXPECT_TRUE(heap.empty());
}

TEST(NonaddressableBinaryHeapTest, PushWorks) {
  DynamicBinaryMinHeap<int, int> heap;
  heap.push(10, 10);
  EXPECT_EQ(heap.peek_id(), 10);
  heap.push(20, 20);
  EXPECT_EQ(heap.peek_id(), 10);
  heap.push(30, 5);
  EXPECT_EQ(heap.peek_id(), 30);
  heap.push(40, 11);
  EXPECT_EQ(heap.peek_id(), 30);
  heap.push(50, 0);
  EXPECT_EQ(heap.peek_id(), 50);
}

TEST(NonaddressableBinaryHeapTest, PopWorks) {
  DynamicBinaryMinHeap<int, int> heap;
  heap.push(10, 10);
  heap.push(20, 20);
  heap.push(30, 5);
  heap.push(40, 11);
  heap.push(50, 0);

  EXPECT_EQ(heap.peek_id(), 50);
  heap.pop();
  EXPECT_EQ(heap.peek_id(), 30);
  heap.pop();
  EXPECT_EQ(heap.peek_id(), 10);
  heap.pop();
  EXPECT_EQ(heap.peek_id(), 40);
  heap.pop();
  EXPECT_EQ(heap.peek_id(), 20);
  heap.pop();
  EXPECT_TRUE(heap.empty());
}

TEST(NonaddressableBinaryHeapTest, RepeatedPushPopWorks) {
  DynamicBinaryMinHeap<int, int> heap;
  heap.push(10, 10);
  EXPECT_EQ(heap.peek_key(), 10);
  heap.pop();
  EXPECT_TRUE(heap.empty());
  heap.push(11, 11);
  EXPECT_EQ(heap.peek_key(), 11);
  heap.pop();
  EXPECT_EQ(heap.size(), 0);
  heap.push(0, 0);
  EXPECT_EQ(heap.peek_key(), 0);
  heap.pop();
  EXPECT_TRUE(heap.empty());
}

TEST(NonaddressableBinaryHeapTest, SortingWithHeapWorks) {
  DynamicBinaryMinHeap<int, int> heap;
  const std::vector<int> sequence{
      13, -12, 0, 4, 129, 21, -123, -23, 12, -5, -1, 434, 13, 3451, 123};
  for (const auto e : sequence) {
    heap.push(e, e);
  }

  std::vector<int> sorted;
  while (!heap.empty()) {
    sorted.push_back(heap.peek_key());
    heap.pop();
  }

  EXPECT_THAT(
      sorted, ElementsAre(-123, -23, -12, -5, -1, 0, 4, 12, 13, 13, 21, 123, 129, 434, 3451)
  );
}

TEST(NonaddressableBinaryHeapTest, PushAfterPopWorks) {
  DynamicBinaryMinHeap<int, int> heap;
  heap.push(10, 10);
  heap.push(5, 5);
  heap.push(15, 15);
  heap.push(0, 0);
  EXPECT_EQ(heap.peek_key(), 0);
  heap.pop();
  EXPECT_EQ(heap.peek_key(), 5);
  heap.push(0, 0);
  EXPECT_EQ(heap.peek_key(), 0);
  heap.pop();
  EXPECT_EQ(heap.peek_key(), 5);
  heap.pop();
  EXPECT_EQ(heap.peek_key(), 10);
  heap.push(-1, -1);
  EXPECT_EQ(heap.peek_key(), -1);
  heap.pop();
  heap.pop();
  EXPECT_EQ(heap.peek_key(), 15);
  heap.pop();
  EXPECT_TRUE(heap.empty());
}

TEST(NonaddressableBinaryHeapTest, MaxHeapWorks) {
  DynamicBinaryMaxHeap<int, int> heap;
  heap.push(0, 0);
  heap.push(1, 1);
  heap.push(0, 0);
  heap.push(10, 10);
  heap.push(10, 10);
  EXPECT_EQ(heap.peek_key(), 10);
  heap.pop();
  EXPECT_EQ(heap.peek_key(), 10);
  heap.pop();
  EXPECT_EQ(heap.peek_key(), 1);
  heap.pop();
  EXPECT_EQ(heap.peek_key(), 0);
  heap.pop();
  EXPECT_EQ(heap.peek_key(), 0);
  heap.pop();
  EXPECT_TRUE(heap.empty());
}

TEST(NonaddressableBinaryHeapTest, WorksWithDuplicateIDs) {
  DynamicBinaryMinHeap<int, int> heap;
  heap.push(0, 10);
  heap.push(0, 20);
  heap.push(0, 5);
  heap.push(0, 7);
  heap.push(1, 11);
  heap.push(1, 12);
  heap.push(1, -1);
  heap.push(2, -2);

  EXPECT_EQ(heap.peek_id(), 2);
  EXPECT_EQ(heap.peek_key(), -2);
  heap.pop();
  EXPECT_EQ(heap.peek_id(), 1);
  EXPECT_EQ(heap.peek_key(), -1);
  heap.pop();
  EXPECT_EQ(heap.peek_id(), 0);
  EXPECT_EQ(heap.peek_key(), 5);
  heap.pop();
  EXPECT_EQ(heap.peek_id(), 0);
  EXPECT_EQ(heap.peek_key(), 7);
  heap.pop();
  EXPECT_EQ(heap.peek_id(), 0);
  EXPECT_EQ(heap.peek_key(), 10);
  heap.pop();
  EXPECT_EQ(heap.peek_id(), 1);
  EXPECT_EQ(heap.peek_key(), 11);
  heap.pop();
  EXPECT_EQ(heap.peek_id(), 1);
  EXPECT_EQ(heap.peek_key(), 12);
  heap.pop();
  EXPECT_EQ(heap.peek_id(), 0);
  EXPECT_EQ(heap.peek_key(), 20);
  heap.pop();
  EXPECT_TRUE(heap.empty());
}

TEST(DynamicBinaryForestTest, PushContainsPopSequenceWorks) {
  DynamicBinaryMaxForest<int, int> heap(10, 2);

  for (const std::size_t i : {0, 1, 2, 3, 4, 5, 6, 7, 8, 9}) {
    EXPECT_FALSE(heap.contains(i));
  }
  EXPECT_TRUE(heap.empty(0));
  EXPECT_TRUE(heap.empty(1));
  EXPECT_EQ(heap.size(0), 0);
  EXPECT_EQ(heap.size(1), 0);

  heap.push(0, 0, 10);
  heap.push(0, 1, 20);
  heap.push(0, 2, 0);
  heap.push(0, 3, 5);

  heap.push(1, 4, 0);
  heap.push(1, 5, 5);
  heap.push(1, 6, -5);

  for (const std::size_t i : {0, 1, 2, 3, 4, 5, 6}) {
    EXPECT_TRUE(heap.contains(i));
  }
  for (const std::size_t i : {7, 8, 9}) {
    EXPECT_FALSE(heap.contains(i));
  }
  EXPECT_EQ(heap.size(0), 4);
  EXPECT_EQ(heap.size(1), 3);

  EXPECT_EQ(heap.peek_key(0), 20);
  EXPECT_EQ(heap.peek_id(0), 1);
  EXPECT_EQ(heap.peek_key(1), 5);
  EXPECT_EQ(heap.peek_id(1), 5);
  heap.pop(0);
  EXPECT_FALSE(heap.contains(1));
  EXPECT_EQ(heap.peek_key(0), 10);
  EXPECT_EQ(heap.peek_id(0), 0);

  heap.pop(1);
  EXPECT_FALSE(heap.contains(5));
  EXPECT_EQ(heap.peek_key(1), 0);
  EXPECT_EQ(heap.peek_id(1), 4);

  heap.pop(1);
  EXPECT_FALSE(heap.contains(4));
  EXPECT_EQ(heap.peek_key(1), -5);
  EXPECT_EQ(heap.peek_id(1), 6);

  EXPECT_EQ(heap.peek_key(0), 10);
  EXPECT_EQ(heap.peek_id(0), 0);
  heap.pop(0);
  EXPECT_EQ(heap.peek_key(0), 5);
  EXPECT_EQ(heap.peek_id(0), 3);

  heap.pop(1);
  heap.pop(0);
  heap.pop(0);

  for (const std::size_t i : {0, 1, 2, 3, 4, 5, 6, 7, 8, 9}) {
    EXPECT_FALSE(heap.contains(i));
  }
  EXPECT_TRUE(heap.empty(0));
  EXPECT_TRUE(heap.empty(1));
  EXPECT_EQ(heap.size(0), 0);
  EXPECT_EQ(heap.size(1), 0);
}

TEST(DynamicBinaryForestTest, SizeAndEmptyWork) {
  DynamicBinaryMaxForest<int, int> heap(10, 1);
  EXPECT_TRUE(heap.empty(0));
  EXPECT_EQ(heap.size(0), 0);

  heap.push(0, 0, 20);
  EXPECT_EQ(heap.size(0), 1);

  heap.push(0, 1, 30);
  EXPECT_EQ(heap.size(0), 2);

  heap.pop(0);
  EXPECT_EQ(heap.size(0), 1);

  heap.pop(0);
  EXPECT_TRUE(heap.empty(0));
  EXPECT_EQ(heap.size(0), 0);
}

TEST(DynamicBinaryForestTest, MinElementInSequenceOfPushesWorks) {
  DynamicBinaryMinForest<int, int> heap(10, 1);

  heap.push(0, 0, 10);
  EXPECT_THAT(heap.peek_id(0), 0);
  EXPECT_THAT(heap.peek_key(0), 10);

  heap.push(0, 1, 5);
  EXPECT_THAT(heap.peek_id(0), 1);
  EXPECT_THAT(heap.peek_key(0), 5);

  heap.push(0, 2, 7);
  EXPECT_THAT(heap.peek_id(0), 1);
  EXPECT_THAT(heap.peek_key(0), 5);

  heap.push(0, 3, -100);
  EXPECT_THAT(heap.peek_id(0), 3);
  EXPECT_THAT(heap.peek_key(0), -100);
}

TEST(DynamicBinaryForestTest, MinElementInSequenceOfPushesAndPopsWorks) {
  DynamicBinaryMinForest<int, int> heap(10, 1);

  heap.push(0, 0, 10);
  heap.push(0, 1, 5);
  heap.push(0, 2, 7);
  heap.push(0, 3, 1);

  EXPECT_EQ(heap.peek_key(0), 1);
  heap.pop(0);
  EXPECT_EQ(heap.peek_key(0), 5);
  heap.pop(0);
  EXPECT_EQ(heap.peek_key(0), 7);
  heap.pop(0);
  EXPECT_EQ(heap.peek_key(0), 10);
}

TEST(DynamicBinaryForestTest, DecreaseKeyWorks) {
  DynamicBinaryMinForest<int, int> heap(10, 1);

  heap.push(0, 0, 10);
  heap.push(0, 1, 20);

  EXPECT_EQ(heap.peek_key(0), 10);
  heap.decrease_priority(0, 1, 5);
  EXPECT_EQ(heap.peek_id(0), 1);
  EXPECT_EQ(heap.peek_key(0), 5);
  heap.pop(0);
  EXPECT_EQ(heap.peek_key(0), 10);
}

TEST(DynamicBinaryForestTest, MaxHeapWorksWithPush) {
  DynamicBinaryMaxForest<int, int> heap(10, 1);

  heap.push(0, 0, 15);
  EXPECT_EQ(heap.peek_key(0), 15);
  heap.push(0, 1, 10);
  EXPECT_EQ(heap.peek_key(0), 15);
  heap.push(0, 2, 20);
  EXPECT_EQ(heap.peek_key(0), 20);
}

TEST(DynamicBinaryForestTest, MaxHeapWorksWithPushAndPop) {
  DynamicBinaryMaxForest<int, int> heap(10, 1);

  heap.push(0, 0, 15);
  heap.push(0, 1, 10);
  heap.push(0, 2, 20);

  EXPECT_EQ(heap.peek_key(0), 20);
  EXPECT_EQ(heap.peek_id(0), 2);
  heap.pop(0);
  EXPECT_EQ(heap.peek_key(0), 15);
  EXPECT_EQ(heap.peek_id(0), 0);
  heap.pop(0);
  EXPECT_EQ(heap.peek_key(0), 10);
  EXPECT_EQ(heap.peek_id(0), 1);
  heap.pop(0);
  EXPECT_TRUE(heap.empty(0));
}

TEST(DynamicBinaryForestTest, MaxHeapWorksWithChangeKey) {
  DynamicBinaryMaxForest<int, int> heap(10, 1);

  heap.push(0, 0, 15);
  heap.push(0, 1, 10);
  heap.push(0, 2, 20);

  heap.change_priority(0, 1, 30);
  EXPECT_EQ(heap.peek_key(0), 30);
  EXPECT_EQ(heap.peek_id(0), 1);

  heap.change_priority(0, 0, 40);
  EXPECT_EQ(heap.peek_key(0), 40);
  EXPECT_EQ(heap.peek_id(0), 0);

  heap.pop(0);
  EXPECT_EQ(heap.peek_key(0), 30);
  EXPECT_EQ(heap.peek_id(0), 1);

  heap.change_priority(0, 2, 31);
  EXPECT_EQ(heap.peek_key(0), 31);
  EXPECT_EQ(heap.peek_id(0), 2);

  heap.pop(0);
  EXPECT_EQ(heap.peek_key(0), 30);
  EXPECT_EQ(heap.peek_id(0), 1);

  heap.pop(0);
  EXPECT_TRUE(heap.empty(0));
}

TEST(DynamicBinaryForestTest, RemoveWorks) {
  DynamicBinaryMaxForest<int, int> heap(10, 1);
  heap.push(0, 0, 15);
  heap.push(0, 1, 10);
  heap.push(0, 2, 20);

  heap.remove(0, 1);
  EXPECT_EQ(heap.size(0), 2);
  EXPECT_EQ(heap.peek_id(0), 2);
  EXPECT_EQ(heap.peek_key(0), 20);

  heap.remove(0, 2);
  EXPECT_EQ(heap.size(0), 1);
  EXPECT_EQ(heap.peek_id(0), 0);
  EXPECT_EQ(heap.peek_key(0), 15);

  heap.remove(0, 0);
  EXPECT_TRUE(heap.empty(0));
}

TEST(DynamicBinaryForestTest, RemoveWorksWithTwoHeaps) {
  DynamicBinaryMaxForest<int, int> heap(10, 2);
  heap.push(0, 0, 10);
  heap.push(1, 1, 11);
  heap.push(0, 2, 12);
  heap.push(1, 3, 13);

  EXPECT_EQ(heap.peek_key(0), 12);
  EXPECT_EQ(heap.peek_key(1), 13);

  heap.remove(1, 3);
  EXPECT_FALSE(heap.contains(3));
  EXPECT_EQ(heap.peek_key(0), 12);
  EXPECT_EQ(heap.peek_key(1), 11);

  heap.remove(1, 1);
  EXPECT_FALSE(heap.contains(1));
  EXPECT_EQ(heap.peek_key(0), 12);
  EXPECT_TRUE(heap.empty(1));

  for (const std::size_t i : {0, 2}) {
    EXPECT_TRUE(heap.contains(i));
  }
  for (const std::size_t i : {1, 3, 4, 5, 6, 7, 8, 9}) {
    EXPECT_FALSE(heap.contains(i));
  }

  heap.remove(0, 0);
  EXPECT_EQ(heap.peek_key(0), 12);

  heap.remove(0, 2);
  EXPECT_TRUE(heap.empty(0));

  for (const std::size_t i : {0, 1, 2, 3, 4, 5, 6, 7, 8, 9}) {
    EXPECT_FALSE(heap.contains(i));
  }
}

TEST(DynamicBinaryMinMaxForestTest, SequenceOfPushAndPopWorks) {
  DynamicBinaryMinMaxForest<int, int> heap(10, 1);

  heap.push(0, 0, 10);
  heap.push(0, 1, 0);
  heap.push(0, 2, 20);
  heap.push(0, 3, 5);
  heap.push(0, 4, 15);

  for (const std::size_t i : {0, 1, 2, 3, 4}) {
    EXPECT_TRUE(heap.contains(i));
  }
  for (const std::size_t i : {5, 6, 7, 8, 9}) {
    EXPECT_FALSE(heap.contains(i));
  }

  EXPECT_EQ(heap.peek_min_key(0), 0);
  EXPECT_EQ(heap.peek_max_key(0), 20);
  heap.pop_min(0);
  EXPECT_FALSE(heap.contains(1));
  EXPECT_EQ(heap.peek_min_key(0), 5);
  EXPECT_EQ(heap.peek_max_key(0), 20);
  heap.pop_max(0);
  EXPECT_FALSE(heap.contains(2));
  EXPECT_EQ(heap.peek_min_key(0), 5);
  EXPECT_EQ(heap.peek_max_key(0), 15);
  heap.pop_min(0);
  EXPECT_FALSE(heap.contains(3));
  EXPECT_EQ(heap.peek_min_key(0), 10);
  EXPECT_EQ(heap.peek_max_key(0), 15);
  heap.pop_max(0);
  EXPECT_FALSE(heap.contains(4));
  EXPECT_EQ(heap.peek_min_key(0), 10);
  EXPECT_EQ(heap.peek_max_key(0), 10);
  heap.pop_min(0);
  EXPECT_TRUE(heap.empty(0));

  for (const std::size_t i : {0, 1, 2, 3, 4, 5, 6, 7, 8, 9}) {
    EXPECT_FALSE(heap.contains(i));
  }
}
} // namespace kaminpar
