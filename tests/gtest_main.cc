#include <gtest/gtest.h>
#include <tbb/global_control.h>

int main(int argc, char **argv) {
  tbb::task_scheduler_handle scheduler{tbb::attach{}};
  ::testing::InitGoogleTest(&argc, argv);
  const int result = RUN_ALL_TESTS();

  // This executable owns the TBB runtime. Join workers before process-wide static destruction.
  tbb::finalize(scheduler);
  return result;
}
