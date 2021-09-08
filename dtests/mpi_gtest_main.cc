#include <gmock/gmock.h>
#include <mpi.h>

int main(int argc, char *argv[]) {
  ::testing::InitGoogleTest(&argc, argv);
  MPI_Init(&argc, &argv);
  auto result = RUN_ALL_TESTS();
  MPI_Finalize();
  return result;
}