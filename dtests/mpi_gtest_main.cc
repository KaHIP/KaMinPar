#include <gmock/gmock.h>
#include <mpi.h>

#ifdef USE_BACKWARD
#include <backward.hpp>
#endif // USE_BACKWARD

int main(int argc, char *argv[]) {
  ::testing::InitGoogleTest(&argc, argv);
  MPI_Init(&argc, &argv);

#ifdef USE_BACKWARD
  auto handler = backward::SignalHandling{};
#endif // USE_BACKWARD

  auto result = RUN_ALL_TESTS();

  MPI_Finalize();
  return result;
}