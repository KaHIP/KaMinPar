#include <gmock/gmock.h>
#include <mpi.h>

#ifdef KAMINPAR_BACKWARD_CPP
#include <backward.hpp>
#endif // KAMINPAR_BACKWARD_CPP

int main(int argc, char *argv[]) {
  ::testing::InitGoogleTest(&argc, argv);
  MPI_Init(&argc, &argv);

#ifdef KAMINPAR_BACKWARD_CPP
  auto handler = backward::SignalHandling{};
#endif // KAMINPAR_BACKWARD_CPP

  auto result = RUN_ALL_TESTS();

  MPI_Finalize();
  return result;
}