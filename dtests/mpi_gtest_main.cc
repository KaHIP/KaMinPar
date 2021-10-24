#include <gmock/gmock.h>
#include <mpi.h>

int main(int argc, char *argv[]) {
  ::testing::InitGoogleTest(&argc, argv);
  MPI_Init(&argc, &argv);
  auto result = RUN_ALL_TESTS();

  // report incoming messages after tests have ended
  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
//  MPI_Status st;
//  int flag;
//  MPI_Iprobe(MPI_ANY_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD, &flag, &st);
//  while (flag) {
//    std::cout << "attention: still incoming messages! rank " << rank << " from " << st.MPI_SOURCE << std::endl;
//    int message_length;
//    MPI_Get_count(&st, MPI_INT, &message_length);
//    MPI_Status rst;
//    std::vector<int> message;
//    message.resize(message_length);
//    MPI_Recv(&message[0], message_length, MPI_UNSIGNED_LONG_LONG, st.MPI_SOURCE, st.MPI_TAG, MPI_COMM_WORLD, &rst);
//    MPI_Iprobe(MPI_ANY_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD, &flag, &st);
//  }

  MPI_Finalize();
  return result;
}