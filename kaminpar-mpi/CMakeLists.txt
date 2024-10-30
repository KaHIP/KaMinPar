file(GLOB_RECURSE MPI_SOURCE_FILES CONFIGURE_DEPENDS
        *.cc *.h)

add_library(kaminpar_mpi OBJECT ${MPI_SOURCE_FILES}
        definitions.h)
target_include_directories(kaminpar_mpi PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/../")
target_compile_options(kaminpar_mpi PRIVATE ${KAMINPAR_WARNING_FLAGS})

find_library(NUMA_LIB numa) # optional

target_link_libraries(kaminpar_mpi PUBLIC kaminpar_common MPI::MPI_CXX)
