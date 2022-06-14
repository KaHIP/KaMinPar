file(GLOB_RECURSE DIST_SOURCE_FILES CONFIGURE_DEPENDS
	*.cc *.h)

find_package(OpenMP REQUIRED)

add_library(dist_partitioner_base OBJECT ${DIST_SOURCE_FILES})
target_include_directories(dist_partitioner_base PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(dist_partitioner_base PUBLIC common_base shm_partitioner_base MPI::MPI_CXX OpenMP::OpenMP_CXX growt)
