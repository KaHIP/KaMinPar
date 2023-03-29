file(GLOB_RECURSE DIST_SOURCE_FILES CONFIGURE_DEPENDS
	*.cc *.h)

find_package(OpenMP REQUIRED)

add_library(dist_partitioner_base ${DIST_SOURCE_FILES})
target_include_directories(dist_partitioner_base PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/../")
target_link_libraries(dist_partitioner_base PUBLIC shm_partitioner_base MPI::MPI_CXX OpenMP::OpenMP_CXX growt Sparsehash::Sparsehash)

# If we can find Mt-KaHyPar, make it available as an option for initial partitioning 
find_library(LIB_MTKAHYPAR_GRAPH mtkahypargraph)
if (NOT LIB_MTKAHYPAR_GRAPH)
    message(SYSTEM "Mt-KaHyPar initial partitioning not available: library could not be found on this system") 
else ()
    target_compile_definitions(dist_partitioner_base PRIVATE -DKAMINPAR_HAVE_MTKAHYPAR_LIB)
    target_link_libraries(dist_partitioner_base PUBLIC mtkahypar)
endif ()
