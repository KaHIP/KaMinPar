file(GLOB_RECURSE DIST_SOURCE_FILES CONFIGURE_DEPENDS
	*.cc *.h)

find_package(OpenMP REQUIRED)

add_library(dkaminpar ${DIST_SOURCE_FILES})
target_include_directories(dkaminpar PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/../")
target_link_libraries(dkaminpar PUBLIC kaminpar MPI::MPI_CXX OpenMP::OpenMP_CXX growt Sparsehash::Sparsehash)

# If we can find Mt-KaHyPar, make it available as an option for initial partitioning
if (KAMINPAR_BUILD_WITH_MTKAHYPAR)
    find_library(LIB_MTKAHYPAR_GRAPH mtkahypargraph)
    if (NOT LIB_MTKAHYPAR_GRAPH)
        message(STATUS "Mt-KaHyPar initial partitioning not available: library could not be found on this system")
    else ()
        message(STATUS "Found Mt-KaHyPar at ${LIB_MTKAHYPAR_GRAPH}")
        target_compile_definitions(dkaminpar PRIVATE -DKAMINPAR_HAVE_MTKAHYPAR_LIB)
        target_link_libraries(dkaminpar PUBLIC mtkahypar)
    endif ()
endif ()
