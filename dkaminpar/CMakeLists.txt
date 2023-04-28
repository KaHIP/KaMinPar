file(GLOB_RECURSE DIST_SOURCE_FILES CONFIGURE_DEPENDS
	*.cc *.h)

find_package(OpenMP REQUIRED)

add_library(dkaminpar ${DIST_SOURCE_FILES})
target_include_directories(dkaminpar PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/../")
target_link_libraries(dkaminpar PUBLIC kaminpar MPI::MPI_CXX OpenMP::OpenMP_CXX growt Sparsehash::Sparsehash)

# If we can find Mt-KaHyPar, make it available as an option for initial partitioning
if (KAMINPAR_BUILD_WITH_MTKAHYPAR)
    target_compile_definitions(dkaminpar PRIVATE -DKAMINPAR_HAVE_MTKAHYPAR_LIB)
    target_link_libraries(dkaminpar PUBLIC mtkahypar)
endif ()
