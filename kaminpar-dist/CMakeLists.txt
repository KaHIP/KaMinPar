file(GLOB_RECURSE KAMINPAR_DIST_SOURCE_FILES CONFIGURE_DEPENDS
	*.cc *.h)

add_library(kaminpar_dist ${KAMINPAR_DIST_SOURCE_FILES})
target_include_directories(kaminpar_dist PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/../")
target_link_libraries(kaminpar_dist PUBLIC kaminpar_shm kaminpar_mpi growt Sparsehash::Sparsehash)

# If we can find Mt-KaHyPar, make it available as an option for initial partitioning
if (KAMINPAR_BUILD_WITH_MTKAHYPAR)
    target_compile_definitions(kaminpar_dist PRIVATE -DKAMINPAR_HAVE_MTKAHYPAR_LIB)
    target_link_libraries(kaminpar_dist PUBLIC mtkahypar)
endif ()
