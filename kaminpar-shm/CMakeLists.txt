file(GLOB_RECURSE KAMINPAR_SHM_SOURCE_FILES CONFIGURE_DEPENDS
	*.cc *.h)

add_library(kaminpar_shm ${KAMINPAR_SHM_SOURCE_FILES}
        coarsening/sparsification/Sampler.h
        coarsening/sparsification/UniformRandomSampler.cpp
        coarsening/sparsification/UniformRandomSampler.h
        coarsening/sparsifing_cluster_coarsener.h
        coarsening/sparsification/networkit_utils.cpp
        coarsening/sparsification/networkit_utils.h
        coarsening/sparsification/SparsificationTarget.h
        coarsening/sparsification/DensitySparsificationTarget.h
        coarsening/sparsification/EdgeReductionSparsificationTarget.h
        coarsening/sparsification/kNeighbourSampler.cpp
        coarsening/sparsification/kNeighbourSampler.h
        coarsening/sparsification/UnionFind.cpp
        coarsening/sparsification/UnionFind.h
        coarsening/sparsification/sparsification_utils.cpp
        coarsening/sparsification/sparsification_utils.h
        coarsening/sparsification/ThresholdSampler.h
        coarsening/sparsification/ScoreBacedSampler.h
        coarsening/sparsification/NetworKitScoreAdapter.cpp
        coarsening/sparsification/NetworKitScoreAdapter.h
        coarsening/sparsification/EffectiveResistanceSampler.cpp)
target_include_directories(kaminpar_shm PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/../")
target_link_libraries(kaminpar_shm PUBLIC kaminpar_common networkit)

# If we can find Mt-KaHyPar, make it available as an option for refinement
if (KAMINPAR_BUILD_WITH_MTKAHYPAR)
    target_compile_definitions(kaminpar_shm PRIVATE -DKAMINPAR_HAVE_MTKAHYPAR_LIB)
    target_link_libraries(kaminpar_shm PUBLIC mtkahypar)
endif ()
