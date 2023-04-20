# Fetch minimal KaGen for graph IO
set(KAGEN_NODEPS ON)
set(KAGEN_BUILD_APPS OFF)
set(KAGEN_BUILD_EXAMPLES OFF)
FetchContent_MakeAvailable(KaGen)

function(add_benchmark target)
    add_executable(${target} ${ARGN})
    target_link_libraries(${target} PRIVATE 
        KaMinPar::KaMinPar 
        KaMinPar::dKaMinPar 
        KaMinPar::KaMinParCLI11 
        KaMinPar::dKaMinParCLI11 
        KaGen::KaGen)
    message(STATUS "Enabled benchmark: ${target}")
endfunction()

# Shared-memory benchmarks
add_benchmark(shm_refinement_benchmark shm_refinement_benchmark.cc)


# @todo update these benchmarks as we need them
#add_benchmark(balancing_benchmark balancing_benchmark.cc)
#add_distributed_benchmark(dist_label_propagation_benchmark dist_label_propagation_benchmark.cc)
#add_distributed_benchmark(dist_coloring_benchmark dist_coloring_benchmark.cc)
#add_distributed_benchmark(dist_flat_refinement_benchmark dist_flat_refinement_benchmark.cc)
#add_distributed_benchmark(dist_contraction_benchmark dist_contraction_benchmark.cc)
#add_distributed_benchmark(dist_balancing_benchmark dist_balancing_benchmark.cc)
#add_distributed_benchmark(dist_coarsening_benchmark dist_coarsening_benchmark.cc)

