# Make commit hash available
include(GetGitRevisionDescription)
get_git_head_revision(GIT_REFSPEC GIT_SHA1)

# Make hostname available
include(GetHostname)
GetHostname(HOSTNAME)

# Make list of modified files available
include(GetModifiedGitFiles)
GetModifiedGitFiles(MODIFIED_FILES)

configure_file("${CMAKE_CURRENT_SOURCE_DIR}/environment.cc.in" "${CMAKE_CURRENT_BINARY_DIR}/environment.cc" @ONLY)

function(add_app target)
    add_executable(${target} ${ARGN} "${CMAKE_CURRENT_BINARY_DIR}/environment.cc")
    target_include_directories(${target} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
    target_link_libraries(${target} PRIVATE common_base shm_partitioner_base)
    install(TARGETS ${target})
    message(STATUS "App ${target}")
endfunction()

function(add_distributed_app target)
    if (TARGET dist_partitioner_base)
        add_app(${target} ${ARGN})
        target_link_libraries(${target} PRIVATE dist_partitioner_base)
    else ()
        message(STATUS "Cannot add distributed app: distributed library is not available, presumably because MPI"
                " could not be found.")
    endif ()
endfunction()

# KaMinPar
add_app(KaMinPar kaminpar.cc)

# dKaMinPar
set(DKAMINPAR_SOURCE_FILES
        ${CMAKE_CURRENT_SOURCE_DIR}/dkaminpar/arguments.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/dkaminpar/arguments.h)
if (KAMINPAR_ENABLE_GRAPHGEN)
    set(DKAMINPAR_SOURCE_FILES
            ${CMAKE_CURRENT_SOURCE_DIR}/dkaminpar/graphgen.cc
            ${CMAKE_CURRENT_SOURCE_DIR}/dkaminpar/graphgen.h
            ${DKAMINPAR_SOURCE_FILES})
endif ()

add_distributed_app(dKaMinPar dkaminpar.cc ${DKAMINPAR_SOURCE_FILES})
if (KAMINPAR_ENABLE_GRAPHGEN)
    target_link_libraries(dKaMinPar PRIVATE KaGen::KaGen) 
endif ()

# Benchmarks
# add_subdirectory(benchmarks)

