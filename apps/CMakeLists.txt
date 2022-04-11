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
    target_link_libraries(${target} PRIVATE shm_partitioner_base)

    if (KAMINPAR_BACKWARD_CPP)
	target_sources(${target} PUBLIC ${BACKWARD_ENABLE})
        add_backward(${target})
    endif ()

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

add_app(KaMinPar kaminpar.cc)

set(DKAMINPAR_SOURCE_FILES
        ${CMAKE_CURRENT_SOURCE_DIR}/dkaminpar_arguments.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/dkaminpar_arguments.h)

if (KAMINPAR_GRAPHGEN)
    set(DKAMINPAR_SOURCE_FILES
            ${CMAKE_CURRENT_SOURCE_DIR}/dkaminpar_graphgen.cc
            ${CMAKE_CURRENT_SOURCE_DIR}/dkaminpar_graphgen.h
            ${DKAMINPAR_SOURCE_FILES})
endif ()

add_distributed_app(dKaMinPar dkaminpar.cc ${DKAMINPAR_SOURCE_FILES})
if (KAMINPAR_GRAPHGEN)
    target_link_libraries(dKaMinPar PUBLIC kagen_library)
endif ()

add_subdirectory(benchmarks)

