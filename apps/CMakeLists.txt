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
    message(STATUS "Enabled app: ${target}")
endfunction()

function(add_distributed_app target)
    if (TARGET dist_partitioner_base)
        add_app(${target} ${ARGN})
        target_link_libraries(${target} PRIVATE dist_partitioner_base)
    else ()
        message(STATUS "Disabled app: ${target}")
    endif ()
endfunction()

add_app(KaMinPar KaMinPar.cc)
add_distributed_app(dKaMinPar dKaMinPar.cc)
