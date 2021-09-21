# Make commit hash available
include(GetGitRevisionDescription)
get_git_head_revision(GIT_REFSPEC GIT_SHA1)
#add_definitions(-DGIT_COMMIT_HASH="${GIT_SHA1}")

# Make hostname available
include(GetHostname)
GetHostname(HOSTNAME)
#add_definitions(-DHOSTNAME="${HOSTNAME}")

# Make list of modified files available
include(GetModifiedGitFiles)
GetModifiedGitFiles(MODIFIED_FILES)
#add_definitions(-DMODIFIED_FILES="${MODIFIED_FILES}")

configure_file("${CMAKE_CURRENT_SOURCE_DIR}/environment.cc.in" "${CMAKE_CURRENT_BINARY_DIR}/environment.cc" @ONLY)

function(add_app target)
    add_executable(${target} ${ARGN} "${CMAKE_CURRENT_BINARY_DIR}/environment.cc")
    target_include_directories(${target} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
    target_link_libraries(${target} PRIVATE kaminpar_impl)
    if (KAMINPAR_USE_BACKWARD_CPP)
        target_compile_definitions(${target} PRIVATE -DUSE_BACKWARD)
        add_backward(${target})
    endif ()
    install(TARGETS ${target})
    message(STATUS "App ${target}")
endfunction()

function(add_distributed_app target)
    if (TARGET dkaminpar_impl)
        add_app(${target} ${ARGN})
        target_link_libraries(${target} PRIVATE dkaminpar_impl)
    else ()
        message(STATUS "Cannot add distributed app: distributed library is not available, presumably because MPI"
                " could not be found.")
    endif ()
endfunction()

add_app(KaMinPar kaminpar.cc)
add_distributed_app(dKaMinPar dkaminpar.cc)

add_subdirectory(tools)
add_subdirectory(benchmarks)
