function(add_app target)
    add_executable(${target} ${ARGN})
    target_include_directories(${target} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
    target_link_libraries(${target} PRIVATE kaminpar_impl)
    message(STATUS "App ${target}")
endfunction()

function(add_distributed_app target)
    if (TARGET dkaminpar_impl)
        add_executable(${target} ${ARGN})
        target_include_directories(${target} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
        target_link_libraries(${target} PRIVATE dkaminpar_impl)
        message(STATUS "Distributed app ${target}")
    else ()
        message(STATUS "Cannot add distributed app: distributed library is not available, presumably because MPI"
                " could not be found.")
    endif ()
endfunction()

add_app(KaMinPar kaminpar.cc)
add_distributed_app(dKaMinPar dkaminpar.cc)

add_subdirectory(tools)