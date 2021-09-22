# taken from http://johnlamp.net/cmake-tutorial-5-functionally-improved-testing.html
function(add_gmock_test target)
    add_executable(${target} ${ARGN})
    target_link_libraries(${target} PRIVATE gtest gmock gmock_main ${CMAKE_THREAD_LIBS_INIT})

    set_property(TARGET ${target} PROPERTY CXX_STANDARD 20)
    set_property(TARGET ${target} PROPERTY CXX_STANDARD_REQUIRED ON)

    add_test(${target} ${target})
endfunction()

function(add_gmock_mpi_test target nproc)
    add_executable(${target} ${ARGN})
    target_link_libraries(${target} PRIVATE gtest gmock ${CMAKE_THREAD_LIBS_INIT})

    set_property(TARGET ${target} PROPERTY CXX_STANDARD 20)
    set_property(TARGET ${target} PROPERTY CXX_STANDARD_REQUIRED 20)

    set(test_parameters -np ${nproc} --oversubscribe "./${target}")
    add_test(NAME ${target} COMMAND "mpirun" ${test_parameters})
endfunction()