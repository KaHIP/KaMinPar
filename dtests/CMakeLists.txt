file(COPY test_instances DESTINATION ${CMAKE_CURRENT_BINARY_DIR})

function(add_kaminpar_dtest target nproc)
    if (TARGET dkaminpar_impl)
        add_gmock_mpi_test(${target} ${nproc} mpi_gtest_main.cc ${ARGN})
        target_include_directories(${target} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
        target_link_libraries(${target} PRIVATE kaminpar_impl dkaminpar_impl)

        if (KAMINPAR_USE_BACKWARD_CPP)
            add_backward(${target})
        endif ()
    else ()
        message(STATUS "Cannot add distributed test: distributed library is not available, presumably because MPI"
                " could not be found.")
    endif ()
endfunction()

add_kaminpar_dtest(local_contraction_test 3 algorithm/local_contraction_test.cc)
add_kaminpar_dtest(locking_label_propagation_clustering_test 3 coarsening/locking_label_propagation_clustering_test.cc)
#add_kaminpar_dtest(global_contraction_test 3 algorithm/global_contraction_test.cc)