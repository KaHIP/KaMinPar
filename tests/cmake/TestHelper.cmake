include(KaTestrophe)
include(GoogleTest)

function(kaminpar_add_dist_test KAMINPAR_TARGET_NAME)
    cmake_parse_arguments(
            "KAMINPAR"
            ""
            ""
            "FILES;CORES"
            ${ARGN}
    )
    katestrophe_add_test_executable(${KAMINPAR_TARGET_NAME} FILES ${KAMINPAR_FILES})
    target_link_libraries(${KAMINPAR_TARGET_NAME} PRIVATE common_base shm_partitioner_base dist_partitioner_base)
    target_include_directories(${KAMINPAR_TARGET_NAME} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
    if (KAMINPAR_BACKWARD_CPP)
        add_backward(${KAMINPAR_TARGET_NAME})
    endif ()
    katestrophe_add_mpi_test(${KAMINPAR_TARGET_NAME} CORES ${KAMINPAR_CORES} DISCOVER_TESTS)
endfunction()

function(kaminpar_add_shm_test target)
    add_gmock_test(${target} ${ARGN})
    target_link_libraries(${target} PRIVATE common_base shm_partitioner_base)
    target_include_directories(${target} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
endfunction()

function(kaminpar_add_common_test target)
    add_gmock_test(${target} ${ARGN})
    target_link_libraries(${target} PRIVATE common_base)
    target_include_directories(${target} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
endfunction()
