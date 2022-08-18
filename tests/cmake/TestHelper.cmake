include(KaTestrophe)
include(GoogleTest)

# TARGET_NAME the target name
# FILES the files of the target
function(kaminpar_add_dist_test KAMINPAR_TARGET_NAME)
    cmake_parse_arguments(
            "KAMINPAR"
            ""
            ""
            "FILES"
            ${ARGN}
    )

    if (TARGET dist_partitioner_base)
        add_executable(${KAMINPAR_TARGET_NAME} ${KAMINPAR_FILES})
        target_link_libraries(${KAMINPAR_TARGET_NAME} PRIVATE gtest gtest_main gmock shm_partitioner_base common_base dist_partitioner_base)
        target_include_directories(${KAMINPAR_TARGET_NAME} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
        gtest_discover_tests(${KAMINPAR_TARGET_NAME} WORKING_DIRECTORY ${PROJECT_DIR})
    else ()
        message(STATUS "Not building unit test ${KAMINPAR_TARGET_NAME}: depends on distributed graph partitioner")
    endif ()
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
