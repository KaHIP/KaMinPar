include(KaTestrophe)
include(GoogleTest)

# Convenience wrapper for adding tests for dKaMinPar
# this creates the target, links googletest and dkaminpar, enables warnings and registers the test
#
# TARGET_NAME the target name
# FILES the files of the target
#
# example: dkaminpar_register_test(mytarget FILES mytarget.cpp)
function(dkaminpar_register_test KAMINPAR_TARGET_NAME)
    cmake_parse_arguments(
            "KAMINPAR"
            ""
            ""
            "FILES"
            ${ARGN}
    )
    add_executable(${KAMINPAR_TARGET_NAME} ${KAMINPAR_FILES})
    target_link_libraries(${KAMINPAR_TARGET_NAME} PRIVATE gtest gtest_main gmock kaminpar common dkaminpar)
    target_include_directories(${KAMINPAR_TARGET_NAME} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
    gtest_discover_tests(${KAMINPAR_TARGET_NAME} WORKING_DIRECTORY ${PROJECT_DIR})
endfunction()

# Convenience wrapper for adding tests for KaMPI.ng which rely on MPI
# this creates the target, links googletest, kamping and MPI, enables warnings and registers the tests
#
# TARGET_NAME the target name
# FILES the files of the target
# CORES the number of MPI ranks to run the test for
#
# example: dkaminpar_register_mpi_test(mytarget FILES mytarget.cpp CORES 1 2 4 8)
function(dkaminpar_register_mpi_test KAMINPAR_TARGET_NAME)
    cmake_parse_arguments(
            "KAMINPAR"
            ""
            ""
            "FILES;CORES"
            ${ARGN}
    )
    katestrophe_add_test_executable(${KAMINPAR_TARGET_NAME} FILES ${KAMINPAR_FILES})
    target_link_libraries(${KAMINPAR_TARGET_NAME} PRIVATE common kaminpar dkaminpar)
    target_include_directories(${KAMINPAR_TARGET_NAME} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
    katestrophe_add_mpi_test(${KAMINPAR_TARGET_NAME} CORES ${KAMINPAR_CORES} DISCOVER_TESTS)
endfunction()

# Convenience wrapper for registering a set of tests that should fail to compile and require KaMPI.ng to be linked.
#
# TARGET prefix for the targets to be built
# FILES the list of files to include in the target
# SECTIONS sections of the compilation test to build
#
function(dkaminpar_register_compilation_failure_test KAMINPAR_TARGET_NAME)
    cmake_parse_arguments(
            "KAMINPAR"
            ""
            ""
            "FILES;SECTIONS"
            ${ARGN}
    )
    katestrophe_add_compilation_failure_test(
            TARGET ${KAMINPAR_TARGET_NAME}
            FILES ${KAMINPAR_FILES}
            SECTIONS ${KAMINPAR_SECTIONS}
            LIBRARIES common kaminpar dkaminpar
    )
endfunction()
