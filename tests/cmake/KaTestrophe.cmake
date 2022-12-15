if (NOT DEFINED KATESTROPHE_INCLUDED)
    set(KATESTROPHE_INCLUDED TRUE)

    # sets the provided output variable KAMPING_OVERSUBSCRIBE_FLAG to the flags required to run mpiexec with
    # more MPI ranks than cores available
    function(katestrophe_has_oversubscribe KATESTROPHE_OVERSUBSCRIBE_FLAG)
        string(FIND ${MPI_CXX_LIBRARY_VERSION_STRING} "OpenMPI" SEARCH_POSITION1)
        string(FIND ${MPI_CXX_LIBRARY_VERSION_STRING} "Open MPI" SEARCH_POSITION2)
        # only Open MPI seems to require the --oversubscribe flag
        # MPICH and Intel don't know it but silently run commands with more ranks than cores available
        if (${SEARCH_POSITION1} EQUAL -1 AND ${SEARCH_POSITION2} EQUAL -1)
            set("${KATESTROPHE_OVERSUBSCRIBE_FLAG}" "" PARENT_SCOPE)
        else ()
            # We are using Open MPI
            set("${KATESTROPHE_OVERSUBSCRIBE_FLAG}" "--oversubscribe" PARENT_SCOPE)
        endif ()
    endfunction()
    katestrophe_has_oversubscribe(MPIEXEC_OVERSUBSCRIBE_FLAG)

    # register the test main class
    add_library(mpi-gtest-main EXCLUDE_FROM_ALL mpi_gtest_main.cc)
    target_link_libraries(mpi-gtest-main PUBLIC MPI::MPI_CXX GTest::gtest GTest::gmock)

    # keep the cache clean
    mark_as_advanced(
            BUILD_GMOCK BUILD_GTEST BUILD_SHARED_LIBS
            gmock_build_tests gtest_build_samples gtest_build_tests
            gtest_disable_pthreads gtest_force_shared_crt gtest_hide_internal_symbols
    )

    # Adds an executable target with the specified files FILES and links gtest and the MPI gtest runner
    #
    # KATESTROPHE_TARGET target name
    # FILES the files to include in the target
    #
    # example: katestrophe_add_test_executable(mytarget FILES mytarget.cpp myotherfile.cpp)
    function(katestrophe_add_test_executable KATESTROPHE_TARGET)
        cmake_parse_arguments(
                "KATESTROPHE"
                ""
                ""
                "FILES"
                ${ARGN}
        )
        add_executable(${KATESTROPHE_TARGET} "${KATESTROPHE_FILES}")
        target_link_libraries(${KATESTROPHE_TARGET} PUBLIC mpi-gtest-main)
    endfunction()

    # Registers an executable target KATESTROPHE_TEST_TARGET as a test to be executed with ctest
    #
    # KATESTROPHE_TEST_TARGET target name
    # DISCOVER_TESTS sets whether the individual tests should be added to the ctest output (like gtest_discover_tests)
    # CORES the number of MPI ranks to run the test with
    #
    # example: katestrophe_add_mpi_test(mytest CORES 2 4 8)
    function(katestrophe_add_mpi_test KATESTROPHE_TEST_TARGET)
        cmake_parse_arguments(
                KATESTROPHE
                ""
                ""
                "CORES"
                ${ARGN}
        )
        if (NOT KATESTROPHE_CORES)
            set(KATESTROPHE_CORES ${MPIEXEC_MAX_NUMPROCS})
        endif ()
        foreach (p ${KATESTROPHE_CORES})
            set(TEST_NAME "${KATESTROPHE_TEST_TARGET}.${p}cores")
            set(MPI_EXEC_COMMAND ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} ${p} ${MPIEXEC_OVERSUBSCRIBE_FLAG} ${MPIEXEC_PREFLAGS})
            add_test(
                NAME "${TEST_NAME}"
                COMMAND ${MPI_EXEC_COMMAND} $<TARGET_FILE:${KATESTROPHE_TEST_TARGET}>
                WORKING_DIRECTORY ${PROJECT_SOURCE_DIR})
        endforeach ()
    endfunction()
endif ()
