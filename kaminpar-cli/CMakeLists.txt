set(KAMINPAR_CLI_SOURCE_FILES
    CLI11.h
    kaminpar_arguments.cc
    kaminpar_arguments.h)
set(DKAMINPAR_CLI_SOURCE_FILES
    CLI11.h
    dkaminpar_arguments.cc
    dkaminpar_arguments.h)

add_library(kaminpar_cli ${KAMINPAR_CLI_SOURCE_FILES})
target_include_directories(kaminpar_cli PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/../")
target_link_libraries(kaminpar_cli PUBLIC kaminpar_shm)

if (TARGET kaminpar_dist)
    add_library(kaminpar_dist_cli ${DKAMINPAR_CLI_SOURCE_FILES})
    target_include_directories(kaminpar_dist_cli PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/../")
    target_link_libraries(kaminpar_dist_cli PUBLIC kaminpar_dist)
endif ()
