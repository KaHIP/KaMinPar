set(KAMINPAR_CLI11_SOURCE_FILES 
    CLI11.h
    kaminpar_arguments.cc
    kaminpar_arguments.h)
set(DKAMINPAR_CLI11_SOURCE_FILES 
    CLI11.h
    dkaminpar_arguments.cc
    dkaminpar_arguments.h)

add_library(kaminpar_cli11 ${KAMINPAR_CLI11_SOURCE_FILES})
target_include_directories(kaminpar_cli11 PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/../")
target_link_libraries(kaminpar_cli11 PUBLIC kaminpar)

if (TARGET dkaminpar)
    add_library(dkaminpar_cli11 ${DKAMINPAR_CLI11_SOURCE_FILES})
    target_include_directories(dkaminpar_cli11 PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/../")
    target_link_libraries(dkaminpar_cli11 PUBLIC dkaminpar)
endif ()
