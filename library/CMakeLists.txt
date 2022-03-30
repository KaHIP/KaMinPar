include(GenerateExportHeader)

option(KAMINPAR_BUILD_METIS_INTERFACE Off)

add_library(kaminpar kaminpar.cc)
set_target_properties(kaminpar PROPERTIES PUBLIC_HEADER "kaminpar.h;${PROJECT_BINARY_DIR}/library/kaminpar_export.h")
target_include_directories(kaminpar PUBLIC ${CMAKE_CURRENT_SOURCE_DIR} ${PROJECT_BINARY_DIR}/library)
target_link_libraries(kaminpar PRIVATE shm_partitioner_base)
generate_export_header(kaminpar)

if (KAMINPAR_BUILD_METIS_INTERFACE)
    target_sources(kaminpar PUBLIC metis_adapter.cc)
endif ()

install(TARGETS kaminpar EXPORT kaminpar
        PUBLIC_HEADER DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}"
        LIBRARY DESTINATION "${CMAKE_INSTALL_LIBDIR}")

add_subdirectory(example)
