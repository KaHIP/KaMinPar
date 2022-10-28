add_library(kaminpar kaminpar.cc)
set_target_properties(kaminpar PROPERTIES PUBLIC_HEADER "kaminpar.h")
target_include_directories(kaminpar PUBLIC ${CMAKE_CURRENT_SOURCE_DIR} ${PROJECT_BINARY_DIR}/library)
target_link_libraries(kaminpar PRIVATE common_base shm_partitioner_base)

if (KAMINPAR_BUILD_METIS_INTERFACE)
    target_sources(kaminpar PUBLIC metis_adapter.cc)
endif ()

install(TARGETS kaminpar EXPORT kaminpar
        PUBLIC_HEADER DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}"
        LIBRARY DESTINATION "${CMAKE_INSTALL_LIBDIR}")

add_subdirectory(example)

if (KAMINPAR_BUILD_PYTHON_MODULE)
    message(STATUS "Building the Python module")
    pybind11_add_module(pykaminpar kaminpar.cc kaminpar.h pykaminpar.cc)
    target_link_libraries(pykaminpar PUBLIC common_base shm_partitioner_base)
endif ()
