option(KAMINPAR_BUILD_METIS_INTERFACE Off)

add_library(kaminpar libkaminpar.cc libkaminpar.h)
target_include_directories(kaminpar PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(kaminpar PRIVATE kaminpar_impl)

if (KAMINPAR_BUILD_METIS_INTERFACE)
  target_sources(kaminpar PUBLIC metis_adapter.cc)
endif ()

add_subdirectory(example)