add_library(kaminpar libkaminpar.cc libkaminpar.h)
target_include_directories(kaminpar PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(kaminpar PRIVATE kaminpar_impl)