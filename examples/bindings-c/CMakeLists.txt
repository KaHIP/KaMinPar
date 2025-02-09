cmake_minimum_required(VERSION 3.21)

project(
  KaMinParCBindingsExample
  DESCRIPTION "Example project on how two use the C bindings for KaMinPar"
  LANGUAGES C
)

add_executable(ShmCToyGraphExample shm_c_toy_graph_example.c)
target_link_libraries(ShmCToyGraphExample PRIVATE KaMinPar::KaMinPar)
