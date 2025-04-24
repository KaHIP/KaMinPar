cmake_minimum_required(VERSION 3.21)

project(
  KaMinParExample
  DESCRIPTION "Example project on how two use the KaMinPar"
  LANGUAGES CXX
)

add_executable(ShmToyGraphExample shm_toy_graph_example.cc)
target_link_libraries(ShmToyGraphExample PRIVATE KaMinPar::KaMinPar)
