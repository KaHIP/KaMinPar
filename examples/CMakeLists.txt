add_executable(ShmToyGraphExample shm_toy_graph_example.cc)
target_link_libraries(ShmToyGraphExample PRIVATE KaMinPar::KaMinPar)

add_executable(ShmCToyGraphExample shm_c_toy_graph_example.c)
target_link_libraries(ShmCToyGraphExample PRIVATE KaMinPar::KaMinPar)

if (TARGET kaminpar_networkit)
    add_executable(ShmNetworkitExample shm_networkit_example.cc)
    target_link_libraries(ShmNetworkitExample PRIVATE kaminpar_networkit)
endif ()
