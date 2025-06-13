function(add_shm_tool target)
    add_executable(${target} ${ARGN})
    target_link_libraries(${target} PRIVATE KaMinPar::KaMinPar KaMinPar::KaMinParCLI KaMinPar::KaMinParIO)
    target_compile_options(${target} PRIVATE ${KAMINPAR_WARNING_FLAGS})
    set_target_properties(${target} PROPERTIES KASSERT_ASSERTION_LEVEL ${KAMINPAR_ASSERTION_LEVEL_CODE})
    message(STATUS "Enabled tool: ${target}")
endfunction()

# Shared-memory tools
add_shm_tool(shm_connected_components_tool shm_connected_components_tool.cc)
add_shm_tool(shm_graph_compression_tool shm_graph_compression_tool.cc)
add_shm_tool(shm_graph_properties_tool shm_graph_properties_tool.cc)
add_shm_tool(shm_graph_rearrangement_tool shm_graph_rearrangement_tool.cc)
add_shm_tool(shm_partition_properties_tool shm_partition_properties_tool.cc)
