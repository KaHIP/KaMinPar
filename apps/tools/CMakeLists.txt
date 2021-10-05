set(LIB_TOOLS_SOURCE_FILES
        graph_tools.cc
        graph_tools.h
        graph_builder.cc
        graph_builder.h
        csv.cc
        csv.h)

add_library(tools OBJECT ${LIB_TOOLS_SOURCE_FILES})
target_link_libraries(tools PUBLIC kaminpar_impl)

if (KAMINPAR_USE_BACKWARD_CPP)
    add_backward(tools)
endif ()

function(add_tool target)
    add_executable(${target} ${ARGN} "${CMAKE_CURRENT_BINARY_DIR}/../environment.cc")
    target_link_libraries(${target} PRIVATE kaminpar_impl tools)
    target_include_directories(${target} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})

    if (KAMINPAR_USE_BACKWARD_CPP)
        add_backward(${target})
    endif ()
endfunction()

add_tool(VerifyPartition verify_partition.cc)
add_tool(GraphStatistics graph_statistics.cc)
add_tool(GraphConverter graph_converter.cc converter/mmio.cc)
add_tool(GraphChecker graph_checker.cc)
add_tool(CsvViewer csv_viewer.cc)
