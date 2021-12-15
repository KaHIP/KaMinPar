function(add_app target)
  add_executable(${target} ${ARGN})
  target_include_directories(${target} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
  target_link_libraries(${target} PRIVATE kaminpar_impl)
  message(STATUS "App ${target}")
endfunction()

add_app(KaMinPar kaminpar.cc)