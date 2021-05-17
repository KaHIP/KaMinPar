function(add_app target)
  add_executable(${target} ${ARGN})
  target_include_directories(${target} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
  target_link_libraries(${target} PUBLIC kaminpar)
  message(STATUS "App ${target}")
  if (UNIX AND NOT APPLE)
    message(STATUS "Link -lnuma")
    target_link_libraries(${target} PUBLIC numa)
  endif ()
endfunction()

add_app(KaMinPar kaminpar.cc)

add_subdirectory(tools)