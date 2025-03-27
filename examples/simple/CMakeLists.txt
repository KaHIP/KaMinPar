cmake_minimum_required(VERSION 3.21)

project(
  KaMinParExample
  DESCRIPTION "Example project on how to use KaMinPar"
  LANGUAGES C CXX
)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(KaMinPar QUIET)
if(NOT KaMinPar_FOUND)
  message(STATUS "KaMinPar not found, downloading it from GitHub")

  include(FetchContent)
  FetchContent_Declare(
    KaMinPar
    GIT_REPOSITORY https://github.com/KaHIP/KaMinPar.git
    GIT_TAG main
    GIT_SHALLOW TRUE
    UPDATE_DISCONNECTED TRUE
  )
  set(KAMINPAR_BUILD_APPS OFF CACHE BOOL "" FORCE)
  FetchContent_MakeAvailable(KaMinPar)
endif()

add_executable(KaMinParExample main.cc)
target_link_libraries(KaMinParExample PRIVATE KaMinPar::KaMinPar KaMinPar::KaMinParIO)
