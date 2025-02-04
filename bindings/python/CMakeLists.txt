cmake_minimum_required(VERSION 3.21)

project(
  ${SKBUILD_PROJECT_NAME}
  VERSION ${SKBUILD_PROJECT_VERSION}
  DESCRIPTION "Python Bindings for KaMinPar"
  LANGUAGES CXX
)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(CMAKE_CXX_VISIBILITY_PRESET hidden)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

find_package(KaMinPar QUIET)
if(NOT TARGET KaMinPar::KaMinPar)
  include(FetchContent)

  # TODO: Update when the branch is merged
  FetchContent_Declare(
    KaMinPar
    GIT_REPOSITORY https://github.com/dsalwasser/KaMinPar.git
    GIT_TAG shm/feat/graph-compression-interface
    GIT_SHALLOW TRUE
    UPDATE_DISCONNECTED TRUE
  )
  set(KAMINPAR_BUILD_TESTS OFF CACHE BOOL "" FORCE)
  FetchContent_MakeAvailable(KaMinPar)
endif()

set(PYBIND11_FINDPYTHON ON)
find_package(pybind11 CONFIG REQUIRED)

pybind11_add_module(kaminpar_python src/PyKaMinPar.cpp)
target_compile_definitions(kaminpar_python PRIVATE VERSION_INFO=${PROJECT_VERSION})
target_link_libraries(kaminpar_python PUBLIC KaMinPar::KaMinPar KaMinPar::KaMinParIO)

add_library(KaMinPar::PyKaMinPar ALIAS kaminpar_python)
install(TARGETS kaminpar_python DESTINATION kaminpar)
