cmake_minimum_required(VERSION 3.21)

project(
  ${SKBUILD_PROJECT_NAME}
  VERSION ${SKBUILD_PROJECT_VERSION}
  DESCRIPTION "NetworKit Bindings for KaMinPar"
  LANGUAGES CXX
)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(Python COMPONENTS Interpreter Development.Module REQUIRED)
find_package(Cython MODULE REQUIRED VERSION 3.0)

include(UseCython)
include(FetchContent)

FetchContent_Declare(
  networkit
  GIT_REPOSITORY https://github.com/networkit/networkit.git
  GIT_TAG f720ba9678943c771ed04e540946a9ea1f221fd5
)
FetchContent_MakeAvailable(networkit)

FetchContent_Declare(
  KaMinPar
  GIT_REPOSITORY https://github.com/dsalwasser/KaMinPar.git
  GIT_TAG shm/feat/graph-compression-interface
)
set(KAMINPAR_BUILD_TESTS OFF CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(KaMinPar)

cython_transpile(src/kaminpar_networkit/nkkaminpar.pyx
  LANGUAGE CXX
  OUTPUT_VARIABLE nkkaminpar_cpp
)

python_add_library(nkkaminpar MODULE "${nkkaminpar_cpp}" src/kaminpar_networkit.h src/kaminpar_networkit.cc WITH_SOABI)
target_link_libraries(nkkaminpar PUBLIC KaMinPar::KaMinPar networkit)
target_include_directories(nkkaminpar PUBLIC "${networkit_SOURCE_DIR}/networkit" "src/")

add_library(KaMinPar::NetworKitKaMinPar ALIAS nkkaminpar)
install(TARGETS nkkaminpar DESTINATION kaminpar_networkit)
