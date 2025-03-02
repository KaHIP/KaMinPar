cmake_minimum_required(VERSION 3.21)

project(
  KaMinParExample
  DESCRIPTION "Example project on how to use KaMinPar"
  LANGUAGES C CXX
)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

option(DOWNLOAD_KAMINPAR "Download KaMinPar from GitHub" ON)
if(DOWNLOAD_KAMINPAR)
  include(FetchContent)
  # TODO: change source when merged
  FetchContent_Declare(
    KaMinPar
    SOURCE_DIR ../../../
  )
  FetchContent_MakeAvailable(KaMinPar)
else()
  find_package(KaMinPar 3.2.0 REQUIRED)
endif()

add_executable(KaMinParExample main.cc)
target_link_libraries(KaMinParExample PRIVATE KaMinPar::KaMinPar KaMinPar::KaMinParIO)
