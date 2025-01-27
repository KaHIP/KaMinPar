cmake_minimum_required(VERSION 3.21)

project(
  CKaMinPar
  VERSION 3.1.0
  DESCRIPTION "C Bindings for KaMinPar"
  LANGUAGES C CXX
)

add_library(kaminpar_c src/ckaminpar.cc)
add_library(KaMinPar::CKaMinPar ALIAS kaminpar_c)

target_include_directories(kaminpar_c PUBLIC "\$<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>")
target_link_libraries(kaminpar_c PRIVATE KaMinPar::KaMinPar)
