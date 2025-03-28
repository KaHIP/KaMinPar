file(GLOB_RECURSE MPI_SOURCE_FILES CONFIGURE_DEPENDS
        *.cc *.h)

add_library(KaMinParMPI OBJECT ${MPI_SOURCE_FILES})
add_library(KaMinPar::KaMinParMPI ALIAS KaMinParMPI)

target_link_libraries(KaMinParMPI PUBLIC KaMinParCommon MPI::MPI_CXX)
target_compile_options(KaMinParMPI PRIVATE ${KAMINPAR_WARNING_FLAGS})
