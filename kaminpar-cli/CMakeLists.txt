set(KAMINPAR_CLI_SOURCE_FILES
    CLI11.h
    kaminpar_arguments.h kaminpar_arguments.cc)

set(DKAMINPAR_CLI_SOURCE_FILES
    CLI11.h
    dkaminpar_arguments.h dkaminpar_arguments.cc)

add_library(KaMinParCLI ${KAMINPAR_CLI_SOURCE_FILES})
add_library(KaMinPar::KaMinParCLI ALIAS KaMinParCLI)

target_link_libraries(KaMinParCLI PUBLIC KaMinPar::KaMinPar)
target_compile_options(KaMinParCLI PRIVATE ${KAMINPAR_WARNING_FLAGS})

if (TARGET dKaMinPar)
    add_library(dKaMinParCLI ${DKAMINPAR_CLI_SOURCE_FILES})
    add_library(KaMinPar::dKaMinParCLI ALIAS dKaMinParCLI)

    target_link_libraries(dKaMinParCLI PUBLIC KaMinPar::dKaMinPar)
    target_compile_options(dKaMinParCLI PRIVATE ${KAMINPAR_WARNING_FLAGS})
endif ()
