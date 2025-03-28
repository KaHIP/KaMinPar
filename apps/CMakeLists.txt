set(COMMON_APP_SOURCE_FILES
    ${CMAKE_CURRENT_SOURCE_DIR}/version.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/version.h)

add_executable(KaMinParApp ${COMMON_APP_SOURCE_FILES} KaMinPar.cc)
set_property(TARGET KaMinParApp PROPERTY OUTPUT_NAME KaMinPar)

target_link_libraries(KaMinParApp PRIVATE KaMinPar::KaMinPar KaMinPar::KaMinParCLI KaMinPar::KaMinParIO)
target_compile_options(KaMinParApp PRIVATE ${KAMINPAR_WARNING_FLAGS})
message(STATUS "Enabled app: KaMinPar")

if (TARGET dKaMinPar)
    add_executable(dKaMinParApp ${COMMON_APP_SOURCE_FILES} dKaMinPar.cc)
    set_property(TARGET dKaMinParApp PROPERTY OUTPUT_NAME dKaMinPar)

    target_link_libraries(dKaMinParApp PRIVATE KaMinPar::dKaMinPar KaMinPar::dKaMinParCLI KaMinPar::dKaMinParIO)
    target_compile_options(dKaMinParApp PRIVATE ${KAMINPAR_WARNING_FLAGS})

    if (KAMINPAR_BUILD_WITH_BACKWARD)
        target_link_libraries(dKaMinParApp PRIVATE Backward::Backward)
        target_compile_definitions(dKaMinParApp PRIVATE -DKAMINPAR_HAVE_BACKWARD)
    endif ()

    message(STATUS "Enabled app: dKaMinPar")
else ()
    message(STATUS "Disabled app: dKaMinPar")
endif ()

if (KAMINPAR_BUILD_BENCHMARKS)
    add_subdirectory(benchmarks)
endif ()

if (KAMINPAR_BUILD_TOOLS)
    add_subdirectory(tools)
endif ()
