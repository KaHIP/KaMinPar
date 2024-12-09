find_library(LIB_NETWORKIT networkit)

if (NOT LIB_NETWORKIT)
    message(WARNING "Networkit library not found -- skipping NetworKit binding")
else ()
    add_library(kaminpar_networkit STATIC kaminpar_networkit.cc kaminpar_networkit.h)
    target_link_libraries(kaminpar_networkit PUBLIC KaMinPar::KaMinPar networkit)
    target_include_directories(kaminpar_networkit PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
endif ()
