add_mlir_dialect_library(MLIRReuse
    ReuseBase.cpp
    ReuseOps.cpp
    DEPENDS

    LINK_COMPONENTS
    Core
)

target_include_directories(MLIRReuse PUBLIC ${CMAKE_SOURCE_DIR}/include ${CMAKE_BINARY_DIR}/include)