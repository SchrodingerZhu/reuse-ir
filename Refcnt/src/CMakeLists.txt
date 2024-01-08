add_mlir_dialect_library(
    MLIRRefcnt
        RefcntDialect.cpp
    DEPENDS
        MLIRRefcntOpsIncGen
    LINK_COMPONENTS
        Core
)
target_link_libraries(MLIRRefcnt PUBLIC MLIRRefcntTableGen)