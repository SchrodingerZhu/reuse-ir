add_mlir_dialect_library(
    MLIRRefcnt
        RefcntOps.cpp
        RefcntOpsDialect.cpp
        RefcntOpsTypes.cpp
    DEPENDS
        MLIRRefcntOpsIncGen
    LINK_COMPONENTS
        Core
)
target_link_libraries(MLIRRefcnt PUBLIC MLIRRefcntTableGen)
