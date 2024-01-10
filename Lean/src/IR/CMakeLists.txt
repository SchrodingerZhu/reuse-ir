add_mlir_dialect_library(
    MLIRLean
        LeanOps.cpp
        LeanOpsDialect.cpp
        LeanOpsTypes.cpp
        LeanOpsAttributes.cpp
    DEPENDS
        MLIRLeanOpsIncGen
    LINK_COMPONENTS
        Core
)
target_link_libraries(MLIRLean PUBLIC MLIRRefcntTableGen MLIRLeanTableGen)
