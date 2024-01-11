add_mlir_dialect_library(
    MLIRLeanAnalysis
        TypeTagAnalysis.cpp
    DEPENDS
        MLIRLean
    LINK_COMPONENTS
        Core
)