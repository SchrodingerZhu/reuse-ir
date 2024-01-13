add_mlir_dialect_library(
    MLIRLeanPass
        TestTypeTagAnalysis.cpp
    DEPENDS
        MLIRLeanAnalysis 
    LINK_COMPONENTS
        Core
)