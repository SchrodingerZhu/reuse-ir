set(LLVM_TARGET_DEFINITIONS LeanOps.td)
mlir_tablegen(LeanOps.h.inc -gen-op-decls)
mlir_tablegen(LeanOps.cpp.inc -gen-op-defs)
mlir_tablegen(LeanOpsTypes.h.inc -gen-typedef-decls)
mlir_tablegen(LeanOpsTypes.cpp.inc -gen-typedef-defs)
mlir_tablegen(LeanOpsDialect.h.inc -gen-dialect-decls -dialect=lean)
mlir_tablegen(LeanOpsDialect.cpp.inc -gen-dialect-defs -dialect=lean)
add_public_tablegen_target(MLIRLeanOpsIncGen)
add_dependencies(mlir-headers MLIRLeanOpsIncGen)
add_library(MLIRLeanTableGen INTERFACE)
add_dependencies(MLIRLeanTableGen mlir-headers)
target_include_directories(MLIRLeanTableGen
  INTERFACE
  ${CMAKE_CURRENT_BINARY_DIR}/..
)
message(STATUS "Lean IR files will be generated to: ${CMAKE_CURRENT_BINARY_DIR}")