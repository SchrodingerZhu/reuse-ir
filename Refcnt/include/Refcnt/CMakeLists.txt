set(LLVM_TARGET_DEFINITIONS RefcntOps.td)
mlir_tablegen(RefcntOps.h.inc -gen-op-decls)
mlir_tablegen(RefcntOps.cpp.inc -gen-op-defs)
mlir_tablegen(RefcntOpsTypes.h.inc -gen-typedef-decls)
mlir_tablegen(RefcntpsTypes.cpp.inc -gen-typedef-defs)
mlir_tablegen(RefcnfOpsDialect.h.inc -gen-dialect-decls)
mlir_tablegen(RefcntOpsDialect.cpp.inc -gen-dialect-defs)
add_public_tablegen_target(MLIRRefcntOpsIncGen)
add_dependencies(mlir-headers MLIRRefcntOpsIncGen)
add_library(MLIRRefcntTableGen INTERFACE)
add_dependencies(MLIRRefcntTableGen mlir-headers)
target_include_directories(MLIRRefcntTableGen
  INTERFACE
  ${CMAKE_CURRENT_BINARY_DIR}/..
)
message(STATUS "Refcnt IR files will be generated : ${CMAKE_CURRENT_BINARY_DIR}")