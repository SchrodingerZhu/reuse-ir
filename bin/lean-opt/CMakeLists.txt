get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
get_property(conversion_libs GLOBAL PROPERTY MLIR_CONVERSION_LIBS)
set(LIBS
        ${dialect_libs}
        ${conversion_libs}
        MLIROptLib
        MLIRRefcnt
        MLIRLean
        MLIRLeanAnalysis
        MLIRAnalysis
        MLIRLeanPass
        )
add_llvm_executable(lean-opt main.cpp)

llvm_update_compile_flags(lean-opt)
target_link_libraries(lean-opt PRIVATE ${LIBS})

mlir_check_all_link_libraries(lean-opt)