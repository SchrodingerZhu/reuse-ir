find_package(GTest REQUIRED CONFIG)
include(GoogleTest)

add_executable(GTestLeanBasic GTestLeanBasic.cpp)

llvm_map_components_to_libnames(LLVM_LIBS support)
get_property(MLIR_DIALECT_LIBS GLOBAL PROPERTY MLIR_DIALECT_LIBS)
get_property(MLIR_CONVERSION_LIBS GLOBAL PROPERTY MLIR_CONVERSION_LIBS)

target_link_libraries(GTestLeanBasic PRIVATE MLIRLean MLIRRefcnt MLIRLeanAnalysis ${LLVM_LIBS} ${MLIR_DIALECT_LIBS} ${MLIR_CONVERSION_LIBS} GTest::gtest GTest::gtest_main)
gtest_discover_tests(GTestLeanBasic)
