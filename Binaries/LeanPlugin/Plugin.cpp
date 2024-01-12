#include <llvm/Config/llvm-config.h>
#include <llvm/Support/Compiler.h>
#include <mlir/IR/MLIRContext.h>
#include <mlir/InitAllDialects.h>
#include <mlir/Tools/Plugins/DialectPlugin.h>

#include <Lean/IR/LeanOpsDialect.h>
#include <Refcnt/RefcntOpsDialect.h>

namespace mlir::lean {
void registerTestTypeTagAnalysisPass();
}

extern "C" LLVM_ATTRIBUTE_WEAK mlir::DialectPluginLibraryInfo
mlirGetDialectPluginInfo() {
  return {MLIR_PLUGIN_API_VERSION, "LeanDialect", LLVM_VERSION_STRING,
          [](mlir::DialectRegistry *registry) {
            registry->insert<mlir::refcnt::RefcntDialect>();
            registry->insert<mlir::lean::LeanDialect>();
            mlir::lean::registerTestTypeTagAnalysisPass();
          }};
}

extern "C" LLVM_ATTRIBUTE_WEAK mlir::PassPluginLibraryInfo
mlirGetPassPluginInfo() {
  return {MLIR_PLUGIN_API_VERSION, "LeanPasses", LLVM_VERSION_STRING,
          []() { mlir::lean::registerTestTypeTagAnalysisPass(); }};
}