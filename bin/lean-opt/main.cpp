#include <mlir/IR/MLIRContext.h>
#include <mlir/InitAllDialects.h>
#include <mlir/InitAllPasses.h>
#include <mlir/Pass/PassRegistry.h>
#include <mlir/Support/FileUtilities.h>
#include <mlir/Tools/mlir-opt/MlirOptMain.h>

#include "Lean/IR/LeanOpsDialect.h"
#include "Lean/Pass/TestReusabilityAnalysis.h"
#include "Lean/Pass/TestTypeTagAnalysis.h"
#include "Refcnt/IR/RefcntOpsDialect.h"

int main(int argc, char **argv) {
  mlir::PassRegistration<mlir::lean::TestTypeTagAnalysisPass>();
  mlir::PassRegistration<mlir::lean::TestReusabilityAnalysisPass>();
  mlir::DialectRegistry registry;
  mlir::registerAllDialects(registry);
  registry.insert<mlir::refcnt::RefcntDialect, mlir::lean::LeanDialect>();
  return mlir::asMainReturnCode(
      mlir::MlirOptMain(argc, argv, "Lean optimizer driver\n", registry));
}