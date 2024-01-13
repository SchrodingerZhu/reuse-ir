#ifndef LEAN_TEST_TYPE_TAG_ANALYSIS_H
#define LEAN_TEST_TYPE_TAG_ANALYSIS_H
#include <mlir/Analysis/DataFlowFramework.h>
#include <mlir/IR/AsmState.h>
#include <mlir/IR/Matchers.h>
#include <mlir/Pass/Pass.h>
namespace mlir::lean {
struct TestTypeTagAnalysisPass
    : public PassWrapper<TestTypeTagAnalysisPass, OperationPass<>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(TestTypeTagAnalysisPass)
  StringRef getArgument() const override { return "test-type-tag-analysis"; }
  static void printAnalysisResults(DataFlowSolver &solver, Operation *op,
                                   raw_ostream &os);
  void runOnOperation() override;
};
} // namespace mlir::lean
#endif // LEAN_TEST_TYPE_TAG_ANALYSIS_H
