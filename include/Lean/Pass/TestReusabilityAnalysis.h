#ifndef LEAN_PASS_TEST_REUSABILITY_ANALYSIS_H
#define LEAN_PASS_TEST_REUSABILITY_ANALYSIS_H
#include <mlir/IR/AsmState.h>
#include <mlir/IR/Matchers.h>
#include <mlir/Pass/Pass.h>

namespace mlir::lean {
struct TestReusabilityAnalysisPass
    : public PassWrapper<TestReusabilityAnalysisPass, OperationPass<>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(TestReusabilityAnalysisPass)
  StringRef getArgument() const override { return "test-reusability-analysis"; }
  static void printAnalysisResults(raw_ostream &os, Operation *op,
                                   const class ReusabilityAnalysis &analysis);
  void runOnOperation() override;
};
} // namespace mlir::lean
#endif // LEAN_PASS_TEST_REUSABILITY_ANALYSIS_H