#include "mlir/IR/AsmState.h"
#include <Lean/Analysis/TypeTagAnalysis.h>
#include <mlir/Analysis/DataFlow/ConstantPropagationAnalysis.h>
#include <mlir/Analysis/DataFlow/DeadCodeAnalysis.h>
#include <mlir/IR/Matchers.h>
#include <mlir/Pass/Pass.h>

using namespace mlir;
using namespace mlir::dataflow;

/// Print the liveness of every block, control-flow edge, and the predecessors
/// of all regions, callables, and calls.
static void printAnalysisResults(DataFlowSolver &solver, Operation *op,
                                 raw_ostream &os) {
  op->walk([&](Operation *op) {
    auto tag = op->getAttrOfType<StringAttr>("tag");
    if (!tag)
      return;
    os << tag.getValue() << ":\n";
    for (Region &region : op->getRegions()) {
      os << " region #" << region.getRegionNumber() << "\n";
      for (Block &block : region) {
        os << "  ";
        block.printAsOperand(os);
        os << " = ";
        auto *type_tag = solver.lookupState<lean::TypeTagSemiLattice>(&block);
        if (type_tag) {
          AsmState asmState(op);
          os << "  type-tag:\n";
          for (auto &pair : type_tag->getTypedValues()) {
            os << "   ";
            pair.getFirst().printAsOperand(os, asmState);
            os << " = " << pair.getSecond().first << "("
               << pair.getSecond().second << ")\n";
          }
        }
      }
    }
  });
}

/// This is a simple pass that runs dead code analysis with a constant value
/// provider that only understands constant operations.
struct TestTypeTagAnalysisPass
    : public PassWrapper<TestTypeTagAnalysisPass, OperationPass<>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(TestTypeTagAnalysisPass)

  StringRef getArgument() const override { return "test-type-tag-analysis"; }

  void runOnOperation() override {
    Operation *op = getOperation();

    DataFlowSolver solver;
    solver.load<DeadCodeAnalysis>();
    solver.load<lean::TypeTagAnalysis>();
    if (failed(solver.initializeAndRun(op)))
      return signalPassFailure();
    printAnalysisResults(solver, op, llvm::errs());
  }
};

namespace mlir::lean {
void registerTestTypeTagAnalysisPass() {
  PassRegistration<TestTypeTagAnalysisPass>();
}
} // namespace mlir::lean