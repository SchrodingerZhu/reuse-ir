#include "Lean/Pass/TestTypeTagAnalysis.h"
#include "Lean/Analysis/TypeTagAnalysis.h"
namespace mlir::lean {
void TestTypeTagAnalysisPass::runOnOperation() {
  Operation *op = getOperation();

  DataFlowSolver solver;
  // TODO: add this once DeadCodeAnalysis is fixed
  // solver.load<dataflow::DeadCodeAnalysis>();
  solver.load<lean::TypeTagAnalysis>();
  if (failed(solver.initializeAndRun(op)))
    return signalPassFailure();
  printAnalysisResults(solver, op, llvm::errs());
}
void TestTypeTagAnalysisPass::printAnalysisResults(DataFlowSolver &solver,
                                                   Operation *op,
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
          os << "{ ";
          for (auto &pair : type_tag->getTypedValues()) {
            pair.getFirst().printAsOperand(os, asmState);
            os << " = " << pair.getSecond().first << "("
               << pair.getSecond().second << ") ";
          }
        }
        os << "}\n";
      }
    }
  });
}
} // namespace mlir::lean