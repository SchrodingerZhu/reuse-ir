#include "Refcnt/Analysis/ReuseAnalysis.h"
namespace mlir::refcnt {
Reusable *ReusabibilityLookupTable::getMostPreferredReusable() const {
  if (this->empty()) {
    return nullptr;
  }
  return this->begin()->get();
}
void Reusable::dump(llvm::raw_ostream &os, AsmState &st) const {
  os << "[";
  value.printAsOperand(os, st);
  os << ", score: " << reusability() << "]";
};
/*
void ReuseTokenCollection::print(llvm::raw_ostream &os) const {
  os << "{";
  llvm::interleaveComma(*this, os, [&](auto token) {
    Value val = token->getReusedValue();
    auto op = val.getDefiningOp();
    auto asmState = AsmState(op);
    val.printAsOperand(os, asmState);
    if (token->getKind() == ReuseToken::Kind::Conditional) {
      os << "[conditional]";
    } else {
      os << "[direct]";
    }
  });
  os << "}";
}
double ConditonalReuseToken::getReuseProbability(DataFlowSolver &solver) const {
  auto blk = getJoinBlock();
  size_t total = 0;
  size_t valid = 0;
  for (auto pred : blk->getPredecessors()) {
    total += 1;
    if (pred->mightHaveTerminator()) {
      auto term = pred->getTerminator();
      auto collection = solver.lookupState<ReuseTokenCollection>(term);
      if (collection && collection->contains(getSource())) {
        valid += 1;
      }
    }
  }
  // TODO: also consider branch probability.
  return static_cast<double>(valid) / static_cast<double>(total);
}
void ReusabibilityLookupTable::print(llvm::raw_ostream &os) const {
  os << "{";
  llvm::interleaveComma(*this, os, [&](auto &reusable) {
    reusable->dump(os, AsmState(reusable->getValue().getDefiningOp()));
  });
  os << "}";
}
*/
} // namespace mlir::refcnt