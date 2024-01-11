#include <iterator>
#include <llvm/Support/Casting.h>
#include <mlir/Analysis/DataFlow/DeadCodeAnalysis.h>
#include <mlir/Dialect/ControlFlow/IR/ControlFlowOps.h>
#include <mlir/Support/LogicalResult.h>

#include "Lean/Analysis/TypeTagAnalysis.h"
#include "Lean/IR/LeanOps.h"
#include "mlir/IR/BuiltinAttributes.h"

namespace mlir::dataflow::lean {
ChangeResult TypeTagSemiLattice::meet(const TypeTagSemiLattice &rhs) {
  ::llvm::SmallVector<Value> removeValues;
  for (const auto &[leftVal, leftTy] : this->typedValues) {
    auto iter = rhs.typedValues.find(leftVal);
    if (iter != rhs.typedValues.end() && iter->second == leftTy)
      continue;
    removeValues.push_back(leftVal);
  }
  if (removeValues.empty())
    return ChangeResult::NoChange;
  for (auto val : removeValues)
    this->typedValues.erase(val);
  return ChangeResult::Change;
}
void TypeTagSemiLattice::print(llvm::raw_ostream &os) const {
  os << "{";
  if (!this->typedValues.empty()) {
    auto iter = this->typedValues.begin();
    os << iter->first << " : " << iter->second.first;
    ++iter;
    for (; iter != this->typedValues.end(); ++iter) {
      os << ", " << iter->first << " : " << iter->second.first;
    }
  }
  os << "}";
}
LogicalResult TypeTagAnalysis::visit(ProgramPoint point) {
  if (auto *block = llvm::dyn_cast_if_present<Block *>(point))
    visitBlock(block);
  return success();
}
LogicalResult TypeTagAnalysis::initialize(Operation *top) {
  // TODO: implement this
  return success();
}
void TypeTagAnalysis::visitBlock(Block *block) {
  // Ignore dead blocks.
  if (!getOrCreateFor<Executable>(block, block)->isLive())
    return;

  auto lattice = getOrCreate<TypeTagSemiLattice>(block);
  // TODO: handle meet operation
  // Check if this is an pattern match target block.
  if (auto pred = block->getSinglePredecessor()) {
    if (pred->mightHaveTerminator()) {
      auto *terminator = pred->getTerminator();
      if (auto switchOp = llvm::dyn_cast_if_present<cf::SwitchOp>(terminator)) {
        auto flag = switchOp.getFlag();
        if (auto typeString =
                switchOp->getAttrDictionary().getAs<StringAttr>("lean.type")) {
          auto defOp = flag.getDefiningOp();
          if (auto tagOp =
                  llvm::dyn_cast_if_present<::mlir::lean::TagOp>(defOp)) {
            auto operand = tagOp.getOperand();
            // find the corresponding tag value of current block
            auto caseValues = switchOp.getCaseValues();
            auto caseDestinations = switchOp.getCaseDestinations();
            auto blockIdx = std::find(caseDestinations.begin(),
                                      caseDestinations.end(), block) -
                            caseDestinations.begin();
            auto tagValue = *std::next(caseValues->begin(), blockIdx);
            TypeTag typeTag = {typeString.getValue().str(),
                               static_cast<size_t>(tagValue.getZExtValue())};
            lattice->getTypedValues().insert({operand, typeTag});
          }
        }
      }
    }
  }
}
} // namespace mlir::dataflow::lean