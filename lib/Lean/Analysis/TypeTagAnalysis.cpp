#include <iterator>
#include <llvm/ADT/DenseMap.h>
#include <llvm/Support/Casting.h>
#include <mlir/Analysis/DataFlow/DeadCodeAnalysis.h>
#include <mlir/Dialect/ControlFlow/IR/ControlFlowOps.h>
#include <mlir/IR/BuiltinAttributes.h>
#include <mlir/Support/LogicalResult.h>

#include "Lean/Analysis/TypeTagAnalysis.h"
#include "Lean/IR/LeanOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"

namespace mlir::dataflow::lean {
void TypeTagSemiLattice::meet(
    ::llvm::DenseMap<::mlir::Value, TypeTag> &lhs,
    const ::llvm::DenseMap<::mlir::Value, TypeTag> &rhs) {
  ::llvm::SmallVector<Value> removeValues;
  for (const auto &[leftVal, leftTy] : lhs) {
    auto iter = rhs.find(leftVal);
    if (iter != rhs.end() && iter->second == leftTy)
      continue;
    removeValues.push_back(leftVal);
  }
  for (auto val : removeValues)
    lhs.erase(val);
}
void TypeTagSemiLattice::print(llvm::raw_ostream &os) const {
  os << "{";
  if (!this->typedValues.empty()) {
    auto iter = this->typedValues.begin();
    os << iter->first << " : " << iter->second.first << "("
       << iter->second.second << ")";
    ++iter;
    for (; iter != this->typedValues.end(); ++iter) {
      os << ", " << iter->first << " : " << iter->second.first << "("
         << iter->second.second << ")";
    }
  }
  os << "}";
}
LogicalResult TypeTagAnalysis::visit(ProgramPoint point) {
  if (auto *block = llvm::dyn_cast_if_present<Block *>(point))
    visitBlock(block);
  return success();
}

void TypeTagAnalysis::initializeRecursively(Operation *current) {
  if (auto func = llvm::dyn_cast_if_present<func::FuncOp>(current)) {
    for (Block &block : func.getBody()) {
      auto lattice = getOrCreate<TypeTagSemiLattice>(&block);
      for (auto successor : block.getSuccessors()) {
        // if (!getOrCreateFor<Executable>(
        //          &block, getProgramPoint<CFGEdge>(&block, successor))
        //          ->isLive())
        //   continue;
        lattice->addDependency(successor, this);
      }
      visitBlock(&block);
    }
    return;
  }
  for (auto &region : current->getRegions()) {
    for (auto &ops : region.getOps()) {
      initializeRecursively(&ops);
    }
  }
}

LogicalResult TypeTagAnalysis::initialize(Operation *top) {
  initializeRecursively(top);
  return success();
}

std::optional<std::pair<Value, TypeTag>>
TypeTagAnalysis::getGenTypeTag(Block *block) {
  // if (!getOrCreateFor<Executable>(block, block)->isLive())
  //   return std::nullopt;

  if (auto pred = block->getSinglePredecessor()) {
    if (pred->mightHaveTerminator()) {
      auto *terminator = pred->getTerminator();
      if (auto switchOp = llvm::dyn_cast_if_present<cf::SwitchOp>(terminator)) {
        if (switchOp.getDefaultDestination() == block)
          return std::nullopt;
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
            return std::make_pair(operand, typeTag);
          }
        }
      }
    }
  }

  return std::nullopt;
}

void TypeTagAnalysis::visitBlock(Block *block) {
  // Ignore dead blocks.
  // if (!getOrCreateFor<Executable>(block, block)->isLive())
  //   return;

  auto lattice = getOrCreate<TypeTagSemiLattice>(block);

  ::llvm::DenseMap<Value, TypeTag> map;

  auto begin = block->pred_begin();
  auto end = block->pred_end();

  while (begin != end) {
    auto pred = *begin;
    // if (!getOrCreateFor<Executable>(block,
    //                                 getProgramPoint<CFGEdge>(pred, block))
    //          ->isLive())
    //   continue;
    auto predLattice = getOrCreate<TypeTagSemiLattice>(pred);
    map = predLattice->getTypedValues();
    ++begin;
    break;
  }

  while (begin != end) {
    auto pred = *begin;
    // if (!getOrCreateFor<Executable>(block,
    //                                 getProgramPoint<CFGEdge>(pred, block))
    //          ->isLive())
    //   continue;
    auto predLattice = getOrCreate<TypeTagSemiLattice>(pred);
    TypeTagSemiLattice::meet(map, predLattice->getTypedValues());
    ++begin;
  }

  if (auto genTypeTag = getGenTypeTag(block)) {
    map.insert(*genTypeTag);
  }

  propagateIfChanged(lattice, lattice->setTypedValue(std::move(map)));
}
RunTypeTagAnalysis::RunTypeTagAnalysis(Operation *op) {
  solver.load<DeadCodeAnalysis>();
  solver.load<TypeTagAnalysis>();
  (void)solver.initializeAndRun(op);
};
const TypeTagSemiLattice *RunTypeTagAnalysis::getKnownTypeTags(Block *block) {
  return solver.lookupState<TypeTagSemiLattice>(block);
}
} // namespace mlir::dataflow::lean