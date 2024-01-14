#include "Lean/Analysis/ReusabilityAnalysis.h"
#include "Lean/Analysis/TypeTagAnalysis.h"
#include "Lean/IR/LeanOps.h"
#include "Lean/IR/LeanOpsTypes.h"
#include <llvm/Support/Casting.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>

namespace mlir::lean {
ReusabilityAnalysis::ReusabilityAnalysis(Operation *op)
    : reusabilityTable(), solver() {
  solver.load<TypeTagAnalysis>();
  (void)solver.initializeAndRun(op);
  analyzeRecursively(op);
};

void ReusabilityAnalysis::analyzeRecursively(Operation *current) {
  if (auto func = llvm::dyn_cast_if_present<mlir::func::FuncOp>(current)) {
    for (Block &block : func.getBody()) {
      for (auto &op : block.getOperations()) {
        if (auto newOp = llvm::dyn_cast_if_present<refcnt::NewOp>(op)) {
          visitNewOp(newOp);
        }
      }
    }
    return;
  }
  for (auto &region : current->getRegions()) {
    for (auto &ops : region.getOps()) {
      analyzeRecursively(&ops);
    }
  }
}

void ReusabilityAnalysis::visitNewOp(refcnt::NewOp newOp) {
  // if new operation is not allocating an object, we cannot
  // perform reusability analysis
  if (!newOp.getType().getPointee().isa<ObjType>())
    return;

  // Check if nessesary type information is present
  auto attr = newOp->getAttrDictionary();
  auto rawLeanType = attr.get("lean.type").dyn_cast_or_null<StringAttr>();
  auto rawLeanTag = attr.get("lean.tag").dyn_cast_or_null<IntegerAttr>();

  if (!rawLeanType || !rawLeanTag)
    return;
  auto leanType = rawLeanType.getValue();
  auto leanTag = rawLeanTag.getUInt();

  auto block = newOp->getBlock();
  const auto *typeTags = solver.lookupState<TypeTagSemiLattice>(block);
  if (!typeTags)
    return;

  auto resultValue = newOp.getResult();
  auto canAnalyzeFieldReuse = true;
  llvm::DenseMap<uint64_t, Value> objFields;
  llvm::DenseMap<uint64_t, Value> scalarFields;
  for (auto &use : resultValue.getUses()) {
    auto owner = use.getOwner();
    if (auto setOp = llvm::dyn_cast_if_present<SetOp>(owner)) {
      auto field = setOp.getField().getUInt();
      // if set operation spans multiple blocks, or if there are multiple
      // set operations on the same field, we cannot perform field-level
      // reusability analysis
      if (owner->getBlock() != block || objFields.contains(field)) {
        canAnalyzeFieldReuse = false;
        break;
      }
      objFields[field] = setOp.getValue();
    }
    if (auto ssetOp = llvm::dyn_cast_if_present<SSetOp>(owner)) {
      auto field = ssetOp.getOffset().getUInt();
      // if sset operation spans multiple blocks, or if there are multiple
      // sset operations at the same offset, we cannot perform field-level
      // reusability analysis
      if (owner->getBlock() != block || scalarFields.contains(field)) {
        canAnalyzeFieldReuse = false;
        break;
      }
      scalarFields[field] = ssetOp.getValue();
    }
  }

  for (const auto &[value, typeTag] : typeTags->getTypedValues()) {
    // We only care about the one with the same type and tag
    if (typeTag.first != leanType || typeTag.second != leanTag)
      continue;
    llvm::SmallVector<size_t, 4> reusableObjs;
    llvm::SmallVector<size_t, 4> reusableScalars;
    if (canAnalyzeFieldReuse) {
      for (const auto &[fieldIdx, fieldVal] : objFields) {
        auto defOp = fieldVal.getDefiningOp();
        if (auto projOp = llvm::dyn_cast_if_present<ProjOp>(defOp)) {
          if (projOp.getOperand() == value &&
              projOp.getField().getUInt() == fieldIdx) {
            reusableObjs.push_back(fieldIdx);
          }
        }
      }
      for (const auto &[fieldOffset, fieldVal] : scalarFields) {
        auto defOp = fieldVal.getDefiningOp();
        if (auto sprojOp = llvm::dyn_cast_if_present<SProjOp>(defOp)) {
          if (sprojOp.getOperand() == value &&
              sprojOp.getOffset().getUInt() == fieldOffset) {
            reusableScalars.push_back(fieldOffset);
          }
        }
      }
    }
    reusabilityTable[resultValue].insert(std::make_unique<ReusableObject>(
        value, std::move(reusableObjs), std::move(reusableScalars)));
  }
}
} // namespace mlir::lean