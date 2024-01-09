#include <mlir/IR/Attributes.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/BuiltinAttributes.h>
#include <mlir/IR/BuiltinTypes.h>
#include <mlir/IR/Diagnostics.h>
#include <mlir/IR/DialectImplementation.h>
#include <mlir/Interfaces/DataLayoutInterfaces.h>
#include <mlir/Support/LLVM.h>
#include <mlir/Support/LogicalResult.h>

#include <llvm/ADT/STLExtras.h>
#include <llvm/ADT/SmallVector.h>
#include <llvm/ADT/TypeSwitch.h>
#include <llvm/Support/ErrorHandling.h>

#include "Lean/LeanOpsDialect.h"
#include "Lean/LeanOpsTypes.h"

#define GET_TYPEDEF_CLASSES
#include "Lean/LeanOpsTypes.cpp.inc"

namespace mlir::lean {
void LeanDialect::addTypesImpl() {
  addTypes<
#define GET_TYPEDEF_LIST
#include "Lean/LeanOpsTypes.cpp.inc"
      >();
}

unsigned ObjType::getTypeSizeInBits(
    const ::mlir::DataLayout &dataLayout,
    [[maybe_unused]] ::mlir::DataLayoutEntryListRef params) const {
  auto indexBits =
      dataLayout.getTypeSizeInBits(::mlir::IndexType::get(getContext()));
  return indexBits * (1 + getSubObjs()) + getScalaSize() * 8;
}
unsigned ObjType::getABIAlignment(
    const ::mlir::DataLayout &dataLayout,
    [[maybe_unused]] ::mlir::DataLayoutEntryListRef params) const {
  return dataLayout.getTypeABIAlignment(::mlir::IndexType::get(getContext()));
}
unsigned ObjType::getPreferredAlignment(
    const ::mlir::DataLayout &dataLayout,
    [[maybe_unused]] ::mlir::DataLayoutEntryListRef params) const {
  return dataLayout.getTypePreferredAlignment(
      ::mlir::IndexType::get(getContext()));
}
} // namespace mlir::lean