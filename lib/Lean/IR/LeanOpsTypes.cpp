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

#include "Lean/IR/LeanOpsDialect.h"
#include "Lean/IR/LeanOpsTypes.h"

#define GET_TYPEDEF_CLASSES
#include "Lean/IR/LeanOpsTypes.cpp.inc"

namespace mlir::lean {
void LeanDialect::addTypesImpl() {
  addTypes<
#define GET_TYPEDEF_LIST
#include "Lean/IR/LeanOpsTypes.cpp.inc"
      >();
}
} // namespace mlir::lean