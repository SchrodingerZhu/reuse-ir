#include "Lean/IR/LeanOpsDialect.h"

#include "Lean/IR/LeanOpsDialect.cpp.inc"

namespace mlir::lean {
void LeanDialect::initialize() {
  addTypesImpl();
  addOpsImpl();
}
} // namespace mlir::lean