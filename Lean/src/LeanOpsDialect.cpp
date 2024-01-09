#include "Lean/LeanOpsDialect.h"

#include "Lean/LeanOpsDialect.cpp.inc"

namespace mlir::lean {
void LeanDialect::initialize() {
  addTypesImpl();
  addOpsImpl();
}
} // namespace mlir::lean