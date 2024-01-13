#include "Refcnt/IR/RefcntOpsDialect.h"

#include "Refcnt/IR/RefcntOps.h"
#include "Refcnt/IR/RefcntOpsDialect.cpp.inc"
#include "Refcnt/IR/RefcntOpsTypes.h"

namespace mlir::refcnt {
void RefcntDialect::initialize() {
  addTypesImpl();
  addOpsImpl();
}
} // namespace mlir::refcnt