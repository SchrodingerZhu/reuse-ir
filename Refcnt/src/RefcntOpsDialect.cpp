#include "Refcnt/RefcntOpsDialect.h"

#include "Refcnt/RefcntOps.h"
#include "Refcnt/RefcntOpsDialect.cpp.inc"
#include "Refcnt/RefcntOpsTypes.h"

namespace mlir::refcnt {
void RefcntDialect::initialize() {
  addTypesImpl();
  addOpsImpl();
}
} // namespace mlir::refcnt