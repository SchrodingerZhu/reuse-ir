#include <mlir/IR/Diagnostics.h>
#include <mlir/Support/LogicalResult.h>

#include "Refcnt/RefcntOps.h"

#define GET_OP_CLASSES
#include "Refcnt/RefcntOps.cpp.inc"

namespace mlir::refcnt {
LogicalResult DropOp::verify() {
  if (this->getValue().getType().getPointee() ==
      this->getToken().getType().getLayout())
    return success();
  ::mlir::emitError(getLoc(), "invalid refcnf.drop: pointee type of rc "
                              "does not match the type of the token.");
  return failure();
}
} // namespace mlir::refcnt