#include <mlir/IR/Diagnostics.h>
#include <mlir/Support/LogicalResult.h>

#include "Lean/LeanOps.h"
#include "Lean/LeanOpsDialect.h"
#include "Lean/LeanOpsTypes.h"
#include "Refcnt/RefcntOpsTypes.h"

#define GET_OP_CLASSES
#include "Lean/LeanOps.cpp.inc"

namespace mlir::lean {

static bool isRcOfObject(const Value &rc) {
  if (!rc.getType().isa<refcnt::RcType>()) {
    return false;
  }
  auto casted = rc.getType().cast<refcnt::RcType>();
  return casted.getPointee().isa<ObjType>();
}

LogicalResult TagOp::verify() {
  if (!isRcOfObject(this->getObject())) {
    ::mlir::emitError(getLoc(), "invalid lean.get_tag: pointee type of rc "
                                "is not a lean object.");
    return failure();
  }
  return success();
}
LogicalResult ProjOp::verify() {
  if (!isRcOfObject(this->getObject())) {
    ::mlir::emitError(getLoc(), "invalid lean.proj: pointee type of object rc "
                                "is not a lean object.");
    return failure();
  }
  if (!this->getField().getType().isIndex()) {
    ::mlir::emitError(getLoc(), "invalid lean.proj: field is not an index.");
    return failure();
  }
  if (!isRcOfObject(this->getResult())) {
    ::mlir::emitError(getLoc(), "invalid lean.proj: pointee type of result rc "
                                "is not a lean object.");
  }
  return success();
}

static bool isKnownScalarType(const mlir::Type &type) {
  return type.isIndex() || type.isF64() || type.isUnsignedInteger();
}

LogicalResult SProjOp::verify() {
  auto inflightErr = ::mlir::emitError(getLoc(), "invalid lean.sproj: ");

  if (!isRcOfObject(this->getObject())) {
    inflightErr << "pointee type of object rc is not a lean object.";
    return failure();
  }
  if (!this->getOffset().getType().isIndex()) {
    inflightErr << "offset is not an index.";
    return failure();
  }
  if (!isKnownScalarType(this->getResult().getType())) {
    inflightErr << "result type is not a known scalar type.";
    return failure();
  }

  inflightErr.abandon();
  return success();
}

LogicalResult AppOp::verify() {
  auto inflightErr = ::mlir::emitError(getLoc(), "invalid lean.app: ");
  if (!isRcOfObject(this->getFn())) {
    inflightErr << "pointee type of function rc is not a lean object.";
    return failure();
  }
  if (this->getArgs().size() < 1) {
    inflightErr << "too few arguments.";
    return failure();
  }
  for (const auto &arg : this->getArgs()) {
    if (!isRcOfObject(arg)) {
      inflightErr << "pointee type of argument" << arg
                  << "is not a lean object.";
      return failure();
    }
  }
  if (!isRcOfObject(this->getResult())) {
    inflightErr << "pointee type of result rc is not a lean object.";
    return failure();
  }
  inflightErr.abandon();
  return success();
}

void LeanDialect::addOpsImpl() {
  addOperations<
#define GET_OP_LIST
#include "Lean/LeanOps.cpp.inc"
      >();
}
} // namespace mlir::lean
