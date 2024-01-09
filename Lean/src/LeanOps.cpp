#include <mlir/IR/Diagnostics.h>
#include <mlir/Support/LogicalResult.h>

#include "Lean/LeanOps.h"
#include "Lean/LeanOpsTypes.h"

#define GET_OP_CLASSES
#include "Lean/LeanOps.cpp.inc"

namespace mlir::lean {
LogicalResult SetTagOp::verify() {
  if (!this->getObject().getType().getPointee().isa<ObjType>()) {
    ::mlir::emitError(getLoc(), "invalid lean.set_tag: pointee type of rc "
                                "is not a lean object.");
    return failure();
  }
  if (!this->getTag().getType().isIndex()) {
    ::mlir::emitError(getLoc(), "invalid lean.set_tag: tag is not an index.");
    return failure();
  }
  return success();
}
LogicalResult GetTagOp::verify() {
  if (!this->getObject().getType().getPointee().isa<ObjType>()) {
    ::mlir::emitError(getLoc(), "invalid lean.get_tag: pointee type of rc "
                                "is not a lean object.");
    return failure();
  }
  return success();
}
LogicalResult ProjOp::verify() {
  if (!this->getObject().getType().getPointee().isa<ObjType>()) {
    ::mlir::emitError(getLoc(), "invalid lean.proj: pointee type of object rc "
                                "is not a lean object.");
    return failure();
  }
  if (!this->getField().getType().isIndex()) {
    ::mlir::emitError(getLoc(), "invalid lean.proj: field is not an index.");
    return failure();
  }
  if (this->getField().getUInt() >=
      this->getObject().getType().getPointee().cast<ObjType>().getSubObjs()) {
    ::mlir::emitError(getLoc(), "invalid lean.proj: field is out of bounds.");
    return failure();
  }
  if (!this->getResult().getType().getPointee().isa<ObjType>()) {
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

  if (!this->getObject().getType().getPointee().isa<ObjType>()) {
    inflightErr << "pointee type of object rc is not a lean object.";
    return failure();
  }
  if (!this->getOffset().getType().isIndex()) {
    inflightErr << "offset is not an index.";
    return failure();
  }

  const auto &resultType = this->getResult().getType();

  if (!isKnownScalarType(resultType)) {
    inflightErr << "result type is not a known scalar type.";
    return failure();
  }

  const auto &inputPointee =
      this->getObject().getType().getPointee().cast<ObjType>();

  if (this->getOffset().getUInt() + resultType.getIntOrFloatBitWidth() / 8 >=
      inputPointee.getScalaSize()) {
    inflightErr << "access of type " << resultType << " at offset "
                << this->getOffset().getUInt() << " is out of bounds.";
    return failure();
  }

  inflightErr.abandon();
  return success();
}
} // namespace mlir::lean
