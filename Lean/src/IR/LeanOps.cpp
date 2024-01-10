#include <mlir/IR/BuiltinTypes.h>
#include <mlir/IR/Diagnostics.h>
#include <mlir/Support/LogicalResult.h>

#include "Lean/IR/LeanOps.h"
#include "Lean/IR/LeanOpsDialect.h"
#include "Lean/IR/LeanOpsTypes.h"
#include "Refcnt/RefcntOpsTypes.h"

#define GET_OP_CLASSES
#include "Lean/IR/LeanOps.cpp.inc"

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

static refcnt::RcType getRcObjType(MLIRContext *context) {
  return refcnt::RcType::get(context, lean::ObjType::get(context));
}

::mlir::ParseResult TagOp::parse(::mlir::OpAsmParser &parser,
                                 ::mlir::OperationState &result) {
  ::mlir::OpAsmParser::UnresolvedOperand objectRawOperands[1];
  ::llvm::ArrayRef<::mlir::OpAsmParser::UnresolvedOperand> objectOperands(
      objectRawOperands);
  ::llvm::SMLoc objectOperandsLoc;
  (void)objectOperandsLoc;
  ::mlir::Type objectRawTypes[1];
  ::llvm::ArrayRef<::mlir::Type> objectTypes(objectRawTypes);

  objectOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(objectRawOperands[0]))
    return ::mlir::failure();

  objectRawTypes[0] = getRcObjType(parser.getContext());

  if (parser.parseOptionalAttrDict(result.attributes))
    return ::mlir::failure();

  ::mlir::Type odsBuildableType0 = IntegerType::get(parser.getContext(), 16);
  result.addTypes(odsBuildableType0);
  if (parser.resolveOperands(objectOperands, objectTypes, objectOperandsLoc,
                             result.operands))
    return ::mlir::failure();
  return ::mlir::success();
}

void TagOp::print(::mlir::OpAsmPrinter &_odsPrinter) {
  _odsPrinter << ' ';
  _odsPrinter << getObject();
  ::llvm::SmallVector<::llvm::StringRef, 2> elidedAttrs;
  _odsPrinter.printOptionalAttrDict((*this)->getAttrs(), elidedAttrs);
}

::mlir::ParseResult ProjOp::parse(::mlir::OpAsmParser &parser,
                                  ::mlir::OperationState &result) {
  ::mlir::OpAsmParser::UnresolvedOperand objectRawOperands[1];
  ::llvm::ArrayRef<::mlir::OpAsmParser::UnresolvedOperand> objectOperands(
      objectRawOperands);
  ::llvm::SMLoc objectOperandsLoc;
  (void)objectOperandsLoc;
  ::mlir::Type objectRawTypes[1];
  ::llvm::ArrayRef<::mlir::Type> objectTypes(objectRawTypes);
  ::mlir::Type resultRawTypes[1];
  ::llvm::ArrayRef<::mlir::Type> resultTypes(resultRawTypes);
  ::mlir::IntegerAttr fieldAttr;

  objectOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(objectRawOperands[0]))
    return ::mlir::failure();

  objectRawTypes[0] = getRcObjType(parser.getContext());
  resultRawTypes[0] = getRcObjType(parser.getContext());

  if (parser.parseComma())
    return ::mlir::failure();

  if (parser.parseCustomAttributeWithFallback(fieldAttr, ::mlir::Type{},
                                              "field", result.attributes)) {
    return ::mlir::failure();
  }
  if (parser.parseOptionalAttrDict(result.attributes))
    return ::mlir::failure();
  result.addTypes(resultTypes);
  if (parser.resolveOperands(objectOperands, objectTypes, objectOperandsLoc,
                             result.operands))
    return ::mlir::failure();
  return ::mlir::success();
}

void ProjOp::print(::mlir::OpAsmPrinter &_odsPrinter) {
  _odsPrinter << ' ';
  _odsPrinter << getObject();
  _odsPrinter << ", ";
  _odsPrinter.printStrippedAttrOrType(getFieldAttr());
  ::llvm::SmallVector<::llvm::StringRef, 2> elidedAttrs;
  elidedAttrs.push_back("field");
  _odsPrinter.printOptionalAttrDict((*this)->getAttrs(), elidedAttrs);
}

::mlir::ParseResult SProjOp::parse(::mlir::OpAsmParser &parser,
                                   ::mlir::OperationState &result) {
  ::mlir::OpAsmParser::UnresolvedOperand objectRawOperands[1];
  ::llvm::ArrayRef<::mlir::OpAsmParser::UnresolvedOperand> objectOperands(
      objectRawOperands);
  ::llvm::SMLoc objectOperandsLoc;
  (void)objectOperandsLoc;
  ::mlir::Type objectRawTypes[1];
  ::llvm::ArrayRef<::mlir::Type> objectTypes(objectRawTypes);
  ::mlir::Type resultRawTypes[1];
  ::llvm::ArrayRef<::mlir::Type> resultTypes(resultRawTypes);
  ::mlir::IntegerAttr offsetAttr;

  objectOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(objectRawOperands[0]))
    return ::mlir::failure();
  objectRawTypes[0] = getRcObjType(parser.getContext());

  if (parser.parseComma())
    return ::mlir::failure();

  if (parser.parseCustomAttributeWithFallback(offsetAttr, ::mlir::Type{},
                                              "offset", result.attributes)) {
    return ::mlir::failure();
  }

  if (parser.parseComma())
    return ::mlir::failure();

  {
    ::mlir::Type type;
    if (parser.parseCustomTypeWithFallback(type))
      return ::mlir::failure();
    resultRawTypes[0] = type;
  }
  result.addTypes(resultTypes);

  if (parser.parseOptionalAttrDict(result.attributes))
    return ::mlir::failure();

  if (parser.resolveOperands(objectOperands, objectTypes, objectOperandsLoc,
                             result.operands))
    return ::mlir::failure();
  return ::mlir::success();
}

void SProjOp::print(::mlir::OpAsmPrinter &_odsPrinter) {
  _odsPrinter << ' ';
  _odsPrinter << getObject();
  _odsPrinter << ',';
  _odsPrinter << ' ';
  _odsPrinter.printStrippedAttrOrType(getOffsetAttr());
  _odsPrinter << ',';
  _odsPrinter << ' ';
  {
    auto type = getResult().getType();
    if (auto validType = type.dyn_cast<::mlir::IndexType>())
      _odsPrinter.printStrippedAttrOrType(validType);
    else
      _odsPrinter << type;
  }
  ::llvm::SmallVector<::llvm::StringRef, 2> elidedAttrs;
  elidedAttrs.push_back("offset");
  _odsPrinter.printOptionalAttrDict((*this)->getAttrs(), elidedAttrs);
}

::mlir::ParseResult AppOp::parse(::mlir::OpAsmParser &parser,
                                 ::mlir::OperationState &result) {
  ::mlir::OpAsmParser::UnresolvedOperand fnRawOperands[1];
  ::llvm::ArrayRef<::mlir::OpAsmParser::UnresolvedOperand> fnOperands(
      fnRawOperands);
  ::llvm::SMLoc fnOperandsLoc;
  (void)fnOperandsLoc;
  ::mlir::Type fnRawTypes[1];
  ::llvm::ArrayRef<::mlir::Type> fnTypes(fnRawTypes);
  ::llvm::SmallVector<::mlir::OpAsmParser::UnresolvedOperand, 4> argsOperands;
  ::llvm::SMLoc argsOperandsLoc;
  (void)argsOperandsLoc;
  ::llvm::SmallVector<::mlir::Type, 1> argsTypes;
  ::mlir::Type resultRawTypes[1];
  ::llvm::ArrayRef<::mlir::Type> resultTypes(resultRawTypes);

  fnOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(fnRawOperands[0]))
    return ::mlir::failure();

  fnRawTypes[0] = getRcObjType(parser.getContext());

  if (parser.parseLParen())
    return ::mlir::failure();

  argsOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperandList(argsOperands))
    return ::mlir::failure();
  if (parser.parseRParen())
    return ::mlir::failure();

  resultRawTypes[0] = getRcObjType(parser.getContext());
  argsTypes.resize(argsOperands.size(), getRcObjType(parser.getContext()));

  if (parser.parseOptionalAttrDict(result.attributes))
    return ::mlir::failure();
  result.addTypes(resultTypes);
  if (parser.resolveOperands(fnOperands, fnTypes, fnOperandsLoc,
                             result.operands))
    return ::mlir::failure();
  if (parser.resolveOperands(argsOperands, argsTypes, argsOperandsLoc,
                             result.operands))
    return ::mlir::failure();
  return ::mlir::success();
}

void AppOp::print(::mlir::OpAsmPrinter &_odsPrinter) {
  _odsPrinter << ' ';
  _odsPrinter << getFn();
  _odsPrinter << ' ' << '(';
  _odsPrinter << getArgs();
  _odsPrinter << ')';
  ::llvm::SmallVector<::llvm::StringRef, 2> elidedAttrs;
  _odsPrinter.printOptionalAttrDict((*this)->getAttrs(), elidedAttrs);
}

LogicalResult SSetOp::verify() {
  auto inflightErr = ::mlir::emitError(getLoc(), "invalid lean.sset: ");

  if (!isRcOfObject(this->getObject())) {
    inflightErr << "pointee type of object rc is not a lean object.";
    return failure();
  }
  if (!this->getOffset().getType().isIndex()) {
    inflightErr << "offset is not an index.";
    return failure();
  }
  if (!isKnownScalarType(this->getValue().getType())) {
    inflightErr << "value type is not a known scalar type.";
    return failure();
  }

  inflightErr.abandon();
  return success();
}

LogicalResult SetOp::verify() {
  auto inflightErr = ::mlir::emitError(getLoc(), "invalid lean.set: ");
  if (!isRcOfObject(this->getObject())) {
    inflightErr << "pointee type of object rc is not a lean object.";
    return failure();
  }
  if (!isRcOfObject(this->getValue())) {
    inflightErr << "pointee type of value rc is not a lean object.";
    return failure();
  }
  if (!this->getField().getType().isIndex()) {
    inflightErr << "field is not an index.";
    return failure();
  }
  inflightErr.abandon();
  return success();
}

::mlir::ParseResult SSetOp::parse(::mlir::OpAsmParser &parser,
                                  ::mlir::OperationState &result) {
  ::mlir::OpAsmParser::UnresolvedOperand objectRawOperands[1];
  ::llvm::ArrayRef<::mlir::OpAsmParser::UnresolvedOperand> objectOperands(
      objectRawOperands);
  ::llvm::SMLoc objectOperandsLoc;
  ::mlir::Type objectRawTypes[1];
  ::llvm::ArrayRef<::mlir::Type> objectTypes(objectRawTypes);
  ::mlir::IntegerAttr offsetAttr;
  ::mlir::OpAsmParser::UnresolvedOperand valueRawOperands[1];
  ::llvm::ArrayRef<::mlir::OpAsmParser::UnresolvedOperand> valueOperands(
      valueRawOperands);
  ::llvm::SMLoc valueOperandsLoc;
  ::mlir::Type valueRawTypes[1];
  ::llvm::ArrayRef<::mlir::Type> valueTypes(valueRawTypes);

  objectOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(objectRawOperands[0]))
    return ::mlir::failure();
  objectRawTypes[0] = getRcObjType(parser.getContext());

  if (parser.parseComma())
    return ::mlir::failure();

  if (parser.parseCustomAttributeWithFallback(offsetAttr, ::mlir::Type{},
                                              "offset", result.attributes)) {
    return ::mlir::failure();
  }

  if (parser.parseComma())
    return ::mlir::failure();

  valueOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(valueRawOperands[0]))
    return ::mlir::failure();

  if (parser.parseColon())
    return ::mlir::failure();

  {
    ::mlir::Type type;
    if (parser.parseCustomTypeWithFallback(type))
      return ::mlir::failure();
    valueRawTypes[0] = type;
  }

  if (parser.parseOptionalAttrDict(result.attributes))
    return ::mlir::failure();

  if (parser.resolveOperands(objectOperands, objectTypes, objectOperandsLoc,
                             result.operands))
    return ::mlir::failure();
  if (parser.resolveOperands(valueOperands, valueTypes, valueOperandsLoc,
                             result.operands))
    return ::mlir::failure();
  return ::mlir::success();
}

void SSetOp::print(::mlir::OpAsmPrinter &_odsPrinter) {
  _odsPrinter << ' ';
  _odsPrinter << getObject();
  _odsPrinter << ',';
  _odsPrinter << ' ';
  _odsPrinter.printStrippedAttrOrType(getOffsetAttr());
  _odsPrinter << ',';
  _odsPrinter << ' ';
  _odsPrinter << getValue();
  ::llvm::SmallVector<::llvm::StringRef, 2> elidedAttrs;
  elidedAttrs.push_back("offset");
  _odsPrinter.printOptionalAttrDict((*this)->getAttrs(), elidedAttrs);
}

::mlir::ParseResult SetOp::parse(::mlir::OpAsmParser &parser,
                                 ::mlir::OperationState &result) {
  ::mlir::OpAsmParser::UnresolvedOperand objectRawOperands[1];
  ::llvm::ArrayRef<::mlir::OpAsmParser::UnresolvedOperand> objectOperands(
      objectRawOperands);
  ::llvm::SMLoc objectOperandsLoc;
  ::mlir::Type objectRawTypes[1];
  ::llvm::ArrayRef<::mlir::Type> objectTypes(objectRawTypes);
  ::mlir::IntegerAttr fieldAttr;
  ::mlir::OpAsmParser::UnresolvedOperand valueRawOperands[1];
  ::llvm::ArrayRef<::mlir::OpAsmParser::UnresolvedOperand> valueOperands(
      valueRawOperands);
  ::llvm::SMLoc valueOperandsLoc;
  ::mlir::Type valueRawTypes[1];
  ::llvm::ArrayRef<::mlir::Type> valueTypes(valueRawTypes);

  objectOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(objectRawOperands[0]))
    return ::mlir::failure();
  objectRawTypes[0] = getRcObjType(parser.getContext());

  if (parser.parseComma())
    return ::mlir::failure();

  if (parser.parseCustomAttributeWithFallback(fieldAttr, ::mlir::Type{},
                                              "field", result.attributes))
    return ::mlir::failure();

  if (parser.parseComma())
    return ::mlir::failure();

  valueOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(valueRawOperands[0]))
    return ::mlir::failure();
  valueRawTypes[0] = getRcObjType(parser.getContext());

  if (parser.parseOptionalAttrDict(result.attributes))
    return ::mlir::failure();

  if (parser.resolveOperands(objectOperands, objectTypes, objectOperandsLoc,
                             result.operands))
    return ::mlir::failure();

  if (parser.resolveOperands(valueOperands, valueTypes, valueOperandsLoc,
                             result.operands))
    return ::mlir::failure();

  return ::mlir::success();
}

void SetOp::print(::mlir::OpAsmPrinter &_odsPrinter) {
  _odsPrinter << ' ';
  _odsPrinter << getObject();
  _odsPrinter << ',';
  _odsPrinter << ' ';
  _odsPrinter.printStrippedAttrOrType(getFieldAttr());
  _odsPrinter << ',';
  _odsPrinter << ' ';
  _odsPrinter << getValue();
  ::llvm::SmallVector<::llvm::StringRef, 2> elidedAttrs;
  elidedAttrs.push_back("field");
  _odsPrinter.printOptionalAttrDict((*this)->getAttrs(), elidedAttrs);
}

void LeanDialect::addOpsImpl() {
  addOperations<
#define GET_OP_LIST
#include "Lean/IR/LeanOps.cpp.inc"
      >();
}
} // namespace mlir::lean
