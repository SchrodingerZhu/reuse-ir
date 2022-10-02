#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/OpImplementation.h"
#include "reuse/ReuseBase.h"
#include <reuse/ReuseOps.h>
#include <mlir/IR/MLIRContext.h>
int main() {
  mlir::MLIRContext context;
  context.getOrLoadDialect<reuse::ReuseDialect>();
  auto int_type = mlir::IntegerType::get(&context, 64);
  auto object = reuse::ObjectType::get(&context, {},
                                       mlir::StringAttr::get(&context, "Cons"),
                                       mlir::IntegerAttr::get(int_type, 16),
                                       mlir::IntegerAttr::get(int_type, 16));
  object.dump();
}