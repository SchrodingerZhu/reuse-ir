#include <reuse/ReuseBase.h>
#include <reuse/ReuseOpsDialect.cpp.inc>
#include <reuse/ReuseTypes.h>
#include <reuse/ReuseOps.h>

#define GET_TYPEDEF_CLASSES
#include <reuse/ReuseOpsTypes.cpp.inc>

namespace reuse {
void ReuseDialect::initialize() {
  addTypes<
#define GET_TYPEDEF_LIST
#include <reuse/ReuseOpsTypes.cpp.inc>
      >();
  addOperations<
#define GET_OP_LIST
#include <reuse/ReuseOps.cpp.inc>
      >();
}
} // namespace reuse