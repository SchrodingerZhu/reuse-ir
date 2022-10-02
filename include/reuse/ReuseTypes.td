#ifndef REUSE_TYPES
#define REUSE_TYPES

include "ReuseBase.td"
include "mlir/IR/AttrTypeBase.td"

class Reuse_Type<string name, string typeMnemonic> : TypeDef<Reuse_Dialect, name> {
  let mnemonic = typeMnemonic;
}

def Reuse_ObjectType : Reuse_Type<"Object", "object"> {

  let summary = "Reuse IR object managed by reference counting";
  let description = [{
    `reuse.object` represents an object managed by reference counting.
  }];

  let parameters = (ins
    ArrayRefParameter<"::mlir::Type", "members">:$members,
    "::mlir::StringAttr":$constructor,
    "::mlir::IntegerAttr":$size,
    "::mlir::IntegerAttr":$alignment
  );

  let assemblyFormat = "`<` `size` `:` $size `,` `alignment` `:` $alignment `,` $constructor `{` $members `}` `>`";
}


def Reuse_RcPtrType :
    Reuse_Type<"RcPtr", "rc_ptr"> {
  let summary = "Reuse IR reference counting pointer";
  let description = [{
    `reuse.rc` is the type for reference counting pointer.
  }];

  let parameters = (ins 
    Reuse_ObjectType:$pointee,
    "::mlir::BoolAttr":$atomic
  );

  let assemblyFormat = "`<` `atomic` `:` $atomic `,` $pointee `>`";
}

#endif // REUSE_TYPES
