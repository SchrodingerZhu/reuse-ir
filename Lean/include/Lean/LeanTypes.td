#ifndef LEAN_TYPES
#define LEAN_TYPES

include "Lean/LeanDialect.td"
include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/BuiltinTypes.td"

class Lean_Type<string name, string typeMnemonic, list<Trait> traits = []>
    : TypeDef<Lean_Dialect, name, traits> {
  let mnemonic = typeMnemonic;
}

def Lean_ObjType
    : Lean_Type<"Obj", "obj"> {
  let summary = "Lean object type";
  let description = [{
    `lean.obj` is a Lean object. The IR uses `lean.obj` in an opaque way. Ctor information
    can be added as an attribute.
  }];
}

#endif // REFCNT_TYPES
