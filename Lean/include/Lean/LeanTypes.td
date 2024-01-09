#ifndef LEAN_TYPES
#define LEAN_TYPES

include "Lean/LeanDialect.td"
include "mlir/Interfaces/DataLayoutInterfaces.td"
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
    `lean.obj<O, S>` is a Lean object
          type.It has two extra parameters `O` and `S`.
    `O` specifies the number of member objects,
      and `S` specifies the size of all scalar fields.}];
  let parameters = (ins "unsigned" : $subObjs, "unsigned" : $scalaSize);
  let assemblyFormat = "`<` $subObjs `,` $scalaSize `>`";
}

#endif // REFCNT_TYPES
