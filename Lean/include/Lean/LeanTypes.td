#ifndef LEAN_TYPES 
#define LEAN_TYPES

include "Lean/LeanDialect.td"
include "mlir/Interfaces/DataLayoutInterfaces.td"
include "mlir/IR/AttrTypeBase.td"

class Lean_Type<string name, string typeMnemonic, list<Trait> traits = []> :
    TypeDef<Lean_Dialect, name, traits> {
  let mnemonic = typeMnemonic;
}

def Lean_UIntType : Lean_Type<"UInt", "uint",
    [DeclareTypeInterfaceMethods<DataLayoutTypeInterface>]> {
  let summary = "Plain unsigned integer type";
  let description = [{
    `lean.uint<W>` is a plain unsigned integer type. It has an extra parameter
    `W` that specifies the bitwidth of the integer.
  }];
  let parameters = (ins "unsigned":$width);
  let assemblyFormat = "`<` $width `>`";
}

def Lean_ObjType : Lean_Type<"Obj", "obj",
    [DeclareTypeInterfaceMethods<DataLayoutTypeInterface>]> {
  let summary = "Lean object type";
  let description = [{
    `lean.obj<O, S>` is a Lean object type. It has two extra parameters `O` and `S`.
    `O` specifies the number of member objects, and `S` specifies the size of all
    scalar fields.
  }];
  let parameters = (ins "unsigned":$subObjs, "unsigned":$scalaSize);
  let assemblyFormat = "`<` $subObjs `,` $scalaSize `>`";
}


#endif // REFCNT_TYPES
