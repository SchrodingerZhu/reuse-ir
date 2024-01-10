#ifndef LEAN_OPS
#define LEAN_OPS
include "Lean/LeanDialect.td"
include "Lean/LeanTypes.td"
include "Lean/LeanAttributes.td"
include "Refcnt/RefcntOps.td"

class Lean_Op<string mnemonic, list<Trait> traits = []>
    : Op<Lean_Dialect, mnemonic, traits>;

def TagOp : Lean_Op<"tag"> {
  let summary = "get tag of a Lean object";
  let description = [{
    `lean.tag` get tag of a Lean object.
    The object must be a RC pointer to a Lean object.

    ```mlir
    lean.tag %obj : !refcnt.rc<!lean.obj>
    ```
    ```
  }];

  let arguments = (
    ins Arg<Refcnt_RcType, "the RC pointer to the Lean object", []>:$object
  );

  let results = (outs Builtin_Index : $tag);

  let assemblyFormat = "$object attr-dict `:` type($object)";

  let hasVerifier = 1;
}

def ProjOp : Lean_Op<"proj"> {
  let summary = "project a field of a Lean object";
  let description = [{
    `lean.proj` project a field of a Lean object.
    The object must be a RC pointer to a Lean object.
    
    ```mlir
    lean.proj %obj : refcnt.rc<Obj<A, B>>, refcnt.rc<Obj<C, D>>, 0 : index
    ```
  }];
  let arguments = (
     ins Arg<Refcnt_RcType, "the RC pointer to the Lean object", []>:$object,
     Builtin_IntegerAttr:$field
   );
  let results = (outs Res<Refcnt_RcType, "the RC pointer to the projected object", []>:$result);
  let assemblyFormat =
      "$object `:` type($object) `,` type($result) `,` $field attr-dict";
  let hasVerifier = 1;
}

def SProjOp : Lean_Op<"sproj"> {
  let summary = "project a scalar field of a Lean object";
  let description = [{
    `lean.sproj` project a scalar field of a Lean object.
    The object must be a RC pointer to a Lean object.
        
    ```mlir
    lean.sproj %obj : refcnt.rc<Obj<A, B>>, u8, 0 : index
    ```
  }];
  let arguments = (
    ins Arg<Refcnt_RcType, "the RC pointer to the Lean object", []>:$object,
    Builtin_IntegerAttr:$offset
  );
  let results = (outs AnyType:$result);
  let assemblyFormat =
    "$object `:` type($object) `,` type($result) `,` $offset attr-dict";
  let hasVerifier = 1;
}

#endif // LEAN_OPS
