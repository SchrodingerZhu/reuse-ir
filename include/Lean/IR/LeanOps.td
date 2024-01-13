#ifndef LEAN_OPS
#define LEAN_OPS
include "Lean/IR/LeanDialect.td"
include "Lean/IR/LeanTypes.td"
include "Lean/IR/LeanAttributes.td"
include "Refcnt/IR/RefcntOps.td"

class Lean_Op<string mnemonic, list<Trait> traits = []>
    : Op<Lean_Dialect, mnemonic, traits>;

def TagOp : Lean_Op<"tag"> {
  let summary = "get tag of a Lean object";
  let description = [{
    `lean.tag` get tag of a Lean object.
    The object must be a RC pointer to a Lean object.

    ```mlir
    lean.tag %obj
    ```
  }];

  let arguments = (
    ins Arg<Refcnt_RcType, "the RC pointer to the Lean object", []>:$object
  );

  let results = (outs Builtin_Integer : $tag);
  let hasCustomAssemblyFormat = 1;
  let hasVerifier = 1;

}

def SetOp : Lean_Op<"set"> {
  let summary = "set field value of a Lean object";
  let description = [{
    `lean.set` set field value of a Lean object.
    The object must be a RC pointer to a Lean object.
    The field must be a RC pointer to a Lean object.
    
    ```mlir
    lean.set %obj, 0 : index, %value
    ```
  }];
  let arguments = (
     ins Arg<Refcnt_RcType, "the RC pointer to the Lean object", []>:$object,
     Builtin_IntegerAttr:$field,
    Arg<Refcnt_RcType, "the RC pointer to the Lean object", []>:$value
   );
  let hasCustomAssemblyFormat = 1;
  let hasVerifier = 1;
}

def SSetOp : Lean_Op<"sset"> {
  let summary = "set scalar field value of a Lean object";
  let description = [{
    `lean.sset` set field value of a Lean object.
    The object must be a RC pointer to a Lean object.
    The field must be a scalar.
    
    ```mlir
    lean.sset %obj, 0 : index, %value : f64
    ```
  }];
  let arguments = (
     ins Arg<Refcnt_RcType, "the RC pointer to the Lean object", []>:$object,
     Builtin_IntegerAttr:$offset,
     Arg<AnyType, "the scalar value to set", []>:$value
   );
  let hasCustomAssemblyFormat = 1;
  let hasVerifier = 1;
}

def ProjOp : Lean_Op<"proj"> {
  let summary = "project a field of a Lean object";
  let description = [{
    `lean.proj` project a field of a Lean object.
    The object must be a RC pointer to a Lean object.
    
    ```mlir
    lean.proj %obj, 0 : index
    ```
  }];
  let arguments = (
     ins Arg<Refcnt_RcType, "the RC pointer to the Lean object", []>:$object,
     Builtin_IntegerAttr:$field
   );
  let results = (outs Res<Refcnt_RcType, "the RC pointer to the projected object", []>:$result);
  let hasCustomAssemblyFormat = 1;
  let hasVerifier = 1;
}

def SProjOp : Lean_Op<"sproj"> {
  let summary = "project a scalar field of a Lean object";
  let description = [{
    `lean.sproj` project a scalar field of a Lean object.
    The object must be a RC pointer to a Lean object.
        
    ```mlir
    lean.sproj %obj : !refcnt.rc<!lean.obj>, u8, 0 : index
    ```
  }];
  let arguments = (
    ins Arg<Refcnt_RcType, "the RC pointer to the Lean object", []>:$object,
    Builtin_IntegerAttr:$offset
  );
  let results = (outs AnyType:$result);
  let hasCustomAssemblyFormat = 1;
  let hasVerifier = 1;
}

def AppOp : Lean_Op<"app"> {
  let summary = "apply a Lean function to a list of arguments";
  let description = [{
    `lean.app` apply a Lean function to a list of arguments.
    The function must be a RC pointer to a Lean object.
    The arguments must be RC pointers to Lean objects.
    
    ```mlir
    %0 = lean.app %fn (%arg0, %arg1)
    ```
  }];
  let arguments = (
    ins Arg<Refcnt_RcType, "the RC pointer to the Lean function", []>:$fn,
    Variadic<Refcnt_RcType>:$args
  );
  let results = (outs Res<Refcnt_RcType, "the RC pointer to the result", []>:$result);
  let hasCustomAssemblyFormat = 1;
  let hasVerifier = 1;
}

#endif // LEAN_OPS
