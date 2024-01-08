#ifndef REFCNT_OPS
#define REFCNT_OPS

include "Refcnt/RefcntDialect.td"
include "Refcnt/RefcntTypes.td"

class Refcnt_Op<string mnemonic, list<Trait> traits = []> :
    Op<Refcnt_Dialect, mnemonic, traits>;

def IncOp : Refcnt_Op<"inc"> {

  let summary = "Increase the reference count of a value";
  let description = [{
    `refcnt.inc` increases the reference count of a value by one. The value must be 
    of `refcnf.rc` type.

    Example:

    ```mlir
    refcnt.inc %0 : refcnt.ptr<T>
    ```
  }];

  let arguments = (ins Arg<Refcnt_RcType, "the RC pointer", []>:$value);

  let assemblyFormat =
      "$value attr-dict `:` type($value)";
}

def DecOp : Refcnt_Op<"dec"> {
  let summary = "Decreaase the reference count of a value";
  let description = [{
    `refcnt.dec` decreases the reference count of a value by one. The value must be 
    of `refcnf.rc` type.

    Example:

    ```mlir
    refcnt.dec %0 : refcnt.ptr<T>
    ```
  }];

  let arguments = (ins Arg<Refcnt_RcType, "the RC pointer", []>:$value);

  let assemblyFormat =
      "$value attr-dict `:` type($value)";
}


def DropOp : Refcnt_Op<"drop"> {
  let summary = "Decrease the reference count of a value, but also generate a memory token";
  let description = [{
    `refcnt.drop` decreases the reference count of a value by one. The value must be 
    of `refcnf.rc` type. Different from `refcnt.dec`, it also generates a memory token.

    Example:

    ```mlir
    %1 = refcnt.drop %0 : refcnt.rc<T> `,` refcnt.token<T>
    ```
  }];

  let arguments = (ins Arg<Refcnt_RcType, "the RC pointer", []>:$value);
  let results = (outs Arg<Refcnt_TokenType, "the memory token", []>:$token);

  let assemblyFormat =
      "$value attr-dict `:` type($value) `,` type($token)"; 

  let hasVerifier = 1;
}

def NewOp : Refcnt_Op<"new"> {
  let summary = "Allocate new memory";
  let description = [{
    `refcnt.new` allocates a new memory block for a given type.

    Example:

    ```mlir
    %0 = refcnt.new : refcnt.rc<T>
    ```
  }];

  let results = (outs Arg<Refcnt_RcType, "allocation result", []>:$result);

  let assemblyFormat =
      "attr-dict `:` type($result)"; 

  let hasVerifier = 1;
}

#endif // REFCNT_OPS
