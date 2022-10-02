# Reuse IR

An on-going research to realize reuse analysis with SSA-flavor IR, using MLIR techniques.

## How to compile

Unfortunately, most distros do not distribute MLIR with LLVM stack. 
It is very likely you will need to compile LLVM with MLIR support on your own.

```bash
git clone https://github.com/llvm/llvm-project
cd llvm-project
mkdir build
cd build && cmake ../llvm -DCMAKE_BUILD_TYPE=RelWithDebInfo -DLLVM_ENABLE_PROJECTS=mlir
cmake --build . -j
cmake --install . --prefix /the/directory/for/llvm/installation
```

Then, configure this project with the LLVM you just compiled:

```bash
cmake /path/to/this/project \
  -DMLIR_ROOT=/the/directory/for/llvm/installation \
  -DLLVM_ROOT=/the/directory/for/llvm/installation
```
