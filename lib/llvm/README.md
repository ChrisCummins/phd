# LLVM Builds

Clone LLVM, clang, and compiler-rt sources here:
```
src/
src/tools/clang
src/projects/compiler-rt
```

Build here:

```
$ mkdir build/x.y.z/
$ cd build/x.y.z
$ cmake ../../src -G Ninja -DLLVM_TARGETS_TO_BUILD="X86"
$ ninja
```

This patch was required to build `release_35` (3.5.2): http://lists.llvm.org/pipermail/llvm-bugs/2014-July/034874.html
