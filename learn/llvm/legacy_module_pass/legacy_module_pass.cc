// "Hello world" module pass. It prints the name of the module that it visits.
#include <iostream>
#include <memory>

#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/LegacyPassManager.h"
#include "llvm/IR/Module.h"
#include "llvm/Pass.h"
#include "llvm/Transforms/IPO/PassManagerBuilder.h"

#include "labm8/cpp/app.h"
#include "labm8/cpp/logging.h"

namespace llvm {
namespace {

// A simple module pass that prints the name of the module.
struct HelloModulePass : public ModulePass {
  static char ID;

  HelloModulePass() : ModulePass(ID) {}

  bool runOnModule(Module& module) override {
    LOG(INFO) << "Hello, " << module.getName();
    return /*modified=*/false;
  }
};

char HelloModulePass::ID = 0;

}  // namespace
}  // namespace llvm

int main(int argc, char** argv) {
  phd::InitApp(&argc, &argv);

  llvm::legacy::PassManager pm;
  llvm::PassManagerBuilder pmb;
  pmb.populateModulePassManager(pm);

  // Ownership of this pointer is transferred to legacy::PassManager on add().
  auto hello_pass = new llvm::HelloModulePass();
  pm.add(hello_pass);

  llvm::LLVMContext ctx;

  llvm::Module m("my_module", ctx);
  pm.run(m);

  LOG(INFO) << "done";
  return 0;
}
