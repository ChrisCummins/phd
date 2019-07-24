// Read a bitcode, run optimisations on it, print it to stderr.
#include <iostream>
#include <memory>

#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/LegacyPassManager.h"
#include "llvm/IR/Module.h"
#include "llvm/IRReader/IRReader.h"
#include "llvm/Pass.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Transforms/IPO/PassManagerBuilder.h"

#include "phd/app.h"
#include "phd/logging.h"

DEFINE_string(input_bitcode, "", "Path of input LLVM bitcode file.");

std::unique_ptr<llvm::Module> ParseIRFileOrDie(const string& path,
                                               llvm::LLVMContext& context) {
  llvm::SMDiagnostic error;
  std::unique_ptr<llvm::Module> module =
      llvm::parseIRFile(path, error, context);

  if (!module) {
    LOG(FATAL) << "Failed to load bitcode file " << path;
  }

  return module;
}

int main(int argc, char** argv) {
  phd::InitApp(&argc, &argv);

  LOG(INFO) << "reading";

  if (FLAGS_input_bitcode.empty()) {
    LOG(FATAL) << "--input_bitcode not specified";
  }

  llvm::LLVMContext context;
  auto module = ParseIRFileOrDie(FLAGS_input_bitcode, context);

  llvm::legacy::PassManager mpm;
  llvm::PassManagerBuilder pass_manager_builder;
  pass_manager_builder.OptLevel = 3;
  pass_manager_builder.populateModulePassManager(mpm);

  mpm.run(*module);
  module->print(llvm::errs(), /*AAW=*/nullptr);

  LOG(INFO) << "done";

  return 0;
}
