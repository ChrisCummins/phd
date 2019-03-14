// opencl_kernel_features: Extract OpenCL kernel features.
//
// An implementation of the feature extractor used in:
//
//     Grewe, D., Wang, Z., & O’Boyle, M. F. P. M. (2013). Portable
//     mapping of data parallel programs to OpenCL for heterogeneous
//     systems. In CGO. IEEE.
//
// Extracts static features from OpenCL source files.
//
//     Usage: ./features [-header-only] [-extra-arg=<arg> ...] <file> [files
//     ...]
//
// Output is comma separated values in the following format:
//
//     file              file path
//     kernel            name of the kernel
//     comp              # compute operations
//     rational          # rational operations
//     mem               # accesses to global memory
//     localmem          # accesses to local memory
//     coalesced         # coalesced memory accesses
//     atomic            # atomic operations
//     F2:coalesced/mem  derived feature
//     F4:comp/mem       derived feature
//
// Originally written by Zheng Wang <z.wang@lancaster.ac.uk>.
//
// Copyright 2016, 2017, 2018, 2019 Chris Cummins <chrisc.101@gmail.com>.
//
// This file is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// This is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with this program.  If not, see <https://www.gnu.org/licenses/>.
//
#include <stdio.h>
#include <algorithm>
#include <array>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <sstream>
#include <string>
#include <system_error>
#include <vector>

#include <dirent.h>
#include <limits.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>

#include "clang/AST/ASTConsumer.h"
#include "clang/AST/ASTContext.h"
#include "clang/AST/ParentMap.h"
#include "clang/AST/RecursiveASTVisitor.h"
#include "clang/AST/Stmt.h"
#include "clang/Basic/Diagnostic.h"
#include "clang/Basic/DiagnosticOptions.h"
#include "clang/Basic/FileManager.h"
#include "clang/Basic/SourceManager.h"
#include "clang/Basic/TargetInfo.h"
#include "clang/Basic/TargetOptions.h"
#include "clang/Frontend/CompilerInstance.h"
#include "clang/Frontend/TextDiagnosticPrinter.h"
#include "clang/Lex/Lexer.h"
#include "clang/Lex/Preprocessor.h"
#include "clang/Lex/PreprocessorOptions.h"
#include "clang/Parse/ParseAST.h"
#include "clang/Rewrite/Core/Rewriter.h"
#include "clang/Rewrite/Frontend/Rewriters.h"

class ParameterInfo {
 public:
  enum VarType { GLOBAL, LOCAL, OTHER };

 private:
  std::string varName;
  VarType type;

 public:
  ParameterInfo(std::string _varName, VarType _type) {
    varName = _varName;
    type = _type;
  }

  std::string getVarName() { return varName; }
  VarType getType() { return type; }
};

class FuncInfo {
  std::string kernelName;
  std::vector<ParameterInfo> vp;
  unsigned global_mem_ls;
  unsigned comp_inst_count;
  unsigned rational_inst_count;
  unsigned barrier_count;
  unsigned cols_mem_access_count;
  unsigned local_mem_ls_count;
  unsigned atomic_op_count;
  bool isOCLKernel;

 public:
  FuncInfo()
      : global_mem_ls(0),
        comp_inst_count(0),
        rational_inst_count(0),
        barrier_count(0),
        cols_mem_access_count(0),
        local_mem_ls_count(0),
        atomic_op_count(0),
        isOCLKernel(false) {}

  explicit FuncInfo(std::string _kernelName)
      : kernelName(_kernelName),
        global_mem_ls(0),
        comp_inst_count(0),
        rational_inst_count(0),
        barrier_count(0),
        cols_mem_access_count(0),
        local_mem_ls_count(0),
        atomic_op_count(0),
        isOCLKernel(false) {}

  void setAsOclKernel() { isOCLKernel = true; }

  bool isOclKernel() { return isOCLKernel; }

  void resetCounters() {
    global_mem_ls = 0;
    comp_inst_count = 0;
    rational_inst_count = 0;
    barrier_count = 0;
    cols_mem_access_count = 0;
    local_mem_ls_count = 0;
    atomic_op_count = 0;
  }

  void addCountersFromOtherFunc(FuncInfo *KI) {
    this->global_mem_ls += KI->getGlobalMemLSCount();
    this->comp_inst_count += KI->getCompInstCount();
    this->rational_inst_count += KI->getRationalInstCount();
    this->barrier_count += KI->getBarrierCount();
    this->cols_mem_access_count += KI->getColMemAccessCount();
    this->local_mem_ls_count += KI->getLocalMemLSCount();
    this->atomic_op_count += KI->getAtomicOpCount();
  }

  void setFuncName(std::string name) { this->kernelName = name; }

  std::string getFuncName() { return kernelName; }

  void addParameter(ParameterInfo p) { vp.push_back(p); }

  unsigned getParameterNum() { return static_cast<unsigned>(vp.size()); }

  ParameterInfo &getParameter(unsigned i) { return vp[i]; }

  void incrLocalMemLSCount() { local_mem_ls_count++; }

  void incrAtomicOpCount() { atomic_op_count++; }

  void incrBarrierCount() { barrier_count++; }

  void addComputationCount(unsigned c) { comp_inst_count += c; }

  void incrGlobalMemLSCount() { global_mem_ls++; }

  void incrCompInstCount() { comp_inst_count++; }

  void incrRationalInstCount() { rational_inst_count++; }

  void incrColMemAccessCount() { cols_mem_access_count++; }

  unsigned getAtomicOpCount() { return atomic_op_count; }

  unsigned getBarrierCount() { return barrier_count; }

  unsigned getLocalMemLSCount() { return local_mem_ls_count; }

  unsigned getColMemAccessCount() { return cols_mem_access_count; }

  unsigned getCompInstCount() { return comp_inst_count; }

  unsigned getRationalInstCount() { return rational_inst_count; }

  unsigned getGlobalMemLSCount() { return global_mem_ls; }

  bool isGlobalVar(std::string var) {
    for (unsigned i = 0; i < vp.size(); i++) {
      if (vp[i].getVarName() == var) return true;
    }

    return false;
  }
};

//
// RecursiveASTVisitor --- the big-kahuna visitor that traverses
// everything in the AST.
//
class RecursiveASTVisitor
    : public clang::RecursiveASTVisitor<RecursiveASTVisitor> {
 private:
  std::vector<FuncInfo *> FuncInfoVec;
  FuncInfo *pCurKI;

  // Return a boolean value to indicate if the array access is coalescaed.
  bool processArrayIndices(clang::ArraySubscriptExpr *Node) {
    clang::Expr *tExpr = Node->getRHS();

    // Check if this is a binary operator
    clang::BinaryOperator *bo = clang::dyn_cast<clang::BinaryOperator>(tExpr);
    if (bo) {
      return processArrayIdxBinaryOperator(bo);
    }

    // If the array index is determined by a function call, assume it is
    // not a coalesced access.
    clang::CallExpr *ce = clang::dyn_cast<clang::CallExpr>(tExpr);
    if (ce) {
      return false;
    }

    clang::ImplicitCastExpr *iCE =
        clang::dyn_cast<clang::ImplicitCastExpr>(tExpr);
    if (iCE) {
      clang::DeclRefExpr *DRE =
          clang::dyn_cast<clang::DeclRefExpr>(iCE->getSubExpr());
      // The array is indexed using a variable
      if (DRE) {
        // Assuming this is colescate memory acessing
        return true;
      }
    }

    clang::IntegerLiteral *IntV = clang::dyn_cast<clang::IntegerLiteral>(tExpr);
    if (IntV) return true;

    clang::ArraySubscriptExpr *aExpr =
        clang::dyn_cast<clang::ArraySubscriptExpr>(tExpr);
    if (aExpr) {
      return processArrayIndices(aExpr);
    }

    return true;
  }

  bool processArrayIdxBinaryOperator(clang::BinaryOperator *bo) {
    if (!(bo->isMultiplicativeOp() || bo->isAdditiveOp())) return false;

    clang::Expr *lhs = bo->getLHS();
    clang::Expr *rhs = bo->getRHS();

    clang::IntegerLiteral *lhsi = clang::dyn_cast<clang::IntegerLiteral>(lhs);
    clang::IntegerLiteral *rhsi = clang::dyn_cast<clang::IntegerLiteral>(rhs);

    // If the index is a constant value, return true
    if (lhsi && rhsi) {
      return true;
    }

    if (lhsi) {
      std::string v = lhsi->getValue().toString(10, /*isSigned*/ false);

      // If the step is 1, or zero colescated
      if (v == "1" || v == "0") {
        return true;
      }
    }

    if (rhsi) {
      std::string v = rhsi->getValue().toString(10, /*isSigned*/ false);

      // If the step is 1, or zero colescated
      if (v == "1" || v == "0") {
        return true;
      }
    }

    return false;
  }

  FuncInfo *findFunctionInfo(std::string funcName) {
    for (unsigned i = 0; i < FuncInfoVec.size(); i++) {
      if (FuncInfoVec[i]->getFuncName() == funcName) return FuncInfoVec[i];
    }

    return NULL;
  }

  void InitializeOCLRoutines() {
    // Utility macros.
#define DEFAULT_OCL_ROUTINE_COUNT 15
#define ADD_OCL_ROUTINE_INFO(__p__, __name__)              \
  {                                                        \
    FuncInfo *__p__ = new FuncInfo(__name__);              \
    __p__->addComputationCount(DEFAULT_OCL_ROUTINE_COUNT); \
    FuncInfoVec.push_back(__p__);                          \
  }

    // The 'stuff'.
    ADD_OCL_ROUTINE_INFO(p, "exp");
    ADD_OCL_ROUTINE_INFO(p, "log");
    ADD_OCL_ROUTINE_INFO(p, "sqrt");

    // Clean up.
#undef DEFAULT_OCL_ROUTINE_COUNT
#undef ADD_OCL_ROUTINE_INFO
  }

  bool isBuiltInAtomicFunc(std::string func) {
    if (func == "atomic_add" || func == "atomic_and" ||
        func == "atomic_cmpxchg" || func == "atomic_compare_exchange_strong" ||
        func == "atomic_compare_exchange_strong_explicit" ||
        func == "atomic_compare_exchange_weak" ||
        func == "atomic_compare_exchange_weak_explicit" ||
        func == "atomic_dec" || func == "atomic_exchange" ||
        func == "atomic_exchange_explicit" || func == "atomic_fetch_add" ||
        func == "atomic_fetch_add_explicit" || func == "atomic_fetch_and" ||
        func == "atomic_fetch_and_explicit" || func == "atomic_fetch_max" ||
        func == "atomic_fetch_max_explicit" || func == "atomic_fetch_min" ||
        func == "atomic_fetch_min_explicit" || func == "atomic_fetch_or" ||
        func == "atomic_fetch_or_explicit" || func == "atomic_fetch_sub" ||
        func == "atomic_fetch_sub_explicit" || func == "atomic_fetch_xor" ||
        func == "atomic_fetch_xor_explicit" || func == "atomic_flag_clear" ||
        func == "atomic_flag_clear_explicit" ||
        func == "atomic_flag_test_and_set" ||
        func == "atomic_flag_test_and_set_explicit" || func == "atomic_inc" ||
        func == "atomic_init" || func == "atomic_load" ||
        func == "atomic_load_explicit" || func == "atomic_max" ||
        func == "atomic_min" || func == "atomic_or" || func == "atomic_store" ||
        func == "atomic_store_explicit" || func == "atomic_sub" ||
        func == "atomic_work_item_fence" || func == "atomic_xchg" ||
        func == "atomic_xor") {
      return true;
    }

    return false;
  }

 public:
  RecursiveASTVisitor() {
    pCurKI = NULL;
    InitializeOCLRoutines();
  }

  // Override Statements which includes expressions and more
  bool VisitStmt(clang::Stmt *s) {
    return true;  // returning false aborts the traversal
  }

  bool VisitFunctionDecl(clang::FunctionDecl *f) {
    std::string Proto = f->getNameInfo().getAsString();
    pCurKI = new FuncInfo(Proto);
    pCurKI->resetCounters();

    unsigned up = f->getNumParams();
    for (unsigned i = 0; i < up; i++) {
      clang::ParmVarDecl *pD = f->getParamDecl(i);
      clang::QualType T =
          pD->getTypeSourceInfo()
              ? pD->getTypeSourceInfo()->getType()
              : pD->getASTContext().getUnqualifiedObjCPointerType(
                    pD->getType());

      std::string varName = pD->getIdentifier()->getName();
      std::string tStr = T.getAsString();

      ParameterInfo::VarType vT = ParameterInfo::OTHER;
      // Pattern match the string representation of parameter qualifiers. This
      // is an ugly hack, which I would prefer a better solution for.
      if (tStr.find("__global") != std::string::npos &&
          tStr.find("global") != std::string::npos) {
        pCurKI->setAsOclKernel();
        vT = ParameterInfo::GLOBAL;
      } else if (tStr.find("__local") != std::string::npos &&
                 tStr.find("local") != std::string::npos) {
        vT = ParameterInfo::LOCAL;
      }

      if (vT != ParameterInfo::OTHER) {
        ParameterInfo pi(varName, vT);
        pCurKI->addParameter(pi);
      }
    }

    if (f->hasBody()) {
      FuncInfoVec.push_back(pCurKI);
    }

    // Returning True to abort the traversal.
    return true;
  }

  // Override Binary Operator expressions
  clang::Expr *VisitBinaryOperator(clang::BinaryOperator *E) {
    if (pCurKI) {
      if (E->isComparisonOp()) {
        pCurKI->incrRationalInstCount();
      } else if (E->isMultiplicativeOp() || E->isAdditiveOp() ||
                 E->isShiftOp() || E->isBitwiseOp() || E->isShiftAssignOp()) {
        pCurKI->incrCompInstCount();
      }
    }

    return E;
  }

  // VISIT Declare Exprs to record load and store to global memory variables
  bool VisitDeclRefExpr(clang::DeclRefExpr *Node) {
    if (pCurKI) {
      std::string varName = Node->getNameInfo().getAsString();
      if (pCurKI->isGlobalVar(varName)) {
        pCurKI->incrGlobalMemLSCount();
      }
    }
    return true;
  }

  bool VisitVarDecl(clang::VarDecl *D) {
    clang::QualType T =
        D->getTypeSourceInfo()
            ? D->getTypeSourceInfo()->getType()
            : D->getASTContext().getUnqualifiedObjCPointerType(D->getType());

    std::string varName = D->getIdentifier()->getName();
    std::string tStr = T.getAsString();

    if (tStr.find("__local") != std::string::npos) {
      if (pCurKI) {
        ParameterInfo pi(varName, ParameterInfo::LOCAL);
        pCurKI->addParameter(pi);
      }
    }

    return true;
  }

  bool VisitCallExpr(clang::CallExpr *E) {
    clang::FunctionDecl *D = E->getDirectCallee();
    if (!D) return true;

    std::string varName = D->getNameInfo().getAsString();

    if (isBuiltInAtomicFunc(varName)) {
      pCurKI->incrAtomicOpCount();
    } else if (varName == "barrier") {
      pCurKI->incrBarrierCount();
    } else {
      // Add up the counters if a function is called
      FuncInfo *pi = findFunctionInfo(varName);
      if (pi) {
        pCurKI->addCountersFromOtherFunc(pi);
      }
    }

    return true;
  }

  // This is use to count coalesced memory accesses
  // This is the most complex function
  bool VisitArraySubscriptExpr(clang::ArraySubscriptExpr *Node) {
    if (!pCurKI) return true;

    ParameterInfo::VarType vT = ParameterInfo::OTHER;

    // Retrive array type
    clang::Expr *tExpr = Node->getLHS();
    if (tExpr) {
      clang::ImplicitCastExpr *iExpr =
          clang::dyn_cast<clang::ImplicitCastExpr>(tExpr);
      if (!iExpr) return true;

      clang::DeclRefExpr *DRE =
          clang::dyn_cast<clang::DeclRefExpr>(iExpr->getSubExpr());
      if (DRE) {
        clang::ValueDecl *D = DRE->getDecl();
        clang::QualType T = D->getType();

        std::string varName = D->getIdentifier()->getName();
        std::string tStr = T.getAsString();

        // FIXME:Urgly tricks
        if (tStr.find("__global") != std::string::npos) {
          vT = ParameterInfo::GLOBAL;
        } else if (tStr.find("__local") != std::string::npos) {
          vT = ParameterInfo::LOCAL;
        }
      }
    }

    // Calculate coalesced global memory accesses
    if (vT == ParameterInfo::GLOBAL) {
      if (processArrayIndices(Node)) {
        pCurKI->incrColMemAccessCount();
      }
    } else if (vT == ParameterInfo::LOCAL) {
      pCurKI->incrLocalMemLSCount();
    }

    return true;
  }

  std::vector<FuncInfo *> &getFuncInfo() { return FuncInfoVec; }
  ~RecursiveASTVisitor() {
    for (unsigned i = 0; i < FuncInfoVec.size(); i++) {
      delete FuncInfoVec[i];
    }
  }
};

//
// AST Consumer.
//
class ASTConsumer : public clang::ASTConsumer {
 public:
  ASTConsumer() : rv() {}

  virtual bool HandleTopLevelDecl(clang::DeclGroupRef d) {
    typedef clang::DeclGroupRef::iterator iter;

    for (iter b = d.begin(), e = d.end(); b != e; ++b) {
      rv.TraverseDecl(*b);
    }

    return true;  // keep going
  }

  RecursiveASTVisitor rv;

  void dumpKernelFeatures(std::string fileName, std::ostream &fout) {
    std::vector<FuncInfo *> &FuncInfoVec = rv.getFuncInfo();

    for (unsigned i = 0; i < FuncInfoVec.size(); i++) {
      if (FuncInfoVec[i]->isOclKernel()) {
        // Derived features:
        const auto F2 = FuncInfoVec[i]->getColMemAccessCount() /
                        static_cast<float>(std::max(
                            FuncInfoVec[i]->getGlobalMemLSCount(), 1u));
        const auto F4 = FuncInfoVec[i]->getCompInstCount() /
                        static_cast<float>(std::max(
                            FuncInfoVec[i]->getGlobalMemLSCount(), 1u));

        fout << fileName << "," << FuncInfoVec[i]->getFuncName() << ","
             << FuncInfoVec[i]->getCompInstCount() << ","
             << FuncInfoVec[i]->getRationalInstCount() << ","
             << FuncInfoVec[i]->getGlobalMemLSCount() << ","
             << FuncInfoVec[i]->getLocalMemLSCount() << ","
             << FuncInfoVec[i]->getColMemAccessCount() << ","
             << FuncInfoVec[i]->getAtomicOpCount() << ","
             << std::setprecision(3) << F2 << "," << F4 << "\n";
      }
    }
  }
};

std::string basename(const std::string &path) {
  size_t i = path.length() - 1u;

  while (i) {
    if (path[i] == '/') break;
    --i;
  }

  return path[i] == '/' ? path.substr(i + 1, path.length()) : path;
}

std::string dirname(const std::string &path) {
  size_t i = path.length() - 1u;

  while (i) {
    if (path[i] == '/') break;
    --i;
  }

  return path.substr(0, static_cast<size_t>(i));
}

// Platform specific code. Provide Mac and Linux implementations.
#ifdef __APPLE__  // Mac OS X
#include <mach-o/dyld.h>
std::string getexepath() {
  char path[1024];
  uint32_t size = sizeof(path);
  return _NSGetExecutablePath(path, &size) == 0 ? path : "BAD";
}
#elif __linux__  // Linux
std::string getexepath() {
  char result[PATH_MAX];
  auto count = readlink("/proc/self/exe", result, PATH_MAX);
  return count > 0 ? result : "BAD";
}
#else
#error "Unsupported platform!"
#endif
// End platform specific code.

//
// Extract features from kernels in an OpenCL program.
//
// @param path Path to OpenCL program.
// @param out Stream to print features to.
//
void extract_features(
    std::string path, std::string cl_header_path, std::ostream &out,
    const std::vector<std::string> &extra_args = std::vector<std::string>{}) {
  clang::CompilerInstance compiler;
  clang::DiagnosticOptions diagnosticOptions;
  compiler.createDiagnostics();

  std::vector<std::string> args{{"-x", "cl", "-include", cl_header_path}};
  for (auto &arg : extra_args) args.push_back(arg);
  std::vector<const char *> argv;
  for (auto &arg : args) argv.push_back(arg.c_str());

  // Create an invocation that passes any flags to preprocessor
  auto Invocation = std::make_shared<clang::CompilerInvocation>();
  clang::CompilerInvocation::CreateFromArgs(
      *Invocation, &(*argv.begin()), &(*argv.end()), compiler.getDiagnostics());

  compiler.setInvocation(Invocation);

  // Set default target triple
  std::shared_ptr<clang::TargetOptions> pto =
      std::make_shared<clang::TargetOptions>();
  pto->Triple = llvm::sys::getDefaultTargetTriple();
  clang::TargetInfo *pti =
      clang::TargetInfo::CreateTargetInfo(compiler.getDiagnostics(), pto);
  compiler.setTarget(pti);

  compiler.createFileManager();
  compiler.createSourceManager(compiler.getFileManager());

  clang::LangOptions langOpts;
  langOpts.OpenCL = 1;

  // default arguments
  llvm::Triple *triple = new llvm::Triple(llvm::sys::getDefaultTargetTriple());
  clang::PreprocessorOptions pproc;

  Invocation->setLangDefaults(langOpts, clang::InputKind::OpenCL, *triple,
                              pproc);

  compiler.createPreprocessor(clang::TU_Complete);
  compiler.getPreprocessorOpts().UsePredefines = false;

  compiler.createASTContext();

  const clang::FileEntry *pFile = compiler.getFileManager().getFile(path);
  compiler.getSourceManager().setMainFileID(
      compiler.getSourceManager().createFileID(pFile, clang::SourceLocation(),
                                               clang::SrcMgr::C_User));
  compiler.getDiagnosticClient().BeginSourceFile(compiler.getLangOpts(),
                                                 &compiler.getPreprocessor());

  ASTConsumer astConsumer;

  // Parse the AST
  ParseAST(compiler.getPreprocessor(), &astConsumer, compiler.getASTContext());
  compiler.getDiagnosticClient().EndSourceFile();

  const std::string base_path = basename(path);
  astConsumer.dumpKernelFeatures(base_path, out);
}

bool file_exists(const std::string &path) {
  struct stat buffer;
  return (stat(path.c_str(), &buffer) == 0);
}

// Check if argument is of the form '-extra-arg=xxx', and if so,
// return 'xxx'. If not, return an empty string.
std::string get_compiler_arg(const std::string &arg) {
  std::string prefix("-extra-arg=");
  if (!arg.compare(0, prefix.size(), prefix))
    return arg.substr(prefix.size());
  else
    return {};
}

void usage(const std::string &progname, std::ostream &out = std::cout) {
  out << "Usage: " << progname << " [-header-only] [-extra-arg=<arg> ...] "
      << "<file> [files ...]\n\n"
      << "Extracts static features from OpenCL source files.";
}

int main(int argc, char **argv) {
  auto cl_header_path = std::string(argv[1]);
  const std::vector<std::string> args{argv + 2, argv + argc};
  std::vector<std::string> paths, compiler_args;

  for (const auto &arg : args) {
    std::string carg = get_compiler_arg(arg);

    if (carg.size())
      compiler_args.push_back(carg);
    else
      paths.push_back(arg);
  }

  if (!paths.size()) {
    usage(basename(argv[0]), std::cerr);
    return 1;
  }

  int ret = 0;
  for (const std::string &path : paths) {
    if (file_exists(path)) {
      extract_features(path, cl_header_path, std::cout, compiler_args);
    } else {
      std::cerr << "error: file not found: " << path << '\n';
      ret = 1;
    }
  }

  return ret;
}
