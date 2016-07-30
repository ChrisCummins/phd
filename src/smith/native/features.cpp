/*
 * FeatureExtractor.cpp
 *
 * An implementation of the feature extractor used in:
 *
 *     Grewe, D., Wang, Z., & O’Boyle, M. F. P. M. (2013). Portable
 *     mapping of data parallel programs to OpenCL for heterogeneous
 *     systems. In CGO. IEEE.
 *
 * Written by Zheng Wang <zh.wang@ed.ac.uk>.
 * Modified by Chris Cummins <chrisc.101@gmail.com>
 */
#include <string>
#include <array>
#include <fstream>
#include <iostream>
#include <sstream>
#include <stdio.h>
#include <system_error>
#include <vector>

#include <dirent.h>
#include <sys/stat.h>
#include <sys/types.h>

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Weverything"
#include <clang/AST/ASTConsumer.h>
#include <clang/AST/ASTContext.h>
#include <clang/AST/ParentMap.h>
#include <clang/AST/RecursiveASTVisitor.h>
#include <clang/AST/Stmt.h>
#include <clang/Basic/Diagnostic.h>
#include <clang/Basic/DiagnosticOptions.h>
#include <clang/Basic/FileManager.h>
#include <clang/Basic/SourceManager.h>
#include <clang/Basic/TargetInfo.h>
#include <clang/Basic/TargetOptions.h>
#include <clang/Frontend/CompilerInstance.h>
#include <clang/Frontend/TextDiagnosticPrinter.h>
#include <clang/Lex/Lexer.h>
#include <clang/Lex/Preprocessor.h>
#include <clang/Parse/ParseAST.h>
#include <clang/Rewrite/Core/Rewriter.h>
#include <clang/Rewrite/Frontend/Rewriters.h>
#pragma GCC diagnostic pop


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
  FuncInfo() {
    resetCounters();
    isOCLKernel = false;
  }

  explicit FuncInfo(std::string _kernelName) : isOCLKernel(false) {
    FuncInfo();
    kernelName = _kernelName;
  }

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
      if (vp[i].getVarName() == var)
        return true;
    }

    return false;
  }
};

static const std::array<std::string, 11> AtomicFuncs{{
    "atomic_add",
        "atomic_sub",
        "atomic_dec",
        "atomic_and",
        "atomic_cmpxchg",
        "atomic_or",
        "atomic_xchg",
        "atomic_min",
        "atomic_xor",
        "atomic_inc",
        "atomic_max"
        }};

// RecursiveASTVisitor is is the big-kahuna visitor that traverses
// everything in the AST.
class MyRecursiveASTVisitor
    : public clang::RecursiveASTVisitor<MyRecursiveASTVisitor> {
  std::vector<FuncInfo *> FuncInfoVec;
  FuncInfo *pCurKI;

  bool processArrayIndices(clang::ArraySubscriptExpr *Node);
  bool processArrayIdxBinaryOperator(clang::BinaryOperator *bo);
  FuncInfo *findFunctionInfo(std::string funcName);
  void InitializeOCLRoutines();
  bool isBuiltInAtomicFunc(std::string f);

 public:
  MyRecursiveASTVisitor() {
    pCurKI = NULL;
    InitializeOCLRoutines();
  }
  bool VisitStmt(clang::Stmt *s);
  bool VisitFunctionDecl(clang::FunctionDecl *f);
  clang::Expr *VisitBinaryOperator(clang::BinaryOperator *op);
  bool VisitDeclRefExpr(clang::DeclRefExpr *Node);
  bool VisitVarDecl(clang::VarDecl *D);
  bool VisitCallExpr(clang::CallExpr *E);

  // This is use to count coalesced memory accesses
  // This is the most complex function
  bool VisitArraySubscriptExpr(clang::ArraySubscriptExpr *Node);
  std::vector<FuncInfo *> &getFuncInfo() { return FuncInfoVec; }
  ~MyRecursiveASTVisitor() {
    for (unsigned i = 0; i < FuncInfoVec.size(); i++) {
      delete FuncInfoVec[i];
    }
  }
};

#define DEFAULT_OCL_ROUTINE_COUNT 15

#define ADD_OCL_ROUTINE_INFO(__p__, __name__)                   \
  {                                                             \
    __p__ = new FuncInfo(__name__);                             \
    __p__->addComputationCount(DEFAULT_OCL_ROUTINE_COUNT);      \
    FuncInfoVec.push_back(__p__);                               \
  }

void MyRecursiveASTVisitor::InitializeOCLRoutines() {
  FuncInfo *p;
  ADD_OCL_ROUTINE_INFO(p, "exp");
  ADD_OCL_ROUTINE_INFO(p, "log");
  ADD_OCL_ROUTINE_INFO(p, "sqrt");
}

bool MyRecursiveASTVisitor::isBuiltInAtomicFunc(std::string f) {
  for (const std::string& builtin : AtomicFuncs)
    if (f == builtin)
      return true;

  return false;
}

FuncInfo *MyRecursiveASTVisitor::findFunctionInfo(std::string funcName) {
  for (unsigned i = 0; i < FuncInfoVec.size(); i++) {
    if (FuncInfoVec[i]->getFuncName() == funcName)
      return FuncInfoVec[i];
  }

  return NULL;
}

bool MyRecursiveASTVisitor::VisitCallExpr(clang::CallExpr *E) {
  clang::FunctionDecl *D = E->getDirectCallee();
  if (!D)
    return true;

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

bool MyRecursiveASTVisitor::processArrayIdxBinaryOperator(
    clang::BinaryOperator *bo) {
  if (!(bo->isMultiplicativeOp() || bo->isAdditiveOp())) {
    return false;
  }

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

//
// Return a boolean value to indicate if the array acess is colescated
bool MyRecursiveASTVisitor::processArrayIndices(
    clang::ArraySubscriptExpr *Node) {
  clang::Expr *tExpr = Node->getRHS();

  // Check if this is a binaryoperator
  clang::BinaryOperator *bo = clang::dyn_cast<clang::BinaryOperator>(tExpr);
  if (bo) {
    return processArrayIdxBinaryOperator(bo);
  }

  // If the array index is determined by a function call,
  // assume it is not colescate accessing
  clang::CallExpr *ce = clang::dyn_cast<clang::CallExpr>(tExpr);
  if (ce) {
    std::cerr << "False becasue the index is determined through a "
              << "function call"
              << std::endl;
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
  if (IntV)
    return true;

  clang::ArraySubscriptExpr *aExpr =
      clang::dyn_cast<clang::ArraySubscriptExpr>(tExpr);
  if (aExpr) {
    return processArrayIndices(aExpr);
  }

  return true;
}

bool MyRecursiveASTVisitor::VisitArraySubscriptExpr(
    clang::ArraySubscriptExpr *Node) {
  if (!pCurKI)
    return true;

  ParameterInfo::VarType vT = ParameterInfo::OTHER;

  // Retrive array type
  clang::Expr *tExpr = Node->getLHS();
  if (tExpr) {
    clang::ImplicitCastExpr *iExpr =
        clang::dyn_cast<clang::ImplicitCastExpr>(tExpr);
    if (!iExpr)
      return true;

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

/*
 * VISIT Declare Exprs to record load and store to global memory variables
 *
 */
bool MyRecursiveASTVisitor::VisitDeclRefExpr(clang::DeclRefExpr *Node) {
  if (pCurKI) {
    std::string varName = Node->getNameInfo().getAsString();
    if (pCurKI->isGlobalVar(varName)) {
      pCurKI->incrGlobalMemLSCount();
    }
  }
  return true;
}

// Override Binary Operator expressions
clang::Expr *MyRecursiveASTVisitor::VisitBinaryOperator(
    clang::BinaryOperator *E) {
  if (pCurKI) {
    if (E->isComparisonOp()) {
      pCurKI->incrRationalInstCount();
    } else if (E->isMultiplicativeOp() || E->isAdditiveOp() || E->isShiftOp() ||
               E->isBitwiseOp() || E->isShiftAssignOp()) {
      pCurKI->incrCompInstCount();
    }
  }

  return E;
}

bool MyRecursiveASTVisitor::VisitVarDecl(clang::VarDecl *D) {
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

// Override Statements which includes expressions and more
bool MyRecursiveASTVisitor::VisitStmt(clang::Stmt *s) {
  return true;  // returning false aborts the traversal
}

bool MyRecursiveASTVisitor::VisitFunctionDecl(clang::FunctionDecl *f) {
  std::string Proto = f->getNameInfo().getAsString();
  pCurKI = new FuncInfo(Proto);
  pCurKI->resetCounters();

  unsigned up = f->getNumParams();
  for (unsigned i = 0; i < up; i++) {
    clang::ParmVarDecl *pD = f->getParamDecl(i);
    clang::QualType T =
        pD->getTypeSourceInfo()
        ? pD->getTypeSourceInfo()->getType()
        : pD->getASTContext().getUnqualifiedObjCPointerType(pD->getType());

    std::string varName = pD->getIdentifier()->getName();
    std::string tStr = T.getAsString();

    ParameterInfo::VarType vT = ParameterInfo::OTHER;
    // FIXME:Urgly tricks
    if (tStr.find("__global") != std::string::npos) {
      pCurKI->setAsOclKernel();
      vT = ParameterInfo::GLOBAL;
    } else if (tStr.find("__local") != std::string::npos) {
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

  return true;  // returning false aborts the traversal
}

class MyASTConsumer : public clang::ASTConsumer {
 public:
  MyASTConsumer() : rv() {}

  virtual bool HandleTopLevelDecl(clang::DeclGroupRef d) {
    typedef clang::DeclGroupRef::iterator iter;

    for (iter b = d.begin(), e = d.end(); b != e; ++b) {
      rv.TraverseDecl(*b);
    }

    return true;  // keep going
  }

  MyRecursiveASTVisitor rv;

  void dumpKernelFeatures(std::string fileName, std::ostream& fout) {
    std::vector<FuncInfo *> &FuncInfoVec = rv.getFuncInfo();

    for (unsigned i = 0; i < FuncInfoVec.size(); i++) {
      if (FuncInfoVec[i]->isOclKernel()) {
        fout << fileName << "," << FuncInfoVec[i]->getFuncName() << ","
             << FuncInfoVec[i]->getCompInstCount() << ","
             << FuncInfoVec[i]->getRationalInstCount() << ","
             << FuncInfoVec[i]->getGlobalMemLSCount() << ","
             << FuncInfoVec[i]->getLocalMemLSCount() << ","
             << FuncInfoVec[i]->getColMemAccessCount() << ","
             << FuncInfoVec[i]->getAtomicOpCount() << "\n";
      }
    }
  }
};

std::string retriveFileName(std::string fname) {
  std::string res = "";
  for (int i = static_cast<int>(fname.length()) - 1; i >= 0; i--) {
    size_t idx = static_cast<size_t>(i);

    if (fname[idx] == '/')
      break;

    res = fname[idx] + res;
  }

  return res;
}

int worker(std::string fileName, std::ostream &fout, int argc, char **argv) {
  clang::CompilerInstance compiler;
  clang::DiagnosticOptions diagnosticOptions;
  compiler.createDiagnostics();

  // Create an invocation that passes any flags to preprocessor
  clang::CompilerInvocation *Invocation = new clang::CompilerInvocation;
  clang::CompilerInvocation::CreateFromArgs(*Invocation, argv, argv + argc,
                                            compiler.getDiagnostics());

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

  Invocation->setLangDefaults(langOpts, clang::IK_OpenCL);

  compiler.createPreprocessor(clang::TU_Complete);
  compiler.getPreprocessorOpts().UsePredefines = false;

  compiler.createASTContext();

  const clang::FileEntry *pFile = compiler.getFileManager().getFile(fileName);
  compiler.getSourceManager().setMainFileID(
      compiler.getSourceManager().createFileID(pFile, clang::SourceLocation(),
                                               clang::SrcMgr::C_User));
  compiler.getDiagnosticClient().BeginSourceFile(compiler.getLangOpts(),
                                                 &compiler.getPreprocessor());

  MyASTConsumer astConsumer;

  // Parse the AST
  ParseAST(compiler.getPreprocessor(), &astConsumer, compiler.getASTContext());
  compiler.getDiagnosticClient().EndSourceFile();

  std::string pFName = retriveFileName(fileName);
  astConsumer.dumpKernelFeatures(pFName, std::cout);

  return 0;
}

std::string addOCLFuncs(std::string fileName) {
  std::ifstream fin(fileName.c_str());
  std::string dest = "tmp/";
  dest = dest + retriveFileName(fileName);

  std::ofstream fout(dest.c_str());
  fout << "#include <cl_platform.h>\n";

  std::string line;
  while (getline(fin, line)) {
    fout << line << "\n";
  }

  fin.close();
  fout.close();

  return dest;
}

std::vector<std::string> listFiles(const char *path) {
  std::vector<std::string> fls;
  DIR *dirFile = opendir(path);
  if (dirFile) {
    struct dirent *hFile;
    errno = 0;
    while ((hFile = readdir(dirFile)) != NULL) {
      if (!strcmp(hFile->d_name, "."))
        continue;
      if (!strcmp(hFile->d_name, ".."))
        continue;

      // in linux hidden files all start with '.'
      if (hFile->d_name[0] == '.')
        continue;

      // dirFile.name is the name of the file. Do whatever string comparison
      // you want here. Something like:
      if (strstr(hFile->d_name, ".cl")) {
        std::string dest = path;
        dest = dest + "/";
        dest = dest + hFile->d_name;
        fls.push_back(dest);
      }
    }
    closedir(dirFile);
  } else {
    std::cerr << "Failed to open " << path << std::endl;
    exit(-1);
  }
  return fls;
}

int main(int argc, char **argv) {
  if (argc < 2) {
    std::cerr << "Please specific the location (folder) "
              << "that contains the OpenCL kernels"
              << std::endl;
    exit(-1);
  }

  snprintf(argv[0], sizeof(argv[0]), "-I.");
  std::vector<std::string> fnl = listFiles(argv[argc - 1]);
  if (fnl.size() <= 0) {
    std::cerr << "Couldn't find any opencl files" << std::endl;
    exit(-1);
  }

  system("mkdir -p tmp");
  std::cout << "file,"
            << "kernel,"
            << "comp,"
            << "rational,"
            << "mem,"
            << "localmem,"
            << "coalesced,"
            << "atomic\n";
  for (unsigned i = 0; i < fnl.size(); i++) {
    std::string dest = addOCLFuncs(fnl[i]);
    char *p = new char[dest.length() + 1];
    strncpy(p, dest.c_str(), dest.length() + 1);
    argv[argc - 1] = p;
    worker(dest, std::cout, argc, argv);
    delete[] p;
  }

  return 0;
}
