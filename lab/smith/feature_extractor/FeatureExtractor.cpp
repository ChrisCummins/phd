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
#include "FeatureExtractor.h"
#include<iostream>
#include<fstream>
#include<dirent.h>

using namespace clang;
using namespace std;



class ParameterInfo {
 public:
  enum VarType {GLOBAL, LOCAL, OTHER};
 private:
  string varName;
  VarType type;
 public:
  ParameterInfo(string varName, VarType type)
  {
    this->varName = varName;
    this->type = type;

  }

  string getVarName() { return varName; }
  VarType getType() { return type; }

};

class FuncInfo {
  string kernelName;
  vector<ParameterInfo> vp;
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

  FuncInfo(string kernelName) : isOCLKernel(false){
    FuncInfo();
    this->kernelName = kernelName;
  }

  void setAsOclKernel()
  {
    isOCLKernel = true;
  }

  bool isOclKernel()
  {
    return isOCLKernel;
  }

  void resetCounters()
  {
    global_mem_ls = 0;
    comp_inst_count = 0;
    rational_inst_count = 0;
    barrier_count = 0;
    cols_mem_access_count = 0;
    local_mem_ls_count = 0;
    atomic_op_count = 0;
  }

  void addCountersFromOtherFunc(FuncInfo *KI)
  {
    this->global_mem_ls += KI->getGlobalMemLSCount();
    this->comp_inst_count += KI->getCompInstCount();
    this->rational_inst_count += KI->getRationalInstCount();
    this->barrier_count += KI->getBarrierCount();
    this->cols_mem_access_count += KI->getColMemAccessCount();
    this->local_mem_ls_count += KI->getLocalMemLSCount();
    this->atomic_op_count += KI->getAtomicOpCount();
  }

  void setFuncName(string name)
  {
    this->kernelName = name;
  }

  string getFuncName()
  {
    return kernelName;
  }

  void addParameter(ParameterInfo p)
  {
    vp.push_back(p);
  }

  unsigned getParameterNum()
  {
    return vp.size();
  }

  ParameterInfo& getParameter(unsigned i)
  {
    return vp[i];
  }

  void incrLocalMemLSCount()
  {
    local_mem_ls_count++;
  }

  void incrAtomicOpCount()
  {
    atomic_op_count++;
  }

  void incrBarrierCount()
  {
    barrier_count++;
  }

  void addComputationCount(unsigned c)
  {
    comp_inst_count += c;
  }

  void incrGlobalMemLSCount()
  {
    global_mem_ls++;
  }

  void incrCompInstCount()
  {
    comp_inst_count++;
  }

  void incrRationalInstCount()
  {
    rational_inst_count++;
  }


  void incrColMemAccessCount()
  {
    cols_mem_access_count++;
  }

  unsigned getAtomicOpCount()
  {
    return atomic_op_count;
  }

  unsigned getBarrierCount()
  {
    return barrier_count;
  }

  unsigned getLocalMemLSCount()
  {
    return local_mem_ls_count;
  }

  unsigned getColMemAccessCount()
  {
    return cols_mem_access_count;
  }

  unsigned getCompInstCount()
  {
    return comp_inst_count;
  }

  unsigned getRationalInstCount()
  {
    return rational_inst_count;
  }

  unsigned getGlobalMemLSCount()
  {
    return global_mem_ls;
  }

  bool isGlobalVar(string var)
  {
    for (unsigned i=0; i<vp.size(); i++)
    {
      if (vp[i].getVarName() == var)
        return true;

    }

    return false;
  }

};


static string AtomicFuncs[]=
{
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
  "atomic_max",
  "END"
};


// RecursiveASTVisitor is is the big-kahuna visitor that traverses
// everything in the AST.
class MyRecursiveASTVisitor
    : public RecursiveASTVisitor<MyRecursiveASTVisitor>
{
  vector<FuncInfo*> FuncInfoVec;
  FuncInfo* pCurKI;

  bool processArrayIndices(ArraySubscriptExpr* Node);
  bool processArrayIdxBinaryOperator(BinaryOperator* bo);
  FuncInfo* findFunctionInfo(string funcName);
  void InitializeOCLRoutines();
  bool isBuiltInAtomicFunc(string f);
 public:
  MyRecursiveASTVisitor() { pCurKI = NULL;  InitializeOCLRoutines(); }
  bool VisitStmt(Stmt *s);
  bool VisitFunctionDecl(FunctionDecl *f);
  Expr *VisitBinaryOperator(BinaryOperator *op);
  bool VisitDeclRefExpr(DeclRefExpr *Node);
  bool VisitVarDecl(VarDecl *D);
  bool VisitCallExpr(CallExpr *E);

  //This is use to count coalesced memory accesses
  //This is the most complex function
  bool VisitArraySubscriptExpr(ArraySubscriptExpr *Node);
  vector<FuncInfo*>& getFuncInfo()
  {
    return FuncInfoVec;
  }
  ~MyRecursiveASTVisitor()
  {
    for (unsigned i=0; i<FuncInfoVec.size(); i++)
    {
      delete FuncInfoVec[i];
    }
  }
};


#define DEFAULT_OCL_ROUTINE_COUNT 15

#define ADD_OCL_ROUTINE_INFO(__p__,__name__) {                  \
    __p__=new FuncInfo(__name__);                               \
    __p__->addComputationCount(DEFAULT_OCL_ROUTINE_COUNT);      \
    FuncInfoVec.push_back(__p__);                               \
  }

void MyRecursiveASTVisitor::InitializeOCLRoutines()
{
  FuncInfo *p;
  ADD_OCL_ROUTINE_INFO(p, "exp");
  ADD_OCL_ROUTINE_INFO(p, "log");
  ADD_OCL_ROUTINE_INFO(p, "sqrt");
}

bool MyRecursiveASTVisitor::isBuiltInAtomicFunc(string f)
{
  int i = 0;
  while(AtomicFuncs[i] != "END")
  {
    if (AtomicFuncs[i] == f)
    {
      return true;
    }
    i++;
  }

  return false;
}

FuncInfo* MyRecursiveASTVisitor::findFunctionInfo(string funcName)
{
  for (unsigned i=0; i<FuncInfoVec.size(); i++)
  {
    if (FuncInfoVec[i]->getFuncName() == funcName)
      return FuncInfoVec[i];
  }

  return NULL;
}



bool MyRecursiveASTVisitor::VisitCallExpr(CallExpr *E)
{
  FunctionDecl *D = E->getDirectCallee();
  if (!D) return true;

  string varName = D->getNameInfo().getAsString();


  if (isBuiltInAtomicFunc(varName))
  {
    pCurKI->incrAtomicOpCount();
  }
  else
    if (varName == "barrier")
    {
      pCurKI->incrBarrierCount();
    }
    else
    {
      //Add up the counters if a function is called
      FuncInfo *pi = findFunctionInfo(varName);
      if (pi)
      {
        pCurKI->addCountersFromOtherFunc(pi);
      }

    }

  return true;
}


bool MyRecursiveASTVisitor::processArrayIdxBinaryOperator(BinaryOperator* bo)
{
  if (!(bo->isMultiplicativeOp() || bo->isAdditiveOp() ))
  {
    return false;
  }


  Expr* lhs = bo->getLHS();
  Expr* rhs = bo->getRHS();

  IntegerLiteral *lhsi = dyn_cast<IntegerLiteral>(lhs);
  IntegerLiteral *rhsi = dyn_cast<IntegerLiteral>(rhs);

  //If the index is a constant value, return true
  if (lhsi && rhsi)
  {
    return true;
  }

  if (lhsi)
  {
    string v = lhsi->getValue().toString(10, /*isSigned*/false);

    //If the step is 1, or zero colescated
    if (v == "1" || v == "0")
    {
      return true;
    }

  }

  if (rhsi)
  {
    string v = rhsi->getValue().toString(10, /*isSigned*/false);

    //If the step is 1, or zero colescated
    if (v == "1" || v == "0")
    {
      return true;
    }
  }

  return false;
}


//
//Return a boolean value to indicate if the array acess is colescated
bool MyRecursiveASTVisitor::processArrayIndices(ArraySubscriptExpr* Node)
{
  Expr* tExpr = Node->getRHS();

  //Check if this is a binaryoperator
  BinaryOperator *bo = dyn_cast<BinaryOperator>(tExpr);
  if (bo)
  {
    return processArrayIdxBinaryOperator(bo);
  }

  //If the array index is determined by a function call,
  //assume it is not colescate accessing
  CallExpr *ce = dyn_cast<CallExpr>(tExpr);
  if (ce) {
    cout << "False becasue the index is determined through a function call" << endl;
    return false;
  }

  ImplicitCastExpr *iCE = dyn_cast<ImplicitCastExpr>(tExpr);
  if (iCE)
  {
    DeclRefExpr *DRE = dyn_cast<DeclRefExpr>(iCE->getSubExpr());
    //The array is indexed using a variable
    if (DRE){
      //Assuming this is colescate memory acessing
      return true;
    }
  }

  IntegerLiteral *IntV = dyn_cast<IntegerLiteral>(tExpr);
  if (IntV) return true;

  ArraySubscriptExpr* aExpr = dyn_cast<ArraySubscriptExpr>(tExpr);
  if (aExpr)
  {
    return processArrayIndices(aExpr);
  }

  return true;


}


bool MyRecursiveASTVisitor::VisitArraySubscriptExpr(ArraySubscriptExpr *Node)
{
  if (!pCurKI) return true;


  ParameterInfo::VarType vT = ParameterInfo::OTHER;

  //Retrive array type
  Expr* tExpr = Node->getLHS();
  if (tExpr)
  {
    ImplicitCastExpr* iExpr = dyn_cast<ImplicitCastExpr>(tExpr);
    if (!iExpr) return true;

    DeclRefExpr *DRE = dyn_cast<DeclRefExpr>(iExpr->getSubExpr());
    if (DRE)
    {
      ValueDecl *D = DRE->getDecl();
      QualType T = D->getType();

      string varName = D->getIdentifier()->getName();
      string tStr = T.getAsString();

      //FIXME:Urgly tricks
      if (tStr.find("__global") != std::string::npos)
      {
        vT = ParameterInfo::GLOBAL;
      }
      else
        if (tStr.find("__local") != std::string::npos)
        {
          vT = ParameterInfo::LOCAL;
        }
    }

  }

  //Calculate coalesced global memory accesses
  if (vT == ParameterInfo::GLOBAL)
  {
    if (processArrayIndices(Node))
    {
      pCurKI->incrColMemAccessCount();
    }
  }
  else
    if (vT == ParameterInfo::LOCAL)
    {
      pCurKI->incrLocalMemLSCount();
    }

  return true;

}


/*
 * VISIT Declare Exprs to record load and store to global memory variables
 *
 */
bool MyRecursiveASTVisitor::VisitDeclRefExpr(DeclRefExpr *Node)
{
  if (pCurKI)
  {
    string varName =  Node->getNameInfo().getAsString();
    if (pCurKI->isGlobalVar(varName))
    {
      pCurKI->incrGlobalMemLSCount();
    }
  }
  return true;

}



// Override Binary Operator expressions
Expr *MyRecursiveASTVisitor::VisitBinaryOperator(BinaryOperator *E)
{
  if (pCurKI)
  {

    if (E->isComparisonOp())
    {
      pCurKI->incrRationalInstCount();
    }
    else
      if (E->isMultiplicativeOp() ||
          E->isAdditiveOp() ||
          E->isShiftOp() ||
          E->isBitwiseOp() ||
          E->isShiftAssignOp()
          )
      {
        pCurKI->incrCompInstCount();
      }
  }


  return E;
}

bool MyRecursiveASTVisitor::VisitVarDecl(VarDecl *D)
{
  QualType T = D->getTypeSourceInfo()
               ? D->getTypeSourceInfo()->getType()
               : D->getASTContext().getUnqualifiedObjCPointerType(D->getType());


  string varName = D->getIdentifier()->getName();
  string tStr = T.getAsString();

  if (tStr.find("__local") != std::string::npos)
  {
    if (pCurKI)
    {
      ParameterInfo pi(varName, ParameterInfo::LOCAL);
      pCurKI->addParameter(pi);
    }

  }

  return true;
}

// Override Statements which includes expressions and more
bool MyRecursiveASTVisitor::VisitStmt(Stmt *s)
{
  return true; // returning false aborts the traversal
}

bool MyRecursiveASTVisitor::VisitFunctionDecl(FunctionDecl *f)
{

  string Proto = f->getNameInfo().getAsString();
  pCurKI = new FuncInfo(Proto);
  pCurKI->resetCounters();

  unsigned up = f->getNumParams();
  for (unsigned i=0; i<up; i++)
  {
    ParmVarDecl* pD = f->getParamDecl(i);
    QualType T = pD->getTypeSourceInfo()
                 ? pD->getTypeSourceInfo()->getType()
                 : pD->getASTContext().getUnqualifiedObjCPointerType(pD->getType());

    string varName = pD->getIdentifier()->getName();
    string tStr = T.getAsString();


    ParameterInfo::VarType vT = ParameterInfo::OTHER;
    //FIXME:Urgly tricks
    if (tStr.find("__global") != std::string::npos)
    {
      pCurKI->setAsOclKernel();
      vT = ParameterInfo::GLOBAL;
    }
    else
      if (tStr.find("__local") != std::string::npos)
      {
        vT = ParameterInfo::LOCAL;
      }

    if (vT != ParameterInfo::OTHER)
    {
      ParameterInfo pi(varName, vT);
      pCurKI->addParameter(pi);
    }
  }

  if (f->hasBody())
  {
    FuncInfoVec.push_back(pCurKI);
  }

  return true; // returning false aborts the traversal
}

class MyASTConsumer : public ASTConsumer
{
 public:

  MyASTConsumer() : rv() { }
  virtual bool HandleTopLevelDecl(DeclGroupRef d);
  MyRecursiveASTVisitor rv;
  void dumpKernelFeatures(string fileName, ofstream& fout);
};


void MyASTConsumer::dumpKernelFeatures(string fileName, ofstream& fout)
{
  vector<FuncInfo*>& FuncInfoVec = rv.getFuncInfo();

  for (unsigned i=0; i<FuncInfoVec.size(); i++)
  {
    if (FuncInfoVec[i]->isOclKernel())
    {
      fout << fileName << "," << FuncInfoVec[i]->getFuncName() << "," << FuncInfoVec[i]->getCompInstCount() << "," << FuncInfoVec[i]->getRationalInstCount() << "," << FuncInfoVec[i]->getGlobalMemLSCount() << "," << FuncInfoVec[i]->getLocalMemLSCount() << "," << FuncInfoVec[i]->getColMemAccessCount() << "," << FuncInfoVec[i]->getAtomicOpCount() << ",,\n";
    }
  }

}

bool MyASTConsumer::HandleTopLevelDecl(DeclGroupRef d)
{
  typedef DeclGroupRef::iterator iter;

  for (iter b = d.begin(), e = d.end(); b != e; ++b)
  {
    rv.TraverseDecl(*b);
  }

  return true; // keep going
}

string retriveFileName(string fname)
{
  string res="";
  for (int i=fname.length()-1; i>=0; i--)
  {
    if (fname[i] == '/') break;

    res = fname[i] + res;
  }

  return res;
}



int worker(string fileName, ofstream& fout, int argc, char **argv)
{
  CompilerInstance compiler;
  DiagnosticOptions diagnosticOptions;
  compiler.createDiagnostics();


  // Create an invocation that passes any flags to preprocessor
  CompilerInvocation *Invocation = new CompilerInvocation;
  CompilerInvocation::CreateFromArgs(*Invocation, argv, argv + argc,
                                     compiler.getDiagnostics());


  compiler.setInvocation(Invocation);

  // Set default target triple
  std::shared_ptr<clang::TargetOptions> pto = std::make_shared<clang::TargetOptions>();
  pto->Triple = llvm::sys::getDefaultTargetTriple();
  TargetInfo *pti = TargetInfo::CreateTargetInfo(compiler.getDiagnostics(), pto);
  compiler.setTarget(pti);

  compiler.createFileManager();
  compiler.createSourceManager(compiler.getFileManager());

  HeaderSearchOptions &headerSearchOptions = compiler.getHeaderSearchOpts();

  // <Warning!!> -- Platform Specific Code lives here
  // This depends on A) that you're running linux and
  // B) that you have the same GCC LIBs installed that
  // I do.
  // Search through Clang itself for something like this,
  // go on, you won't find it. The reason why is Clang
  // has its own versions of std* which are installed under
  // /usr/local/lib/clang/<version>/include/
  // See somewhere around Driver.cpp:77 to see Clang adding
  // its version of the headers to its include path.
  // To see what include paths need to be here, try
  // clang -v -c test.c
  // or clang++ for C++ paths as used below:
  headerSearchOptions.AddPath("/usr/include/c++/4.6",
                              clang::frontend::Angled,
                              false,
                              false);
  headerSearchOptions.AddPath("/usr/include/c++/4.6/i686-linux-gnu",
                              clang::frontend::Angled,
                              false,
                              false);
  headerSearchOptions.AddPath("/usr/include/c++/4.6/backward",
                              clang::frontend::Angled,
                              false,
                              false);
  headerSearchOptions.AddPath("/usr/local/include",
                              clang::frontend::Angled,
                              false,
                              false);
  headerSearchOptions.AddPath("/usr/local/lib/clang/3.3/include",
                              clang::frontend::Angled,
                              false,
                              false);
  headerSearchOptions.AddPath("/usr/include/i386-linux-gnu",
                              clang::frontend::Angled,
                              false,
                              false);
  headerSearchOptions.AddPath("/usr/include",
                              clang::frontend::Angled,
                              false,
                              false);
  // </Warning!!> -- End of Platform Specific Code


  LangOptions langOpts;
  langOpts.OpenCL = 1;


  Invocation->setLangDefaults(langOpts,
                              clang::IK_OpenCL);

  compiler.createPreprocessor(clang::TU_Complete);
  compiler.getPreprocessorOpts().UsePredefines = false;

  compiler.createASTContext();

  const FileEntry *pFile = compiler.getFileManager().getFile(fileName);
  compiler.getSourceManager().setMainFileID( compiler.getSourceManager().createFileID( pFile, clang::SourceLocation(), clang::SrcMgr::C_User));
  compiler.getDiagnosticClient().BeginSourceFile(compiler.getLangOpts(),
                                                 &compiler.getPreprocessor());

  MyASTConsumer astConsumer;

  //Parse the AST
  ParseAST(compiler.getPreprocessor(), &astConsumer, compiler.getASTContext());
  compiler.getDiagnosticClient().EndSourceFile();

  string pFName = retriveFileName(fileName);
  astConsumer.dumpKernelFeatures(pFName, fout);

  return 0;
}

string addOCLFuncs(string fileName)
{
  ifstream fin(fileName.c_str());
  string dest="tmp/";
  dest = dest + retriveFileName(fileName);

  ofstream fout(dest.c_str());
  fout << "#include \"cl_platform.h\" \n";

  string line;
  while (getline(fin, line))
  {
    fout << line << "\n";
  }

  fin.close();
  fout.close();

  return dest;
}

vector<string> listFiles( const char* path )
{
  vector<string> fls;
  DIR* dirFile = opendir( path );
  if ( dirFile )
  {
    struct dirent* hFile;
    errno = 0;
    while (( hFile = readdir( dirFile )) != NULL )
    {
      if ( !strcmp( hFile->d_name, "."  )) continue;
      if ( !strcmp( hFile->d_name, ".." )) continue;

      // in linux hidden files all start with '.'
      if (  hFile->d_name[0] == '.' ) continue;

      // dirFile.name is the name of the file. Do whatever string comparison
      // you want here. Something like:
      if ( strstr( hFile->d_name, ".cl" ))
      {
        string dest = path;
        dest = dest + "/";
        dest = dest + hFile->d_name;
        fls.push_back(dest);
      }
    }
    closedir( dirFile );
  }
  else
  {
    cerr << "Failed to open " << path << endl;
    exit(-1);
  }
  return fls;
}


int main(int argc, char** argv)
{
  if (argc < 2) {

    cerr << "Please specific the location (folder) that contains the OpenCL kernel" << endl;
    exit (-1);
  }

  strcpy(argv[0], "-I.");
  vector<string> fnl = listFiles(argv[argc - 1]);
  if (fnl.size() <= 0)
  {
    cerr << "Couldn't find any opencl files" << endl;
    exit(-1);
  }

  system("mkdir -p tmp");
  ofstream fout("features.csv");
  fout << "File,Kernel Name,#compute operations,#rational operations,#accesses to global memory, #accesses to local memory, #coalesced memory accesses, #atomic op, amount of data transfers, #work-items\n";
  for (unsigned i=0; i<fnl.size(); i++)
  {
    string dest = addOCLFuncs(fnl[i]);
    char *p = new char[dest.length() + 1];
    strcpy(p, dest.c_str());
    argv[argc-1] = p;
    worker(dest, fout, argc, argv);
    delete[] p;
  }

  fout.close();
  return 0;
}
