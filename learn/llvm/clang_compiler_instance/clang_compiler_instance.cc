// A simple example of using the clang API.
// Usage: $ bazel run //learn/llvm/clang_compiler_instance -- <path_of_file>
#include <iostream>
#include <memory>

#include "clang/Basic/FileManager.h"
#include "clang/Basic/SourceManager.h"
#include "clang/Basic/TargetInfo.h"
#include "clang/Basic/TargetOptions.h"
#include "clang/Frontend/CompilerInstance.h"
#include "clang/Lex/Preprocessor.h"
#include "clang/Lex/Token.h"

#include "labm8/cpp/app.h"
#include "labm8/cpp/logging.h"
#include "labm8/cpp/string.h"

DEFINE_string(input_path, "", "Path of the file to lex.");

namespace {

void LexAndPrintTokens(const string& input_path) {
  LOG(INFO) << "lexing " << input_path;

  // CompilerInstance will hold the instance of the Clang compiler for us,
  // managing the various objects needed to run the compiler.
  clang::CompilerInstance compiler_instance;

  // Diagnostics manage problems and issues in compile
  compiler_instance.createDiagnostics(
      /*Client=*/nullptr, /*ShouldOwnClient=*/false);

  // Set target platform options
  // Initialize target info with the default triple for our platform.
  auto target_options = std::make_shared<clang::TargetOptions>();
  target_options->Triple = llvm::sys::getDefaultTargetTriple();
  auto target_info = clang::TargetInfo::CreateTargetInfo(
      compiler_instance.getDiagnostics(), target_options);
  compiler_instance.setTarget(target_info);

  // FileManager supports for file system lookup, file system caching, and
  // directory search management.
  compiler_instance.createFileManager();
  clang::FileManager& file_manager = compiler_instance.getFileManager();

  // SourceManager handles loading and caching of source files into memory.
  compiler_instance.createSourceManager(file_manager);
  clang::SourceManager& source_manager = compiler_instance.getSourceManager();

  // Preprocessor runs within a single source file
  compiler_instance.createPreprocessor(clang::TU_Complete);

  // ASTContext holds long‐lived AST nodes (such as types and decls) .
  compiler_instance.createASTContext();

  const clang::FileEntry* pFile =
      compiler_instance.getFileManager().getFile(input_path);
  if (!pFile) {
    LOG(FATAL) << "File not found: " << input_path;
  }
  const clang::FileID file_id = source_manager.createFileID(
      pFile, clang::SourceLocation(), clang::SrcMgr::C_User);
  source_manager.setMainFileID(file_id);

  compiler_instance.getPreprocessor().EnterMainSourceFile();
  compiler_instance.getDiagnosticClient().BeginSourceFile(
      compiler_instance.getLangOpts(), &compiler_instance.getPreprocessor());

  // Lex source code and print tokens.
  clang::Token tok;
  do {
    compiler_instance.getPreprocessor().Lex(tok);
    if (compiler_instance.getDiagnostics().hasErrorOccurred()) {
      break;
    }
    compiler_instance.getPreprocessor().DumpToken(tok);
    std::cerr << std::endl;
  } while (tok.isNot(clang::tok::eof));
  compiler_instance.getDiagnosticClient().EndSourceFile();
}

}  // namespace

int main(int argc, char** argv) {
  labm8::InitApp(&argc, &argv);

  if (FLAGS_input_path.empty()) {
    LOG(FATAL) << "--input_path not specified";
  }

  ::LexAndPrintTokens(FLAGS_input_path);
  LOG(INFO) << "done";

  return 0;
}
