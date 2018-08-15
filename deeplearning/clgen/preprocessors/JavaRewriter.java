// Work in progress on Java AST rewriter.
package deeplearning.clgen.preprocessors;

import java.util.Properties;
import org.eclipse.jdt.core.JavaCore;
import org.eclipse.jdt.core.ToolFactory;
import org.eclipse.jdt.core.dom.AST;
import org.eclipse.jdt.core.dom.ASTParser;
import org.eclipse.jdt.core.dom.CompilationUnit;
import org.eclipse.jdt.core.dom.SimpleName;
import org.eclipse.jdt.core.dom.TypeDeclaration;
import org.eclipse.jdt.core.dom.rewrite.ASTRewrite;
import org.eclipse.jdt.core.formatter.CodeFormatter;
import org.eclipse.jdt.internal.compiler.impl.CompilerOptions;
import org.eclipse.jface.text.BadLocationException;
import org.eclipse.jface.text.Document;
import org.eclipse.jface.text.IDocument;
import org.eclipse.text.edits.TextEdit;

public class JavaRewriter {

  private static void FormatterDemo() {
    System.out.println("=> Formatter Demo\n");
    String result;

    String javaCode = "public class MyClass{ "
        + "public static void main(String[] args) { "
        + "System.out.println(\"Hello World\");"
        + " }"
        + " }";
    System.out.println("Unformatted source code:\n\n" + javaCode + "\n");

    Properties prefs = new Properties();
    prefs.setProperty(JavaCore.COMPILER_SOURCE, CompilerOptions.VERSION_1_8);
    prefs.setProperty(JavaCore.COMPILER_COMPLIANCE, CompilerOptions.VERSION_1_8);
    prefs.setProperty(JavaCore.COMPILER_CODEGEN_TARGET_PLATFORM, CompilerOptions.VERSION_1_8);

    CodeFormatter codeFormatter = ToolFactory.createCodeFormatter(prefs);
    IDocument doc = new Document(javaCode);
    try {
      TextEdit edit = codeFormatter
          .format(CodeFormatter.K_COMPILATION_UNIT | CodeFormatter.F_INCLUDE_COMMENTS,
              javaCode, 0, javaCode.length(), 0, null);
      if (edit != null) {
        edit.apply(doc);
        result = doc.get();
      } else {
        result = javaCode;
      }
    } catch (BadLocationException e) {
      throw new RuntimeException(e);
    }

    System.out.println("Formatted source code:\n\n" + result + "\n");
  }

  private static void RewriterDemo() {
    System.out.println("=> Rewriter Demo\n");

    // creation of a Document
    Document document = new Document("import java.util.List;\nclass X {}");
    ASTParser parser = ASTParser.newParser(AST.JLS10);
    parser.setSource(document.get().toCharArray());
    CompilationUnit cu = (CompilationUnit) parser.createAST(null);

    System.out.println("Input source code:\n\n" + document.get() + "\n");
    // One more time with feeling.
    parser.setSource(document.get().toCharArray());

    // creation of DOM/AST from a ICompilationUnit
    CompilationUnit astRoot = (CompilationUnit) parser.createAST(null);

    // creation of ASTRewrite
    ASTRewrite rewrite = ASTRewrite.create(astRoot.getAST());

    // description of the change
    SimpleName oldName = ((TypeDeclaration) astRoot.types().get(0)).getName();
    SimpleName newName = astRoot.getAST().newSimpleName("Y");
    rewrite.replace(oldName, newName, null);

    // computation of the text edits
    TextEdit edits = rewrite.rewriteAST(document, null);

    // computation of the new source code
    try {
      edits.apply(document);
      System.out.println("Output source code:\n\n" + document.get() + "\n");
    } catch (BadLocationException e) {
      System.out.println("exception");
    }
  }

  public static void main(final String[] args) {
    FormatterDemo();
    RewriterDemo();
  }
}
