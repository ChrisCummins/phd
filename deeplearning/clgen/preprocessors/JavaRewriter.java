// Work in progress on Java AST rewriter.
package deeplearning.clgen.preprocessors;

import java.io.IOException;
import java.nio.charset.Charset;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.List;
import java.util.ListIterator;
import java.util.Properties;
import org.eclipse.jdt.core.JavaCore;
import org.eclipse.jdt.core.ToolFactory;
import org.eclipse.jdt.core.dom.AST;
import org.eclipse.jdt.core.dom.ASTParser;
import org.eclipse.jdt.core.dom.ASTVisitor;
import org.eclipse.jdt.core.dom.Assignment;
import org.eclipse.jdt.core.dom.Comment;
import org.eclipse.jdt.core.dom.CompilationUnit;
import org.eclipse.jdt.core.dom.ExpressionStatement;
import org.eclipse.jdt.core.dom.FieldDeclaration;
import org.eclipse.jdt.core.dom.ImportDeclaration;
import org.eclipse.jdt.core.dom.Initializer;
import org.eclipse.jdt.core.dom.MemberRef;
import org.eclipse.jdt.core.dom.MethodDeclaration;
import org.eclipse.jdt.core.dom.MethodInvocation;
import org.eclipse.jdt.core.dom.MethodRef;
import org.eclipse.jdt.core.dom.NameQualifiedType;
import org.eclipse.jdt.core.dom.PackageDeclaration;
import org.eclipse.jdt.core.dom.QualifiedName;
import org.eclipse.jdt.core.dom.QualifiedType;
import org.eclipse.jdt.core.dom.SimpleName;
import org.eclipse.jdt.core.dom.SimpleType;
import org.eclipse.jdt.core.dom.TypeDeclaration;
import org.eclipse.jdt.core.dom.rewrite.ASTRewrite;
import org.eclipse.jdt.core.formatter.CodeFormatter;
import org.eclipse.jdt.internal.compiler.impl.CompilerOptions;
import org.eclipse.jface.text.BadLocationException;
import org.eclipse.jface.text.Document;
import org.eclipse.text.edits.DeleteEdit;
import org.eclipse.text.edits.TextEdit;

/**
 * A rewriter for Java source code.
 */
public class JavaRewriter {

  /**
   * Strip the comments from a compilation unit.
   *
   * This iterates *backwards* over the comments, creating a DeleteEdit instance
   * for each comment. These edits must be applied in the order they are
   * returned, since we use simple indexing into the string source to determine
   * where to snip the code.
   *
   * @param edits The edit list to append comment-removing edits to.
   * @param compilationUnit The compilation unit to remove comments from.
   */
  private static void StripComments(ArrayList<TextEdit> edits,
      final CompilationUnit compilationUnit) {
    final List comments = compilationUnit.getCommentList();
    ListIterator li = comments.listIterator(comments.size());
    while (li.hasPrevious()) {
      Object object = li.previous();
      if (object instanceof Comment) {
        final int start = ((Comment) object).getStartPosition();
        final int length = ((Comment) object).getLength();
        edits.add(new DeleteEdit(start, length));
      }
    }
  }

  /**
   * Format a Java source code.
   *
   * @param source The source code to format.
   * @return The formatted source code.
   */
  private static String FormatSource(final String source) {
    Document document = new Document(source);
    ArrayList<TextEdit> edits = new ArrayList<>();
    Properties codeFormatterProperties = new Properties();
    codeFormatterProperties.setProperty(
        JavaCore.COMPILER_SOURCE, CompilerOptions.VERSION_1_8);
    codeFormatterProperties.setProperty(
        JavaCore.COMPILER_COMPLIANCE, CompilerOptions.VERSION_1_8);
    codeFormatterProperties.setProperty(
        JavaCore.COMPILER_CODEGEN_TARGET_PLATFORM, CompilerOptions.VERSION_1_8);
    // Allow really, really long lines.
    codeFormatterProperties.setProperty(
        "org.eclipse.jdt.core.formatter.lineSplit", "9999");
    codeFormatterProperties.setProperty(
        "org.eclipse.jdt.core.formatter.comment.line_length", "9999");
    CodeFormatter codeFormatter = ToolFactory.createCodeFormatter(
        codeFormatterProperties);
    edits.add(codeFormatter.format(
        CodeFormatter.K_COMPILATION_UNIT, source, 0, source.length(),
        0, null));
    ApplyEdits(edits, document);
    return document.get();
  }

  /**
   * Get the compilation unit for a document.
   *
   * @param document The document to get the compilation unit for.
   * @return The compilation unit.
   */
  private static CompilationUnit GetCompilationUnit(final Document document) {
    ASTParser parser = ASTParser.newParser(AST.JLS10);
    parser.setSource(document.get().toCharArray());
    return (CompilationUnit) parser.createAST(null);
  }

  /**
   * Apply a list of edits to a document.
   *
   * @param edits The edits to apply.
   * @param document The document to apply the edits to.
   */
  private static void ApplyEdits(final List<TextEdit> edits,
      Document document) {
    for (TextEdit edit : edits) {
      System.err.println("Applying edit: " + edit.toString());
      try {
        edit.apply(document);
      } catch (BadLocationException e) {
        System.out.println("Failed to apply text edit!");
      }
    }
  }

  /**
   * Rewrite a Java source file to make it more amenable to machine learning.
   *
   * @param source The source code to rewrite.
   * @return The rewritten source code.
   */
  public String RewriteSource(String source) {
    Document document = new Document(source);
    CompilationUnit compilationUnit = GetCompilationUnit(document);
    ArrayList<TextEdit> edits = new ArrayList<>();
    StripComments(edits, compilationUnit);
    ApplyEdits(edits, document);
    source = document.get();
    edits.clear();
    document = new Document(source);
    compilationUnit = GetCompilationUnit(document);
    RewriteIdentifiers(edits, compilationUnit, document);
    ApplyEdits(edits, document);
    return FormatSource(document.get());
  }

  private ASTRewrite traversalRewrite;
  private AST traversalAST;

  private void RewriteIdentifiers(ArrayList<TextEdit> edits,
      final CompilationUnit compilationUnit,
      final Document document) {
    this.traversalRewrite = ASTRewrite.create(compilationUnit.getAST());
    this.traversalAST = compilationUnit.getAST();

    System.err.println("\n==========================\nBEGIN AST TRAVERSAL\n");
    compilationUnit.accept(new ASTVisitor() {

      private boolean ReplaceIdentifier(SimpleName old, final String newName) {
        traversalRewrite.replace(
            old, traversalAST.newSimpleName(newName), null);
        return true;
      }

      public boolean visit(Assignment node) {
        System.err.println("=> Assignment: " + node.toString());
        return true;
      }

      public boolean visit(ExpressionStatement node) {
        System.err.println("=> ExpressionStatement: " + node.toString());
        return true;
      }

      public boolean visit(FieldDeclaration node) {
        System.err.println("=> FieldDeclaration: " + node.toString());
        return true;
      }

      public boolean visit(ImportDeclaration node) {
        System.err.println("=> ImportDeclaration: " + node.toString());
        return true;
      }

      public boolean visit(Initializer node) {
        System.err.println("=> Initializer: " + node.toString());
        return true;
      }

      public boolean visit(MemberRef node) {
        System.err.println("=> MemberRef: " + node.toString());
        return true;
      }

      public boolean visit(MethodDeclaration node) {
        System.err.println("=> MethodDeclaration: " + node.toString());
        return true;
      }

      public boolean visit(MethodInvocation node) {
        // System.err.println("=> MethodInvocation: " + node.toString());
        return ReplaceIdentifier(node.getName(), "METHOD_INVOCATION");
      }

      public boolean visit(MethodRef node) {
        System.err.println("=> MethodRef: " + node.toString());
        return true;
      }

      public boolean visit(NameQualifiedType node) {
        System.err.println("=> NameQualifiedType: " + node.toString());
        return true;
      }

      public boolean visit(PackageDeclaration node) {
        System.err.println("=> PackageDeclaration: " + node.toString());
        return true;
      }

      public boolean visit(QualifiedName node) {
        System.err.println("=> QualifiedName: " + node.toString());
        return true;
      }

      public boolean visit(QualifiedType node) {
        System.err.println("=> QualifiedType: " + node.toString());
        return true;
      }

      public boolean visit(SimpleName node) {
        // System.err.println("=> SimpleName: " + node.toString());
        return ReplaceIdentifier(node, "SIMPLE_NAME");
      }

      public boolean visit(SimpleType node) {
        System.err.println("=> SimpleType: " + node.toString());
        return true;
      }

      public boolean visit(TypeDeclaration node) {
        System.err.println("=> TypeDeclaration: " + node.toString());
        return true;
      }
    });
    System.err.println("END AST TRAVERSAL\n==========================\n");

    edits.add(this.traversalRewrite.rewriteAST(document, null));
  }

  private static String ReadFile(String path, Charset encoding)
      throws IOException {
    byte[] encoded = Files.readAllBytes(Paths.get(path));
    return new String(encoded, encoding);
  }

  public static void main(final String[] args) {
    JavaRewriter rewriter = new JavaRewriter();
    // TODO(cec): Pass flags.
    final String path = args[0];
    try {
      final String input = ReadFile(path, Charset.defaultCharset());
      final String source = rewriter.RewriteSource(input);
      if (source == null) {
        System.out.println("fatal: RewriteSource() returned null.");
        System.exit(1);
      }

      System.out.println(source);
      // rewriter.TraverseAST(source);
    } catch (IOException e) {
      System.err.println("fatal: Could not read file: '" + path + "'.");
      System.exit(1);
    }
  }
}
