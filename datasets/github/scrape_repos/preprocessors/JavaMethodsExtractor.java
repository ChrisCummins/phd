/* Extract methods from Java source file. */
package datasets.github.scrape_repos.preprocessors;

import com.google.common.io.ByteStreams;
import datasets.github.scrape_repos.ScrapeReposProtos.MethodsList;
import java.io.IOException;
import org.eclipse.jdt.core.dom.AST;
import org.eclipse.jdt.core.dom.ASTParser;
import org.eclipse.jdt.core.dom.ASTVisitor;
import org.eclipse.jdt.core.dom.CompilationUnit;
import org.eclipse.jdt.core.dom.MethodDeclaration;
import org.eclipse.jface.text.Document;

/**
 * Extract methods from Java source code.
 */
public class JavaMethodsExtractor {

  private MethodsList.Builder message;

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
   * Extract all methods from a Java source.
   *
   * @param source A Java source string.
   * @return A MethodsList proto instance.
   */
  private MethodsList ExtractMethods(String source) {
    Document document = new Document(source);
    CompilationUnit compilationUnit = GetCompilationUnit(document);
    message = MethodsList.newBuilder();
    compilationUnit.accept(new ASTVisitor() {
      public boolean visit(MethodDeclaration node) {
        message.addMethod(node.toString());
        return true;
      }
    });
    return message.build();
  }

  public static void main(final String[] args) {
    JavaMethodsExtractor extractor = new JavaMethodsExtractor();

    try {
      final String input = new String(ByteStreams.toByteArray(System.in));
      System.out.println(extractor.ExtractMethods(input).toString());
    } catch (IOException e) {
      System.err.println("fatal: I/O error");
      System.exit(1);
    }
  }
}
