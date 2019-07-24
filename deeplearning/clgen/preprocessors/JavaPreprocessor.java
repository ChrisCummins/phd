// Pre-process java files.
package deeplearning.clgen.preprocessors;

import com.google.common.io.ByteStreams;
import datasets.github.scrape_repos.ScrapeReposProtos.ListOfStrings;
import deeplearning.clgen.InternalProtos.PreprocessorWorkerJobOutcome;
import deeplearning.clgen.InternalProtos.PreprocessorWorkerJobOutcomes;
import java.io.BufferedReader;
import java.io.IOException;
import java.io.StringReader;
import java.util.concurrent.Callable;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.TimeoutException;
import org.apache.commons.jci.compilers.CompilationResult;
import org.apache.commons.jci.compilers.EclipseJavaCompiler;
import org.apache.commons.jci.compilers.EclipseJavaCompilerSettings;
import org.apache.commons.jci.compilers.JavaCompilerSettings;
import org.apache.commons.jci.problems.CompilationProblem;
import org.apache.commons.jci.problems.CompilationProblemHandler;
import org.apache.commons.jci.readers.MemoryResourceReader;
import org.apache.commons.jci.stores.MemoryResourceStore;
import org.apache.commons.jci.stores.ResourceStore;
import org.eclipse.jdt.core.dom.AST;
import org.eclipse.jdt.core.dom.ASTParser;
import org.eclipse.jdt.core.dom.ASTVisitor;
import org.eclipse.jdt.core.dom.CompilationUnit;
import org.eclipse.jdt.core.dom.MethodDeclaration;
import org.eclipse.jface.text.Document;

public final class JavaPreprocessor {

  // Configuration options.
  private static final int MIN_CHAR_COUNT = 50;
  private static final int MIN_LINE_COUNT = 4;
  private static final int PREPROCESS_TIMEOUT_SECONDS = 60;
  // End of configuration options.

  /** Construct a preprocessor. */
  public JavaPreprocessor() {
    settings = new EclipseJavaCompilerSettings();
    // Bump Java source version. Version >= 1.5 is required for generics.
    settings.setSourceVersion("1.7");

    compiler = new EclipseJavaCompiler(settings);
    classloader = JavaPreprocessor.class.getClassLoader();

    compiler.setCompilationProblemHandler(
        new CompilationProblemHandler() {
          @Override
          public boolean handle(final CompilationProblem problem) {
            if (problem.isError()) {
              // Stop on first error.
              return false;
            }
            // Ignore warnings.
            return true;
          }
        });
  }

  /**
   * Wrap the method in a class definition.
   *
   * @param methodSrc The method to wrap.
   * @return The method, embedded in a class "A".
   */
  protected String WrapMethodInClass(final String methodSrc) {
    return "public class A{" + methodSrc + "}";
  }

  /**
   * Wrap the method in a class definition.
   *
   * @param methodSrc The method to wrap.
   * @return The method, embedded in a class "A".
   */
  public String WrapMethodInClassWithShim(final String methodSrc) {
    // Template:
    //     import java.io.*;
    //     import java.math.*;
    //     import java.nio.charset.*;
    //     import java.nio.file.*;
    //     import java.time.format.*;
    //     import java.util.*;
    //     public class A {
    //       /* METHOD GOES HERE */
    //     }
    return ("import java.io.*;\n"
        + "import java.math.*;\n"
        + "import java.nio.charset.*;\n"
        + "import java.nio.file.*;\n"
        + "import java.time.format.*;\n"
        + "import java.util.*;\n"
        + "public class A{"
        + methodSrc
        + "}");
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

  protected String UnwrapMethodInClassOr(final String src) {
    Document document = new Document(src);
    CompilationUnit compilationUnit = GetCompilationUnit(document);

    // Assign to array because variable must be final.
    final String[] method = new String[1];

    compilationUnit.accept(
        new ASTVisitor() {
          public boolean visit(MethodDeclaration node) {
            method[0] = node.toString();
            return false; // Stop iteration at the first method.
          }
        });
    if (method[0] == null) {
      throw new IllegalArgumentException("Could not unwrap method in class: " + src);
    }
    return method[0];
  }

  /**
   * Rewrite a Java method.
   *
   * @param methodSrc The method source string.
   * @return The rewritten method.
   * @throws TimeoutException
   * @throws ExecutionException
   * @throws InterruptedException
   */
  protected String RewriteSource(final String methodSrc)
      throws TimeoutException, ExecutionException, InterruptedException {
    final String wrappedSrc = WrapMethodInClass(methodSrc);
    final String rewrittenSrc =
        new JavaRewriter().RewriteSource(wrappedSrc, "A.java", PREPROCESS_TIMEOUT_SECONDS);
    return UnwrapMethodInClassOr(rewrittenSrc);
  }

  /**
   * Determine if the given source string compiles without error.
   *
   * <p>WARNING: The source string must define a class "A"! As produced by WrapMethodInClass().
   *
   * @param src The source string for a java class.
   * @return True if compiles, false if errors.
   */
  protected boolean CompilesWithoutError(final String src) {
    // Prepare input to feed to compiler.
    final String fakeInputPath = "A.java";
    MemoryResourceReader input = new MemoryResourceReader();
    input.add("A.java", src.getBytes());

    // Write result in memory.
    final ResourceStore unusedOutput = new MemoryResourceStore();

    // Assumes that class is named "A".
    final String[] resources = {"A.java"};
    JavaCompilerSettings settings = compiler.createDefaultSettings();

    try {
      final CompilationResult result =
          compiler.compile(resources, input, unusedOutput, classloader);
      return result.getErrors().length == 0;
    } catch (NullPointerException e) {
      // A bug in compiler raised this error:
      //
      //     Exception in thread "main" java.lang.NullPointerException
      //     at
      // org.eclipse.jdt.internal.compiler.ast.QualifiedNameReference.optimizedBooleanConstant(QualifiedNameReference.java:926)
      //     at
      // org.eclipse.jdt.internal.compiler.ast.CastExpression.optimizedBooleanConstant(CastExpression.java:553)
      //     at
      // org.eclipse.jdt.internal.compiler.ast.UnaryExpression.resolveType(UnaryExpression.java:299)
      //     at org.eclipse.jdt.internal.compiler.ast.Assignment.resolveType(Assignment.java:211)
      //     at org.eclipse.jdt.internal.compiler.ast.Expression.resolve(Expression.java:1031)
      //     at org.eclipse.jdt.internal.compiler.ast.Block.resolve(Block.java:122)
      //     at org.eclipse.jdt.internal.compiler.ast.IfStatement.resolve(IfStatement.java:280)
      //     at org.eclipse.jdt.internal.compiler.ast.Block.resolve(Block.java:122)
      //     at org.eclipse.jdt.internal.compiler.ast.IfStatement.resolve(IfStatement.java:280)
      //     ...
      return false;
    }
  }

  /**
   * Check that string contains a minimum number of lines.
   *
   * @param text The text to check.
   * @param minLineCount The minimum nuber of lines.
   * @return True if text has >= minimum number of lines.
   */
  protected boolean TextHasMinimumLineCount(final String text, final int minLineCount) {
    BufferedReader bufReader = new BufferedReader(new StringReader(text));
    try {
      int lineCount = 0;
      while ((bufReader.readLine()) != null) {
        ++lineCount;
        if (lineCount >= minLineCount) {
          return true;
        }
      }
    } catch (IOException e) {
      System.err.println("fatal: IO error encountered in TextHasMinimumLineCount()");
      System.exit(1);
    }
    return false;
  }

  public PreprocessorWorkerJobOutcome PreprocessSourceOrDie(
      final String src, final int timeoutSeconds) {
    try {
      ExecutorService executor = Executors.newCachedThreadPool();
      Callable<PreprocessorWorkerJobOutcome> task =
          new Callable<PreprocessorWorkerJobOutcome>() {
            public PreprocessorWorkerJobOutcome call() {
              return PreprocessSourceOrDie(src);
            }
          };

      Future<PreprocessorWorkerJobOutcome> future = executor.submit(task);
      return future.get(timeoutSeconds, TimeUnit.SECONDS);
    } catch (Exception e) {
      PreprocessorWorkerJobOutcome.Builder message = PreprocessorWorkerJobOutcome.newBuilder();
      message.setStatus(PreprocessorWorkerJobOutcome.Status.DOES_NOT_COMPILE);
      message.setContents("Preprocess timeout");
      return message.build();
    }
  }

  /**
   * Preprocess a single Java source text.
   *
   * @param src The source to preprocess.
   * @return A PreprocessorWorkerJobOutcome message with the results of preprocessing.
   */
  public PreprocessorWorkerJobOutcome PreprocessSourceOrDie(final String src) {
    PreprocessorWorkerJobOutcome.Builder message = PreprocessorWorkerJobOutcome.newBuilder();
    String contents = src;

    if (!CompilesWithoutError(WrapMethodInClassWithShim(contents))) {
      message.setStatus(PreprocessorWorkerJobOutcome.Status.DOES_NOT_COMPILE);
      message.setContents("Failed to compile");
      return message.build();
    }

    try {
      contents = RewriteSource(contents);
    } catch (Exception e) {
      message.setStatus(PreprocessorWorkerJobOutcome.Status.REWRITER_FAIL);
      message.setContents("Failed to rewrite");
      return message.build();
    }

    if (contents.length() < MIN_CHAR_COUNT) {
      message.setStatus(PreprocessorWorkerJobOutcome.Status.TOO_FEW_CHARACTERS);
      message.setContents("Fewer than " + MIN_CHAR_COUNT + " chars");
      return message.build();
    }

    if (!TextHasMinimumLineCount(contents, MIN_LINE_COUNT)) {
      message.setStatus(PreprocessorWorkerJobOutcome.Status.TOO_FEW_LINES);
      message.setContents("Fewer than " + MIN_LINE_COUNT + " lines");
      return message.build();
    }

    // Re-run compilation. We already checked if the code compiles prior to
    // re-writing. Checking that the code compiles again is a safeguard against
    // shortcomings in the re-writer where code can "break" after re-writing.
    if (!CompilesWithoutError(WrapMethodInClassWithShim(contents))) {
      message.setStatus(PreprocessorWorkerJobOutcome.Status.DOES_NOT_COMPILE);
      message.setContents("Failed to compile after re-writing");
      return message.build();
    }

    message.setStatus(PreprocessorWorkerJobOutcome.Status.OK);
    message.setContents(contents);
    return message.build();
  }

  /**
   * Preprocess a list of Java source texts.
   *
   * @param srcs The sources to preprocess.
   * @return A PreprocessorWorkerJobOutcomes message with the results of preprocessing.
   */
  public PreprocessorWorkerJobOutcomes PreprocessInputsOrDie(final ListOfStrings srcs) {
    PreprocessorWorkerJobOutcomes.Builder message = PreprocessorWorkerJobOutcomes.newBuilder();

    for (int i = 0; i < srcs.getStringCount(); ++i) {
      final long startTime = System.currentTimeMillis();
      PreprocessorWorkerJobOutcome outcome =
          PreprocessSourceOrDie(srcs.getString(i), PREPROCESS_TIMEOUT_SECONDS);
      message.addOutcome(outcome);
      message.addPreprocessTimeMs(System.currentTimeMillis() - startTime);
    }

    return message.build();
  }

  private final EclipseJavaCompiler compiler;
  private final EclipseJavaCompilerSettings settings;
  private ClassLoader classloader;

  private static ListOfStrings GetInputOrDie() {
    try {
      return ListOfStrings.parseFrom(ByteStreams.toByteArray(System.in));
    } catch (IOException e) {
      System.err.println("fatal: IO error");
      System.exit(1);
      return null;
    }
  }

  public static void main(String[] args) throws IOException {
    ListOfStrings input = GetInputOrDie();
    JavaPreprocessor preprocessor = new JavaPreprocessor();
    preprocessor.PreprocessInputsOrDie(input).writeTo(System.out);
  }
}
