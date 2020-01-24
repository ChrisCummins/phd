package deeplearning.clgen.preprocessors;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertTrue;

import com.clgen.PreprocessorWorkerJobOutcome;
import labm8.java.bazelutil.BazelRunfiles;
import org.junit.Test;

public class JavaPreprocessorTest {

  @Test
  public void testCompilesWithoutErrorEmptyClass() throws Exception {
    JavaPreprocessor pp = new JavaPreprocessor();
    assertTrue(pp.CompilesWithoutError("public class A {}"));
  }

  @Test
  public void testCompilesWithoutErrorEmptyFile() throws Exception {
    JavaPreprocessor pp = new JavaPreprocessor();
    assertTrue(pp.CompilesWithoutError(""));
  }

  @Test
  public void testCompilesWithoutErrorWithEventListener() throws Exception {
    JavaPreprocessor pp = new JavaPreprocessor();
    assertTrue(
        pp.CompilesWithoutError(
            "import java.util.EventListener;\n"
                + "public class A {\n"
                + "public static void Foo(EventListener a) {}}"));
  }

  @Test
  public void testCompilesWithoutErrorWithGeneric() throws Exception {
    JavaPreprocessor pp = new JavaPreprocessor();
    assertTrue(
        pp.CompilesWithoutError(
            "import java.util.ArrayList;\n"
                + "public class A {\n"
                + "public static void Foo(ArrayList<Integer> a) {}}"));
  }

  @Test
  public void testCompilesWithoutErrorFail() throws Exception {
    JavaPreprocessor pp = new JavaPreprocessor();
    assertFalse(pp.CompilesWithoutError("syntax error!"));
    assertFalse(pp.CompilesWithoutError("public static void Foo() {}"));

    // Class has wrong name.
    assertFalse(pp.CompilesWithoutError("public class B {}"));
  }

  @Test
  public void testPreprocessSourceOrDieRegressionTest() throws Exception {
    JavaPreprocessor pp = new JavaPreprocessor();

    // Regression test using a source found on GitHub that stalled my rewriter.
    //
    // The workaround for this input was to add a timeout to the rewriter to
    // prevent huge inputs from causing the code formatter to run for
    // excessively long periods of time.
    //
    // Original source:
    //    https://github.com/calphool/romannumeralskata
    //    RomanNumerals/src/com/rounceville/RomanNumeralList.java
    final String src =
        BazelRunfiles.getDataString(
            "phd/deeplearning/clgen/tests/data/java_preprocessor_regression_test_1.java");
    PreprocessorWorkerJobOutcome outcome = pp.PreprocessSourceOrDie(src);
    assertEquals(outcome.getStatus(), PreprocessorWorkerJobOutcome.Status.REWRITER_FAIL);
    assertEquals(outcome.getContents(), "Failed to rewrite");
  }
}
