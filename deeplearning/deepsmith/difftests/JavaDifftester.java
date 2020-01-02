package deeplearning.deepsmith.difftests;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;

public class JavaDifftester {

  private static String getUsage() {
    return "Usage: JavaDifftester <directory>";
  }

  // The mode used to create argument values.
  enum JavadDifftestOutcome {
    // Majority outcome is not a build failure/crash, but this result is.
    ANOMALOUS_BUILD_FAILURE,
    // Majority outcome is a build failure, but this result builds.
    ANOMALOUS_BUILD_PASS,
    // Majority outcome is a pass, but this result crashes.
    ANOMALOUS_RUNTIME_CRASH,
    // Majority outcome is a runtime crash, but this result passes.
    ANOMALOUS_RUNTIME_PASS,
    // Outputs differ between this result and the majority.
    ANOMALOUS_WRONG_OUTPUT,
    // Majority pass, but this result times out.
    ANOMALOUS_RUNTIME_TIMEOUT,
    // Outcome and outputs agrees with majority.
    PASS,
  }

  /**
   * Main entry point. Run differential testing on the given method.
   *
   * @param methodSrc
   */
  public static void differentialTest(final String methodSrc) {
    System.out.println(methodSrc);
  }

  public static void main(String[] args) {
    if (args.length != 1) {
      System.err.println(getUsage());
      System.exit(1);
    }

    final String testDir = args[0];
    File[] files = new File(testDir).listFiles();

    try {
      for (File file : files) {
        System.out.println("File: " + file.getName());
        final String methodSrc = readFileToString(file.getAbsolutePath());
        differentialTest(methodSrc);
      }
    } catch (IOException e) {
      System.err.println(e);
      System.exit(1);
    }
  }

  private static String readFileToString(final String path) throws IOException {
    InputStream is = new FileInputStream(path);
    BufferedReader buf = new BufferedReader(new InputStreamReader(is));

    String line = buf.readLine();
    StringBuilder sb = new StringBuilder();
    while (line != null) {
      sb.append(line).append("\n");
      line = buf.readLine();
    }

    return sb.toString();
  }
}
