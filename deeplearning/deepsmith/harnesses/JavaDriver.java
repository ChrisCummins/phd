package deeplearning.deepsmith.harnesses;

import java.io.*;
import java.lang.reflect.*;
import java.net.URL;
import java.net.URLClassLoader;
import java.util.*;
import java.util.Arrays;
import java.util.stream.*;
import javax.tools.JavaCompiler;
import javax.tools.JavaFileObject;
import javax.tools.StandardJavaFileManager;
import javax.tools.StandardLocation;
import javax.tools.ToolProvider;

public class JavaDriver {
  enum Mode {
    DEFAULT,
    RANDOM;
  }

  /**
   * Given a parameter type, generates an array of default values, where "default" is determined in
   * the sense of {@link #generate(Class<?>, Mode) generatePrimitiveDefault}
   *
   * @param arrayType The type of the array
   * @param arrayLength The length of the array
   * @param m The generation mode. Either DEFAULT or RANDOM
   */
  private static Object generateArrayDefault(Class<?> arrayType, int arrayLength, Mode m) {
    Object arrayDefaults = Array.newInstance(arrayType, arrayLength);
    for (int i = 0; i < arrayLength; ++i) Array.set(arrayDefaults, i, generate(arrayType, m));

    return arrayDefaults;
  }

  /**
   * Given a parameter type, generates an array of random values, where "random" is determined in
   * the sense of {@link #generate(Class<?>, Mode, long) generatePrimitiveRandom}
   *
   * @param arrayType The type of the array
   * @param arrayLength The length of the array
   * @param m The generation mode. Either DEFAULT or RANDOM
   * @param seed The seed for random number generation
   */
  private static Object generateArrayRandom(
      Class<?> arrayType, int arrayLength, Mode m, long seed) {
    Object arrayDefaults = Array.newInstance(arrayType, arrayLength);
    for (int i = 0; i < arrayLength; ++i) Array.set(arrayDefaults, i, generate(arrayType, m, seed));
    return arrayDefaults;
  }

  /**
   * Given a parameter type that is a primitive value, generates a default value that matches the
   * given type. That is, if the type is a boolean , generates true a character, generates 'A' any
   * number , generates 0 otherwise , generates null when under DEFAULT mode. Otherwise, randomly
   * generates values for each primitive type
   *
   * @param parameterType The supplied type, should be one of Boolean.TYPE, Character.TYPE,
   *     Byte.TYPE , Short.TYPE, Integer.TYPE, Long.TYPE, Float.TYPE , Double.TYPE
   */
  private static Object generatePrimitiveDefault(Class<?> parameterType) {
    if (parameterType == Boolean.TYPE) {
      System.out.printf("  Instantiating boolean to default true\n");
      return true;
    } else if (parameterType == Character.TYPE) {
      System.out.printf("  Instantiating char to default 'A'\n");
      return 'A';
    } else if (parameterType == Byte.TYPE
        || parameterType == Short.TYPE
        || parameterType == Integer.TYPE
        || parameterType == Long.TYPE
        || parameterType == Float.TYPE
        || parameterType == Double.TYPE) {
      System.out.printf("  Instantiating number to default 0\n");
      return 0;
    }
    return null;
  }

  private static Object generatePrimitiveRandom(Class<?> parameterType, long seed) {
    Random randomno = new Random();
    randomno.setSeed(seed);
    if (parameterType == Boolean.TYPE) {
      System.out.printf(" Instantiating boolean to a random value\n");
      return randomno.nextBoolean();
    } else if (parameterType == Character.TYPE) {
      System.out.printf(" Instantiating char to a random value\n");
      return (char) (randomno.nextInt(95) + 32);
    } else if (parameterType == Byte.TYPE) {
      System.out.printf(" Instantiating byte to a random value\n");
      byte[] value = new byte[1];
      randomno.nextBytes(value);
      return value[0];
    } else if (parameterType == Short.TYPE) {
      System.out.printf(" Instantiating short to a random value\n");
      return (short) (randomno.nextInt(65536) - 32768);
    } else if (parameterType == Integer.TYPE) {
      System.out.printf(" Instantiating int to a random value\n");
      return randomno.nextInt();
    } else if (parameterType == Long.TYPE) {
      System.out.printf(" Instantiating long to a random value\n");
      return randomno.nextLong();
    } else if (parameterType == Float.TYPE) {
      System.out.printf(" Instantiating float to a random value\n");
      return Float.MIN_VALUE + (Float.MAX_VALUE - Float.MIN_VALUE) * randomno.nextFloat();
    } else if (parameterType == Double.TYPE) {
      System.out.printf(" Instantiating double to a random value\n");
      return Double.MIN_VALUE + (Double.MAX_VALUE - Double.MIN_VALUE) * randomno.nextDouble();
    }
    return null;
  }

  /**
   * Given a parameter type, generates a default value. That is, if parameterType is a primitive,
   * generate a primitive value using {@link #generatePrimitiveDefault(Class<?>)
   * generatePrimitiveDefault}. Otherwise, try to use the default constructor to generate a default.
   * If this fails, then return null.
   */
  private static Object generate(Class<?> parameterType, Mode m) {
    if (parameterType.isPrimitive()) return generatePrimitiveDefault(parameterType);
    else if (parameterType.isArray())
      return generateArrayDefault(parameterType.getComponentType(), 1, m);
    else {
      // For non-primitive and non-arrays, try using the default constructor.
      try {
        return parameterType.getConstructor().newInstance();
      } catch (NoSuchMethodException e) {
        System.err.printf("Cannot find constructor for class %s\n", parameterType.getName());
        e.printStackTrace();
        return null;
      } catch (InvocationTargetException e) {
        System.err.printf("Invocation of constructor failed because it threw an exception\n");
        e.printStackTrace();
        return null;
      } catch (InstantiationException e) {
        System.err.printf(
            "Class %s is abstract/interface/primitive/void or has no empty constructor\n",
            parameterType.getName());
        e.printStackTrace();
        return null;
      } catch (IllegalAccessException e) {
        System.err.printf(
            "Cannot access the empty constructor of class %s\n", parameterType.getName());
        e.printStackTrace();
        return null;
      }
    }
  }

  // Random generation with overload of previous function, seed is provided.
  private static Object generate(Class<?> parameterType, Mode m, long seed) {
    if (parameterType.isPrimitive()) return generatePrimitiveRandom(parameterType, seed);
    else if (parameterType.isArray()) return generateArrayRandom(parameterType, 1, m, seed);
    else {
      // For non-primitive and non-arrays, try using the default constructor.
      try {
        return parameterType.getConstructor().newInstance();
      } catch (NoSuchMethodException e) {
        System.err.printf("Cannot find constructor for class %s\n", parameterType.getName());
        e.printStackTrace();
        return null;
      } catch (InvocationTargetException e) {
        System.err.printf("Invocation of constructor failed because it threw an exception\n");
        e.printStackTrace();
        return null;
      } catch (InstantiationException e) {
        System.err.printf(
            "Class %s is abstract/interface/primitive/void or has no empty constructor\n",
            parameterType.getName());
        e.printStackTrace();
        return null;
      } catch (IllegalAccessException e) {
        System.err.printf(
            "Cannot access the empty constructor of class %s\n", parameterType.getName());
        e.printStackTrace();
        return null;
      }
    }
  }

  public static void main(String[] args) throws Throwable {

    Mode mode = Mode.DEFAULT;
    long seed;
    Scanner in = new Scanner(System.in);
    System.out.print("Use random input value (y/n): ");
    while (true) {
      if (in.hasNextLine()) {
        String result = in.nextLine();
        if (result.equalsIgnoreCase("y")) {
          mode = Mode.RANDOM;
          System.out.print("Will fill in random input value, please provide the seed: ");
          Scanner sc = new Scanner(System.in);
          do {
            try {
              seed = sc.nextLong();
              break;

            } catch (InputMismatchException e) {
            } finally {
              sc.nextLine(); // always advances (even after the break)
            }
            System.out.println("Input must be a number!");
            System.out.print("please provide the seed: ");
          } while (true);
          sc.close();
        } else if (result.equalsIgnoreCase("n")) {
          System.out.println("Will fill in default input value");
        } else {
          System.out.println("Invalid input, please enter again!");
          System.out.print("Use random input value (y/n): ");
          continue;
        }
        break;
      }
    }

    in.close();

    // create an empty source file
    File sourceFile = File.createTempFile("Dummy", ".java");
    sourceFile.deleteOnExit();

    // generate the source code, using the source filename as the class name
    String classname = sourceFile.getName().split("\\.")[0];
    System.out.println(classname);
    String sourceCode = "public class " + classname + " {";
    File dir = new File("src/test");
    File[] directoryListing = dir.listFiles();
    for (File child : directoryListing) {
      System.out.println(child);
      try {
        BufferedReader bf = new BufferedReader(new FileReader(child));
        String temp = "";
        boolean flag = false;
        while ((temp = bf.readLine()) != null) {
          if (!temp.contains("RE-WRITTEN")) {
            if (flag) {
              sourceCode += temp;
            } else {
              continue;
            }
          } else {
            flag = true;
          }
        }
        bf.close();
      } catch (Exception e) {
        System.out.println(child);
        e.printStackTrace();
      }
    }
    sourceCode += "}";

    /*
     * try (PrintWriter pw = new PrintWriter(new FileWriter(sourceFile))) {
     * pw.println(sourceCode); }
     */
    // compile the source file
    JavaCompiler compiler = ToolProvider.getSystemJavaCompiler();
    // The default file manager the compiler uses
    StandardJavaFileManager fileManager = compiler.getStandardFileManager(null, null, null);
    File parentDirectory = sourceFile.getParentFile();
    // convert the provided source file into a JavaFileObjects that serve as the
    // compilation units
    fileManager.setLocation(StandardLocation.CLASS_OUTPUT, Arrays.asList(parentDirectory));
    Iterable<? extends JavaFileObject> compilationUnits =
        fileManager.getJavaFileObjectsFromFiles(Arrays.asList(sourceFile));
    // Now, create and invoke the compilation task
    compiler
        .getTask(
            // The Writer for any compiler output. null means use System.err
            null,
            // Use the standard file manager provided by the compiler
            fileManager,
            // Use the compiler's default method for reporting compilation diagonistics
            null,
            null,
            // No annotations are to be processed
            null,
            compilationUnits)
        .call();
    fileManager.close();

    // load the compiled class
    URLClassLoader classLoader =
        URLClassLoader.newInstance(new URL[] {parentDirectory.toURI().toURL()});
    Class<?> testClass = classLoader.loadClass(classname);
    List<Object> invocationResults =
        Arrays.stream(testClass.getDeclaredMethods())
            .map(
                (Method m) -> {
                  System.out.printf("Method name:     %s\n", m.getName());
                  System.out.printf(
                      "Parameter types: %s\n",
                      Arrays.stream(m.getParameterTypes())
                          .map(Class::getName)
                          .collect(Collectors.joining(", ")));
                  System.out.printf("Return type:     %s\n", m.getReturnType().getName());

                  List<Object> actualArguments = new ArrayList<>();
                  for (Class<?> parameterType : m.getParameterTypes()) {
                    System.out.printf("- Creating instance of %s\n", parameterType.getName());
                    System.out.printf("  Primitive: %s\n", parameterType.isPrimitive());
                    System.out.printf("  Array: %s\n", parameterType.isArray());
                    if (parameterType.isArray()) {
                      System.out.printf(
                          "  Array type: %s\n", parameterType.getComponentType().getName());
                      actualArguments.add(
                          parameterType.cast(generate(parameterType, Mode.DEFAULT)));
                    } else actualArguments.add(generate(parameterType, Mode.DEFAULT));
                  }

                  System.out.printf("Arguments: %s\n", actualArguments);
                  System.out.printf("Invoking method...\n");
                  Object result = null;
                  try {
                    // In invoke, first argument should be an instance of the class the method
                    // belongs to,
                    // but because the methods right now are just static, the first argument is
                    // ignored so
                    // making the first argument null.
                    result = m.invoke(null, actualArguments.toArray());
                  } catch (IllegalAccessException e) {
                    System.out.printf("Method %s is unaccessible\n", m.getName());
                    e.printStackTrace();
                  } catch (InvocationTargetException e) {
                    System.out.printf(
                        "Invocation of method %s failed because it threw an exception\n",
                        m.getName());
                    e.printStackTrace();
                  }
                  System.out.printf("Invocation result succeeded with = %s\n", result);

                  System.out.println();

                  return result;
                })
            .collect(Collectors.toList());
  }
}
