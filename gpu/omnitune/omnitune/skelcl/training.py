import random


def random_wg_value(max_wg_size):
  wg_c = random.randrange(2, max_wg_size / 2, 2)
  wg_r = random.randrange(2, max_wg_size / 2, 2)
  while wg_c * wg_r > max_wg_size:
    wg_c = random.randrange(2, max_wg_size / 2, 2)
    wg_r = random.randrange(2, max_wg_size / 2, 2)
  return [wg_c, wg_r]


simple_kernel = """
// Type of input and output data.
typedef float DATA_T;


// "Simple" kernel. Returns the average of all neighbouring element values.
DATA_T func(input_matrix_t *img) {
        // Number of neighbouring elements.
        int numElements = (SCL_NORTH + SCL_SOUTH + 1) * (SCL_WEST + SCL_EAST + 1);
        DATA_T sum = 0;

        // Loop over all neighbouring elements.
        for (int y = -SCL_NORTH; y <= SCL_SOUTH; y++) {
                for (int x = -SCL_WEST; x <= SCL_EAST; x++) {
                        // Sum values of neighbouring elements.
                        sum += getData(img, y, x);
                }
        }

        // If/then/else branch:
        DATA_T out = (int)sum % 2 ? sum : 0;

        return out;
}
"""

complex_kernel = """
// Type of input and output data.
typedef float DATA_T;


// "Complex" kernel. Performs lots of trigonometric heavy lifting.
DATA_T func(input_matrix_t *img) {
        DATA_T sum = 0;

        // Iterate over all except outer neighbouring elements.
        for (int y = -SCL_NORTH + 1; y < SCL_SOUTH; y++) {
                for (int x = -SCL_WEST + 1; x < SCL_EAST; x++) {
                        // Do *some* computation on values.
                        DATA_T a = sin((float)getData(img, -1, 0));
                        DATA_T b = native_sin((float)getData(img, 0, 1) * a);
                        sum += getData(img, y, x) * a * (b / b);
                }
        }

        DATA_T out = 0;
        // Loop over horizontal region.
        for (int i = SCL_EAST; i >= -SCL_WEST; i--) {
                // DO *some* computation on values.
                sum *= cos((DATA_T)((int)getData(img, 0, i) % i) + sqrt((float)getData(img, 0, 0)));
                out += sinpi((float)sum);
                out /= sum;
        }

        return out;
}
"""

common_header = """
#include <algorithm>
#include <fstream>
#include <iostream>
#include <iterator>
#include <cstdlib>

#include <pvsutil/CLArgParser.h>
#include <pvsutil/Logger.h>

#include <SkelCL/SkelCL.h>
#include <SkelCL/Stencil.h>

using namespace skelcl;

typedef float DATA_T;
"""

main = """
int main(int argc, char** argv)
{
        using namespace pvsutil;
        using namespace pvsutil::cmdline;

        CLArgParser cmd(Description("Synthetic kernel."));

        // Parse arguments.
        auto verbose = Arg<bool>(Flags(Short('v'), Long("verbose")),
                                 Description("Enable verbose logging."),
                                 Default(false));

        auto iterations = Arg<size_t>(Flags(Short('i'), Long("iterations")),
                                      Description("Number of iterations."),
                                      Default<size_t>(10));

        auto deviceCount = Arg<int>(Flags(Long("device-count")),
                                    Description("Number of devices used by SkelCL."),
                                    Default(1));

        auto deviceType = Arg<device_type>(Flags(Long("device-type")),
                                           Description("Device type: ANY, CPU, "
                                                       "GPU, ACCELERATOR"),
                                           Default(device_type::ANY));

        cmd.add(&verbose, &iterations, &deviceCount, &deviceType);
        cmd.parse(argc, argv);

        if (verbose)
                defaultLogger.setLoggingLevel(Logger::Severity::DebugInfo);

        init(nDevices(deviceCount).deviceType(deviceType));


        // Create input and populate with random values.
        Matrix<DATA_T> data({HEIGHT, WIDTH});
        for (size_t y = 0; y < data.rowCount(); y++)
                for (size_t x = 0; x < data.columnCount(); x++)
                        data[y][x] = static_cast<DATA_T>(rand());

        // Create stencil.
        Stencil<DATA_T(DATA_T)> stencil =
            Stencil<DATA_T(DATA_T)>(KERNEL, "func",
                                    stencilShape(north(NORTH), west(WEST),
                                                 south(SOUTH), east(EAST)),
                                    Padding::NEUTRAL, 0);

        // Run stencil.
        for (size_t i = 0; i < iterations; i++)
          data = stencil(data);

        // Copy data back to host.
        data.copyDataToHost();

        return 0;
}
"""


def define(name, val):
  return "#define {name} {val}".format(name=name, val=val)


def escape_kernel(kernel):
  return 'R"(' + kernel + ')";'


def make_synthetic_benchmark(complexity, north, south, east, west, width,
                             height):
  program = [common_header]

  program.append(define("HEIGHT", height))
  program.append(define("WIDTH", width))
  program.append(define("NORTH", north))
  program.append(define("SOUTH", south))
  program.append(define("EAST", east))
  program.append(define("WEST", west))

  program.append("const char *KERNEL = ")

  if complexity > .5:
    program.append(escape_kernel(complex_kernel))
  else:
    program.append(escape_kernel(simple_kernel))

  program.append(main)

  return "\n".join(program)


# sizes = (512, 1024, 2048, 4096)
# stencil_direction_values(1, 5, 10, 20, 30)
