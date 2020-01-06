#include <pybind11/pybind11.h>

int add(int a, int b) { return a + b; }

// IMPORTANT: The first argument must be the name of the module, else
// compilation will succeed but import will fail with:
//     ImportError: dynamic module does not define module export function
//     (PyInit_example)
PYBIND11_MODULE(example, m) {
  m.doc() = "pybind11 example plugin";

  m.def("add", &add, "A function which adds two numbers");
}
