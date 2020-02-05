// Python binding for accessing MNIST dataset.
#include "datasets/mnist/cpp/mnist_data_loader.h"
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "labm8/cpp/statusor.h"

namespace py = pybind11;

namespace mnist {

PYBIND11_MODULE(mnist_data_loader, m) {
  m.doc() = "This module provides access to the MNIST dataset.";

  py::class_<LabeledImages>(m, "LabeledImages")
      .def(py::init<>())
      .def_readonly("images", &LabeledImages::images)
      .def_readonly("labels", &LabeledImages::labels);

  py::class_<Mnist>(m, "Mnist")
      .def(py::init<>())
      .def_readonly("train", &Mnist::train)
      .def_readonly("test", &Mnist::test);

  py::class_<MnistDataLoader>(m, "MnistDataLoader")
      .def(py::init<>())
      .def("Load", [](MnistDataLoader &d) {
        return labm8::StatusOrToException(d.Load());
      });
}

}  // namespace mnist
