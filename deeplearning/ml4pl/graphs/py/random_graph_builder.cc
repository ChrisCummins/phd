// This file defines the pybind11 bindings for xla2graph.
//
// Copyright 2019-2020 the ProGraML authors.
//
// Contact Chris Cummins <chrisc.101@gmail.com>.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//    http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
#include "deeplearning/ml4pl/graphs/random_graph_builder.h"
#include <pybind11/pybind11.h>
#include <sstream>

namespace py = pybind11;

namespace ml4pl {

PYBIND11_MODULE(random_graph_builder, m) {
  m.doc() = "Class for building random graphs";

  py::class_<RandomGraphBuilder>(m, "RandomGraphBuilder")
      .def(py::init<>())
      .def("GetSerializedGraphProto",
           [&](RandomGraphBuilder& builder, int nodeCount) {
             ProgramGraphProto graph =
                 builder.FastCreateRandom(nodeCount).ValueOrException();
             std::stringstream str;
             graph.SerializeToOstream(&str);
             return py::bytes(str.str());
           },
           py::arg("node_count") = 0);
}

}  // namespace ml4pl
