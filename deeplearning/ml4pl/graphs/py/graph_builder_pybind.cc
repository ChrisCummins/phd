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
#include <pybind11/pybind11.h>
#include <sstream>
#include "deeplearning/ml4pl/graphs/graph_builder.h"

namespace py = pybind11;

namespace ml4pl {

PYBIND11_MODULE(graph_builder_pybind, m) {
  m.doc() = "Class for building graphs";

  py::class_<GraphBuilder>(m, "GraphBuilder")
      .def(py::init<>())
      .def("GetSerializedGraphProto",
           [&](GraphBuilder& builder) {
             ProgramGraphProto graph = builder.GetGraph().ValueOrException();
             std::stringstream str;
             graph.SerializeToOstream(&str);
             return py::bytes(str.str());
           })
      .def("AddFunction",
           [&](GraphBuilder& builder, const string& name) {
             return builder.AddFunction(name).ValueOrException();
           },
           py::arg("name"))
      .def("AddStatement",
           [&](GraphBuilder& builder, const string& text, int function) {
             return builder.AddStatement(text, function).ValueOrException();
           },
           py::arg("text"), py::arg("function"))
      .def("AddIdentifier",
           [&](GraphBuilder& builder, const string& text, int function) {
             return builder.AddIdentifier(text, function).ValueOrException();
           },
           py::arg("text"), py::arg("function"))
      .def("AddImmediate",
           [&](GraphBuilder& builder, const string& text) {
             return builder.AddImmediate(text).ValueOrException();
           },
           py::arg("text"))
      .def("AddControlEdge",
           [&](GraphBuilder& builder, int sourceNode, int destinationNode) {
             builder.AddControlEdge(sourceNode, destinationNode)
                 .RaiseException();
           },
           py::arg("source_node"), py::arg("destination_node"))
      .def("AddCallEdge",
           [&](GraphBuilder& builder, int sourceNode, int destinationNode) {
             builder.AddCallEdge(sourceNode, destinationNode).RaiseException();
           },
           py::arg("source_node"), py::arg("destination_node"))
      .def("AddDataEdge",
           [&](GraphBuilder& builder, int sourceNode, int destinationNode,
               int position) {
             builder.AddDataEdge(sourceNode, destinationNode, position)
                 .RaiseException();
           },
           py::arg("source_node"), py::arg("destination_node"),
           py::arg("position") = 0);
}

}  // namespace ml4pl
