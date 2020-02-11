# Copyright 2019-2020 the ProGraML authors.
#
# Contact Chris Cummins <chrisc.101@gmail.com>.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from deeplearning.ml4pl.graphs import programl_pb2
from deeplearning.ml4pl.testing import random_programl_generator
from labm8.py import test

pytest_plugins = ["deeplearning.ml4pl.testing.fixtures.llvm_ir"]


@test.Fixture(scope="session")
def random_program_graph() -> programl_pb2.ProgramGraphProto:
  """A test fixture which yields a program graph."""
  return random_programl_generator.CreateRandomProto()
