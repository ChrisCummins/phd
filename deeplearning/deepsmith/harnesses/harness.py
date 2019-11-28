# Copyright (c) 2017, 2018, 2019 Chris Cummins.
#
# DeepSmith is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# DeepSmith is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with DeepSmith.  If not, see <https://www.gnu.org/licenses/>.
from deeplearning.deepsmith import services
from deeplearning.deepsmith.proto import harness_pb2
from labm8.py import app
from labm8.py import pbutil

FLAGS = app.FLAGS

app.DEFINE_string("harness_config", None, "Path to a harness config proto.")


class HarnessBase(services.ServiceBase):
  def __init__(self, config: pbutil.ProtocolBuffer):
    self.config = config
    self.testbeds = []

  def GetHarnessCapabilities(
    self, request: harness_pb2.GetHarnessCapabilitiesRequest, context
  ) -> harness_pb2.GetHarnessCapabilitiesResponse:
    raise NotImplementedError("abstract class")

  def RunTestcases(
    self, request: harness_pb2.RunTestcasesRequest, context
  ) -> harness_pb2.RunTestcasesResponse:
    raise NotImplementedError("abstract class")
