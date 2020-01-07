# Copyright 2019 Chris Cummins <chrisc.101@gmail.com>.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Shared code for LLVM modules."""
import typing

from labm8.py import app

FLAGS = app.FLAGS


class LlvmError(EnvironmentError):
  def __init__(
    self,
    msg: typing.Optional[str] = None,
    returncode: typing.Optional[int] = None,
    stderr: typing.Optional[str] = None,
    command: typing.Optional[typing.List[str]] = None,
  ):
    self.msg = msg
    self.returncode = returncode
    self.stderr = stderr
    self.command = command

  def __repr__(self):
    if self.command:
      msg = f"{' '.join(self.command)} ->"
    else:
      msg = ""

    if self.returncode:
      msg = f"{msg} [returncode={self.returncode}]"

    if self.msg:
      msg = f"{msg} {self.msg}"
    elif self.stderr:
      msg = f"{msg} {self.msg}"
    else:
      msg = f"{msg} {type(self).__name__}"

    return msg

  def __str__(self):
    return repr(self)


class LlvmTimeout(LlvmError):
  pass
