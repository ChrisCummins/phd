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
"""This module defines a container for storing epoch results."""
from typing import Optional

from labm8.py import shell
from programl.proto import epoch_pb2


def EpochToString(
  epoch: epoch_pb2.Epoch, previous: Optional[epoch_pb2.Epoch] = None
) -> str:
  previous = previous or epoch_pb2.Epoch()
  return "\n".join(
    [
      f"epoch {epoch.epoch_num} train {EpochResultsToString(epoch.train_results, previous.train_results)}",
      f"epoch {epoch.epoch_num} val {EpochResultsToString(epoch.val_results, previous.val_results)}"
      f"epoch {epoch.epoch_num} test {EpochResultsToString(epoch.test_results, previous.test_results)}",
    ]
  )


def EpochResultsToString(
  results: epoch_pb2.EpochResults,
  previous: Optional[epoch_pb2.EpochResults] = None,
) -> str:
  previous = previous or epoch_pb2.EpochResults()

  def Colorize(new, old, string):
    if new >= old:
      return f"{shell.ShellEscapeCodes.BOLD}{shell.ShellEscapeCodes.GREEN}{string}{shell.ShellEscapeCodes.END}"
    else:
      return f"{shell.ShellEscapeCodes.BOLD}{shell.ShellEscapeCodes.RED}{string}{shell.ShellEscapeCodes.END}"

  strings = [
    Colorize(
      results.mean_accuracy,
      previous.mean_accuracy,
      f"accuracy={results.mean_accuracy:.2%}",
    ),
    Colorize(
      results.mean_precision,
      previous.mean_precision,
      f"precision={results.mean_precision:.3f}",
    ),
    Colorize(
      results.mean_recall,
      previous.mean_recall,
      f"recall={results.mean_recall:.3f}",
    ),
    Colorize(results.mean_f1, previous.mean_f1, f"f1={results.mean_f1:.3f}"),
  ]
  if results.mean_has_loss:
    strings.append(
      Colorize(
        results.mean_loss,
        previous.mean_loss or 0,
        f"loss={results.mean_loss:.6f}",
      )
    )

  return ", ".join(strings)
