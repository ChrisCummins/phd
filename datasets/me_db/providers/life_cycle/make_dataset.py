# Copyright 2018-2020 Chris Cummins <chrisc.101@gmail.com>
#
# Permission is hereby granted, free of charge, to any person obtaining a copy of
# this software and associated documentation files (the "Software"), to deal in
# the Software without restriction, including without limitation the rights to
# use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of
# the Software, and to permit persons to whom the Software is furnished to do so,
# subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
# FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
# COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
# IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
# CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""A module to generate random LifeCycle datasets."""
import csv
import io
import pathlib
import random
import tempfile
import time
import typing
import zipfile

from labm8.py import app

FLAGS = app.FLAGS


class RandomDatasetGenerator(object):
  """A random data generator for Life Cycle."""

  def __init__(
    self,
    start_time_seconds_since_epoch: float,
    locations: typing.List[str],
    names: typing.List[str],
  ):
    self.start_time_seconds_since_epoch = start_time_seconds_since_epoch
    self.locations = locations
    self.names = names

  def Sample(self, writer: csv.writer, num_rows: int) -> None:
    """Generate num_rows random rows and write to given CSV."""
    # LifeCycle CSV file has whitespace after commas.
    writer.writerow(
      [
        "START DATE(UTC)",
        " END DATE(UTC)",
        " START TIME(LOCAL)",
        " END TIME(LOCAL)",
        " DURATION",
        " NAME",
        " LOCATION",
        " NOTE",
      ]
    )
    writer.writerow([])
    start_time = self.start_time_seconds_since_epoch + random.randint(
      0, 3600 * 23
    )
    for _ in range(num_rows):
      start_time += random.randint(1, 3600)
      end_time = start_time + random.randint(60, 3600 * 8)

      writer.writerow(
        [
          time.strftime(
            "%Y-%m-%d %H:%M:%S", time.localtime(start_time)
          ),  # START DATE(UTC)
          time.strftime(
            " %Y-%m-%d %H:%M:%S", time.localtime(end_time)
          ),  # END DATE(UTC)
          " unused",  # START TIME(LOCAL)
          " unused",  # END TIME(LOCAL)
          " unused",  # DURATION
          " " + random.choice(self.names),  # NAME
          " " + random.choice(self.locations),  # LOCATION
          " unused",  # NOTE
        ]
      )

  def SampleFile(self, path: pathlib.Path, *sample_args) -> pathlib.Path:
    """Run sample and write result to a file."""
    with open(path, "w") as f:
      writer = csv.writer(f, lineterminator="\n")
      self.Sample(writer, *sample_args)
    return path

  def SampleZip(self, path: pathlib.Path, *sample_args) -> pathlib.Path:
    """Run sample and write result to a zipfile."""
    # Create a CSV file in a temporary directory.
    with tempfile.TemporaryDirectory(prefix="phd_") as d:
      csv_path = pathlib.Path(d) / "LC_export.csv"
      self.SampleFile(csv_path, *sample_args)
      # Create the zip file and add the CSV.
      with zipfile.ZipFile(path, "w") as z:
        z.write(csv_path, arcname="LC_export.csv")
    return path


def main(argv):
  """Main entry point."""
  if len(argv) > 1:
    raise app.UsageError("Unrecognized command line flags.")

  generator = RandomDatasetGenerator(
    start_time_seconds_since_epoch=time.mktime(
      time.strptime("1/1/2018", "%m/%d/%Y")
    ),
    locations=["My House", "The Office", "A Restaurant",],
    names=[
      "Work",
      "Home",
      "Sleep",
      "Fun",
      "Commute to work",
      "Commute to home",
    ],
  )
  buf = io.StringIO()
  generator.Sample(csv.writer(buf), 20)
  print(buf.getvalue().rstrip())


if __name__ == "__main__":
  app.RunWithArgs(main)
