"""A script to concatenate a sequence of video files produced a dashcam into
contiguous videos.

For a set of video files in a directory, this script determines the contiguous
ranges of them and produces a concatenated video file for each contiguous range.

This script depends on ffmpeg and ffprobe being installed in the system path.
"""
import datetime
import os
import pathlib
import re
import subprocess
import tempfile
import typing

from labm8.py import app
from labm8.py import humanize
from labm8.py import labtypes
from util.photolib import dashcam

FLAGS = app.FLAGS
app.DEFINE_input_path("working_dir",
                      os.getcwd(),
                      "Working directory.",
                      is_dir=True)
app.DEFINE_output_path(
    "output_dir",
    None,
    "Directory to write concatenated segments to. If not provided, outputs are written to --working_dir",
    is_dir=True)
app.DEFINE_string(
    "segment_prefix", "segment_",
    "The prefix for the names of concatenated dash video segments.")
app.DEFINE_boolean("rm", False,
                   "Delete original video files after concatenating them.")

# A regex to the match the line of output printed by ffprobe containing the
# video duration.
_DURATION_RE = re.compile(
    r"Duration: (?P<hours>\d\d):(?P<minutes>\d\d):(?P<seconds>\d\d)\.\d\d")


def GetDurationOfVideoOrDie(path: pathlib.Path) -> datetime.timedelta:
  """Read the duration of a video file using ffprobe."""
  result = subprocess.check_output(["ffprobe", path.absolute()],
                                   stderr=subprocess.STDOUT,
                                   universal_newlines=True)
  durations = [x for x in result.split("\n") if "Duration" in x]
  if not len(durations) == 1:
    app.FatalWithoutStackTrace("Could not determine duration of video `%s`",
                               path)
  m = _DURATION_RE.search(durations[0])
  if not m:
    app.FatalWithoutStackTrace("Failed to pass duration `%s`", durations[0])

  parts = m.groupdict()
  time_params = {}
  for (name, param) in parts.items():
    if param:
      time_params[name] = int(param)
  return datetime.timedelta(**time_params)


def Segmentize(files: typing.Iterable[pathlib.Path]
              ) -> typing.Iterator[typing.List[pathlib.Path]]:
  """Group a list of dashcam video files into contiguous sequences."""
  files = list(sorted(files))
  segment_duration = datetime.timedelta()
  while files:
    for i in range(1, len(files)):
      duration = GetDurationOfVideoOrDie(files[i - 1])
      segment_duration += duration

      curr_date = dashcam.ParseDatetimeFromFilenameOrDie(files[i].name)
      prev_date = dashcam.ParseDatetimeFromFilenameOrDie(files[i - 1].name)
      timestamp_delta = curr_date - prev_date

      if abs(timestamp_delta.seconds - duration.seconds) > 2:
        yield files[:i], segment_duration
        files = files[i:]
        segment_duration = datetime.timedelta()
        break
    else:
      yield files, segment_duration
      break


def ConcatenateSegment(entries: typing.List[pathlib.Path],
                       target: str) -> pathlib.Path:
  with tempfile.TemporaryDirectory(prefix="phd_dashcam_concat_") as d:
    concat_list = pathlib.Path(d) / "concat.txt"

    with open(concat_list, "w") as f:
      for entry in entries:
        print(f"file '{entry.absolute()}'", file=f)

    cmd = [
        "ffmpeg", "-f", "concat", "-safe", "0", "-i",
        str(concat_list), "-c", "copy",
        str(target)
    ]
    app.Log(1, "$ %s", " ".join(cmd))
    subprocess.check_call(cmd)
  if not target.is_file():
    app.FatalWithoutStackTrace("Failed to concatenate %d entries", len(entries))

  return target


def main():
  """Main entry point."""
  working_dir = FLAGS.working_dir

  files = [x for x in working_dir.iterdir() if not x.name.startswith('.')]

  # Segmentize all of the videos before concatenating so that we can sanity
  # check the sequences.
  sequences = []
  for sequence, duration in Segmentize(files):
    app.Log(1, "Segment with %s, duration %s",
            humanize.Plural(len(sequence), "entry", "entries"), duration)
    for j, f in enumerate(sequence):
      app.Log(1, "  entry %s: %s", j + 1, f.name)
    sequences.append(sequence)

  # Sanity check that segments do not contain duplicates.
  sequences_files = labtypes.flatten(sequences)
  sequences_files_set = set(f.name for f in sequences_files)
  if len(sequences_files_set) != len(sequences_files):
    app.FatalWithoutStackTrace(
        "%s segments contain duplicates. Expected %s files, found %s",
        len(sequences), len(sequences_files), len(sequences_files_set))

  # Sanity check that segments contain all files.
  if len(sequences_files) != len(files):
    files_set = set(f.name for f in files)
    app.Error("Files not included in segments:\n    %s",
              "    \n".join(files_set - sequences_files_set))
    app.FatalWithoutStackTrace("%d segments contain %s files, expected %s",
                               len(sequences), len(sequences_files), len(files))

  # Concatenate each sequence.
  for i, sequence in enumerate(sequences):
    base_file = sequence[0]
    start_date = dashcam.ParseDatetimeFromFilenameOrDie(base_file.name)

    # Use a separate output directory if required.
    output_dir = base_file.parent if FLAGS.output_dir is None else FLAGS.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    target = output_dir / f'{FLAGS.segment_prefix}{start_date.strftime("%Y%m%dT%H%M%S")}.part_{i + 1}_of_{len(sequences)}{base_file.suffix.lower()}'
    app.Log(1, "Creating sequence %s of %s: %s", i + 1, len(sequences),
            target.name)
    ConcatenateSegment(sequence, target)

    # Delete the files when we're done with them, if required.
    if FLAGS.rm:
      for f in sequence:
        f.unlink()

  app.Log(1, "Concatenated %s sequences, done.", len(sequences))


if __name__ == "__main__":
  app.Run(main)
