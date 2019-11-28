"""A script to concatenate a sequence of video files produced a dashcam into
contiguous videos.

For a set of video files in a directory, this script determines the contiguous
ranges of them and produces a concatenated video file for each contiguous range.

This script depends on ffmpeg and ffprobe being installed in the system path.
"""
import datetime
import os

from labm8.py import app
from util.photolib import dashcam

FLAGS = app.FLAGS
app.DEFINE_input_path("working_dir",
                      os.getcwd(),
                      "Working directory.",
                      is_dir=True)
app.DEFINE_integer("minutes", 0,
                   "The number of minutes to offset timestamps by.")


def main():
  """Main entry point."""
  working_dir = FLAGS.working_dir

  files = [x for x in working_dir.iterdir() if not x.name.startswith('.')]

  offset = datetime.timedelta(seconds=FLAGS.minutes * 60)

  # Work from front to back when offset is negative, else from back to front.
  # This prevents renaming conflicts.
  if offset.seconds < 0:
    order = lambda x: x
  else:
    order = lambda x: reversed(x)

  for file in order(list(sorted(files))):
    date = dashcam.ParseDatetimeFromFilenameOrDie(file.name)
    new_name = dashcam.DatetimeToFilename(date + offset)

    new_path = file.parent / new_name

    app.Log(1, "Rename %s -> %s", file.name, new_name)
    assert not new_path.is_file()
    os.rename(file, new_path)


if __name__ == "__main__":
  app.Run(main)
