"""Script to transcode.

This file depends on external binaries 'find' and 'ffmpeg'.
"""
import csv
import datetime
import os
import re
import subprocess
import typing

from labm8 import app
from labm8 import fs
from labm8 import humanize
from labm8 import system


FLAGS = app.FLAGS

app.DEFINE_string('root_dir', fs.path('~/Music/Music Library'), 'Directory of ')
app.DEFINE_string('csv_log_path',
                  fs.path('~/Music/Music Library/transcode_mp3s.csv'),
                  'Path to CSV log file.')
app.DEFINE_string(
    'find_bin', 'find', 'Path of `find` binary. Set this flag is find is not '
    'in the system $PATH.')
app.DEFINE_string(
    'ffmpeg_bin', 'ffmpeg',
    'Path of `ffmpeg` binary. Set this flag is ffmpeg is not '
    'in the system $PATH.')


def FindMp3Files(dir: str) -> typing.List[str]:
  """List mp3 files in a directory and subdirectories."""
  cmd = [FLAGS.find_bin, dir, '-type', 'f', '-name', '*.mp3']
  output = subprocess.check_output(cmd, universal_newlines=True)
  return output.rstrip().split('\n')


def TranscodeMp3(in_path: str, out_path: str, lame_option: int) -> None:
  """Overwrite output files."""
  cmd = [
      FLAGS.ffmpeg_bin, '-y', '-loglevel', 'panic', '-i', in_path, '-codec:a',
      'libmp3lame', '-qscale:a',
      str(lame_option), out_path
  ]
  subprocess.check_call(cmd)


def MaybeTranscodeMp3(path: str,
                      out_csv: csv.writer,
                      max_bitrate: int = 192,
                      lame_option: int = 3) -> bool:
  """

  See https://trac.ffmpeg.org/wiki/Encode/MP3 for a table of variable bitrate
  values.
  """
  output = subprocess.check_output(['file', path], universal_newlines=True)
  match = re.search(r' (\d+) kbps', output)
  bit_rate = int(match.group(1))

  if bit_rate > max_bitrate:
    size_before = os.path.getsize(path)
    system.ProcessFileAndReplace(
        path,
        lambda inpath, outpath: TranscodeMp3(inpath, outpath, lame_option),
        tempfile_prefix='phd_system_machines_florence_transcode_musiclib_',
        tempfile_suffix='.mp3')
    size_after = os.path.getsize(path)
    app.Log(1, f'%s changed from %s to %s (%.1f%% reduction)',
             os.path.basename(path), humanize.BinaryPrefix(size_before, 'B'),
             humanize.BinaryPrefix(size_after, 'B'),
             (1 - (size_after / size_before)) * 100)
    out_csv.writerow(
        [datetime.datetime.now(), path, bit_rate, size_before, size_after])
    return True
  else:
    app.Log(2, 'Ignoring %s', path)
    return False


def main(argv):
  if len(argv) > 1:
    raise app.UsageError("Unknown arguments.")

  files = FindMp3Files(FLAGS.root_dir)
  with open(FLAGS.csv_log_path, 'a') as f:
    out_csv = csv.writer(f)
    for f in files:
      try:
        MaybeTranscodeMp3(f, out_csv)
      except Exception:
        app.Warning('Failed to process %s', f)


if __name__ == '__main__':
  app.RunWithArgs(main)
