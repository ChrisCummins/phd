"""Report the sizes of data tiers.

Data directories are classified by a "tier", described in
//system/machines/proto/data_tiers.proto. This program reports the sizes of
these data directories.
"""
import os
import pathlib
import subprocess

import humanize
import pandas as pd
from absl import app
from absl import flags
from absl import logging

from labm8 import pbutil
from system.machines.proto import data_tiers_pb2

FLAGS = flags.FLAGS

flags.DEFINE_string('data_tiers', None, 'The path of the directory to package.')
flags.DEFINE_bool('summary', False, 'TODO')

flags.register_validator(
    'data_tiers',
    lambda path: pbutil.ProtoIsReadable(path, data_tiers_pb2.DataTiers()),
    message='--data_tiers must be a DataTiers message.')


def _SetDirectorySize(tier: data_tiers_pb2.Directory):
  path = pathlib.Path(tier.path).expanduser()
  if not path.is_dir():
    logging.warning("Directory '%s' not found", path)
    return

  os.chdir(path)
  excludes = [
      '--exclude={}'.format(pathlib.Path(e).expanduser()) for e in tier.exclude
  ]
  cmd = ['du', '-b', '-s', '.'] + excludes
  logging.info('$ cd %s && %s', path, ' '.join(cmd))
  proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, universal_newlines=True)
  stdout, _ = proc.communicate()
  if proc.returncode:
    raise OSError

  size = int(stdout.split('\t')[0])
  tier.size_bytes = size


def main(argv) -> None:
  """Main entry point."""
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')

  tiers = pbutil.FromFile(
      pathlib.Path(FLAGS.data_tiers), data_tiers_pb2.DataTiers())
  for tier in tiers.directory:
    logging.info('Processing %s', tier.path)
    _SetDirectorySize(tier)

  if FLAGS.summary:
    # Print the size per directory.
    df = pd.DataFrame([{
        'Path': d.path,
        'Tier': d.tier,
        'Size': humanize.naturalsize(d.size_bytes),
        'Size (bytes)': d.size_bytes
    } for d in tiers.directory if d.size_bytes])
    df = df.sort_values(['Tier', 'Size (bytes)'], ascending=[True, False])
    print(df[['Path', 'Tier', 'Size']].to_string(index=False))

    # Print the total size per tier.
    df2 = df.groupby('Tier').sum()
    df2['Size'] = [
        humanize.naturalsize(d['Size (bytes)']) for _, d in df2.iterrows()
    ]
    df2 = df2.reset_index()
    df2 = df2.sort_values('Tier')
    print()
    print("Totals:")
    print(df2[['Tier', 'Size']].to_string(index=False))
  else:
    print(tiers)


if __name__ == '__main__':
  app.run(main)
