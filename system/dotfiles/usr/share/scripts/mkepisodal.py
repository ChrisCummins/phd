#!/usr/bin/env python3
"""Print the commands to rename files in directory to a sequential order.

The format for renamed files is: "<show> S<season>E<episode>.<ext>" format.
"""
from argparse import ArgumentParser
from pathlib import Path


def escape_path(path: str):
  quoted = path.replace('"', '\\"')
  return f'"{quoted}"'


def mkepisodal(show_name: str,
               season_num: int,
               directory: Path,
               start_at: int = 1):
  files = [
      f for f in sorted(directory.iterdir(), key=lambda s: s.name.lower())
      if f.is_file()
  ]

  commands = []
  # Consider each file extension seperately. This is because in most cases,
  # the file types for a set of videos is homogeneous, but there may be
  # subtitle (.srt) files running in tamden.
  extensions = set({f.suffix for f in files})
  for extension in extensions:
    # Ignore files that begin with a "." (i.e. hidden files on Unix).
    extension_files = [
        x for x in files if x.suffix == extension and not x.name.startswith(".")
    ]
    for i, episode in enumerate(extension_files):
      episode_num = i + start_at
      ext = episode.suffix
      newname = f"{show_name} S{season_num:02d}E{episode_num:02d}{ext}"

      src_path = escape_path(str(directory / episode.name))
      dst_path = escape_path(str(directory / newname))

      # Record the command to be emmitted later.
      commands.append(f"mv -v {src_path} {dst_path}")

  # Emit all commands at once so that they may be sorted alphabetically by file
  # name, not grouped by file extension.
  print('\n'.join(sorted(commands)))


def main():
  parser = ArgumentParser(description=__doc__)
  parser.add_argument("show_name",
                      metavar="<show-name>",
                      help="Name of the show, e.g. 'The Simponsons'")
  parser.add_argument("season_num",
                      metavar="<season number>",
                      type=int,
                      help="Season number, e.g. '1'")
  parser.add_argument("directory",
                      metavar="[directory]",
                      nargs="?",
                      default=".",
                      help="Path to the directory containing the show's files")
  parser.add_argument("--start-at",
                      metavar="<num>",
                      type=int,
                      default="1",
                      help="Episode start number (default: 1)")
  args = parser.parse_args()

  mkepisodal(args.show_name,
             args.season_num,
             Path(args.directory),
             start_at=args.start_at)


if __name__ == "__main__":
  main()
