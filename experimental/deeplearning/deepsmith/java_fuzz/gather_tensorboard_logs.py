"""Gather and serve tensorboard logs."""
import pathlib
import shutil
import tempfile
import time
import typing

import fs
from tensorboard import default
from tensorboard import program

from labm8.py import app
from labm8.py import system

FLAGS = app.FLAGS

app.DEFINE_input_path(
  "src_logdir", None, "Path to store CLgen cache files.", is_dir=True
)
app.DEFINE_list(
  "hosts",
  list(sorted({"cc1", "cc2", "cc3"} - {system.HOSTNAME})),
  "List of hosts to gather data from",
)


def OpenFsFromPaths(paths: typing.List[str]):
  return [fs.open_fs(path) for path in paths]


def ParseTensorBoardPath(path: pathlib.Path) -> typing.Tuple[str, str]:
  if not path.name.startswith("events.out.tfevents."):
    raise ValueError(f"Invalid log path: `{path}`")
  name = path.name[len("events.out.tfevents.") :]
  components = name.split(".")
  timestamp, host = components[0], ".".join(components[1:])
  return host, timestamp


def GatherAndArrangeLogs(paths, workdir: pathlib.Path) -> pathlib.Path:
  """Gather """
  indir = workdir / "in"
  outdir = workdir / "out"
  indir.mkdir(exist_ok=True)

  logdirs = OpenFsFromPaths(paths)

  # Gather files.
  if outdir.is_dir():
    shutil.rmtree(outdir)
  outdir.mkdir()

  dst_fs = fs.open_fs(str(indir))
  for path, logdir in zip(paths, logdirs):
    app.Log(1, "Gathering tensorboard events from: %s", path)
    fs.copy.copy_fs(logdir, dst_fs)

  # Arrange files.
  for path in indir.iterdir():
    host, timestamp = ParseTensorBoardPath(path)
    run_name = f"{host}_{timestamp}"
    dst = outdir / run_name / path.name
    dst.parent.mkdir()
    app.Log(1, "mv '%s' '%s'", path, dst)
    path.rename(dst)
  shutil.rmtree(indir)

  return outdir


def GatherAndServe(logdir: pathlib.Path, hosts: typing.List[str]):
  paths = [str(logdir)] + [f"ssh://{host}{logdir}" for host in FLAGS.hosts]

  with tempfile.TemporaryDirectory(prefix="phd_java_fuzz_") as d:
    working_dir = pathlib.Path(d)
    outdir = GatherAndArrangeLogs(paths, working_dir)

    # Launch tensorboard
    tensorboard = program.TensorBoard(
      default.get_plugins() + default.get_dynamic_plugins(),
      program.get_default_assets_zip_provider(),
    )
    tensorboard.configure(["unused_arg0", "--logdir", str(outdir)])
    tensorboard_url = tensorboard.launch()
    app.Log(1, "Serving tensorboard at %s", tensorboard_url)

    while True:
      GatherAndArrangeLogs(paths, working_dir)
      time.sleep(10)


def main():
  """Main entry point."""
  GatherAndServe(FLAGS.src_logdir, FLAGS.hosts)


if __name__ == "__main__":
  app.Run(main)
