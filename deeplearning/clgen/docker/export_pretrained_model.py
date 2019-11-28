"""Script to build a docker image for a pre-trained CLgen instance."""
import pathlib
import subprocess
import tempfile
import typing

from deeplearning.clgen import clgen
from labm8.py import app
from labm8.py import fs
from labm8.py import pbutil
from research.cummins_2017_cgo import generative_model

FLAGS = app.FLAGS

app.DEFINE_string(
  "export_instance_config", None, "Path of CLgen instance proto to export."
)
app.DEFINE_string(
  "docker_base_image",
  "chriscummins/clgen:latest",
  "The name of the base docker image to build using.",
)


def ExportInstance(
  instance: clgen.Instance,
  export_dir: pathlib.Path,
  docker_base_image: typing.Optional[str] = None,
) -> None:
  """Export a self-contained CLgen instance to a directory.

  Args:
    instance: The instance to export.
    export_dir: The directory to export files to.
    docker_base_image: The name of the base docker image for the generated
      Dockerfile.
  """
  config = instance.ToProto()

  instance.ExportPretrainedModel(export_dir / "model")
  config.pretrained_model = "/clgen/model"

  config.working_dir = "/clgen"
  pbutil.ToFile(config, export_dir / "config.pbtxt")

  fs.Write(
    export_dir / "Dockerfile",
    f"""
FROM {docker_base_image}
MAINTAINER Chris Cummins <chrisc.101@gmail.com>

ADD . /clgen
""".encode(
      "utf-8"
    ),
  )


def main():
  """Main entry point."""
  instance = generative_model.CreateInstanceFromFlags()

  with tempfile.TemporaryDirectory(prefix="deeplearning_clgen_docker_") as d:
    ExportInstance(instance, pathlib.Path(d), FLAGS.docker_base_image)
    subprocess.check_call(["docker", "build", d])


if __name__ == "__main__":
  app.Run(main)
