"""A module for caching custom builds of LLVM."""
import contextlib
import pathlib
import subprocess
import sys
import typing

from labm8.py import app
from labm8.py import fs
from labm8.py import prof

FLAGS = app.FLAGS

app.DEFINE_string("llvm_version", "6.0.0", "The version of llvm to build.")
app.DEFINE_output_path(
  "llvm_build_prefix", "/var/phd/llvm", "The path to store LLVM builds in."
)


class LlvmBuild(object):
  """A class representing a local LLVM build."""

  def __init__(
    self,
    version: typing.Optional[str] = None,
    cmake_opts: typing.List[str] = None,
    build_prefix: typing.Optional[pathlib.Path] = None,
  ):
    build_prefix = build_prefix or FLAGS.llvm_build_prefix

    build_prefix.mkdir(exist_ok=True, parents=True)

    self.version = version or FLAGS.llvm_version
    cmake_opts = cmake_opts or []

    # Create a unique hash for the given llvm version and cmake options.
    build_id = str(
      hash(":".join([self.version] + sorted(cmake_opts)))
      % ((sys.maxsize + 1) * 2)
    )

    cmake_flags_file = build_prefix / f"{build_id}.cmake_flags.txt"
    fs.Write(cmake_flags_file, "\n".join(sorted(cmake_opts)).encode("utf-8"))

    self.install_dir = FLAGS.llvm_build_prefix / build_id

  @contextlib.contextmanager
  def Session(self, **build_opts) -> pathlib.Path:
    try:
      self._in_session = True
      self.MaybeBuild(**build_opts)
      yield self.install_dir
    finally:
      self._in_session = False

  @property
  def source_url(self) -> str:
    """Get the URL of the LLVM source tarball."""
    return (
      f"https://releases.llvm.org/{self.version}/"
      f"llvm-{self.version}.src.tar.xz"
    )

  def MaybeBuild(self, run_regression_test: bool = False) -> None:
    """Run a temporary build."""
    if (self.install_dir / "bin" / "llvm-config").is_file():
      return

    app.Log(1, "Beginning LLVM build. This may take some time!")
    with fs.TemporaryWorkingDir(prefix="phd_llvm_build_") as build_dir:
      app.Log(1, "Downloading source tarball")
      with prof.Profile(f"Downloaded {self.source_url}"):
        subprocess.check_call(["wget", self.source_url, "-O", "src.tar.xz"])

      app.Log(1, "Extracting source tarball")
      with prof.Profile("Extracted source tarball"):
        subprocess.check_call(["tar", "-xf", "src.tar.xz"])

      src_dir = build_dir / f"llvm-{self.version}.src"

      cmake_cmd = [
        "cmake",
        "-G",
        "Ninja",
        f"-DCMAKE_INSTALL_PREFIX={self.install_dir}",
        str(src_dir),
      ]
      app.Log(1, "Running `%s`", " ".join(cmake_cmd))
      with prof.Profile("Ran cmake"):
        subprocess.check_call(cmake_cmd)

      app.Log(1, "Running build")
      with prof.Profile("Ran ninja"):
        subprocess.check_call(["ninja"])

      if run_regression_test:
        app.Log(1, "Running regression tests")
        with prof.Profile("Ran regression tests"):
          subprocess.check_call(["ninja", "check-all"])

      app.Log(1, "Running install")
      with prof.Profile("Ran install"):
        subprocess.check_call(["ninja", "install"])

      app.Log(1, "Completed LLVM build and installed to %s", self.install_dir)


def main():
  """Main entry point."""
  with LlvmBuild().Session():
    pass


if __name__ == "__main__":
  app.Run(main)
