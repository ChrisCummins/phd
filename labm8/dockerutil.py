"""A module for launching docker images from within python applications."""
import contextlib
import pathlib
import random
import subprocess
import typing

from labm8 import app
from labm8 import bazelutil


def IsDockerContainer() -> bool:
  """Determine if running inside a docker container."""
  return pathlib.Path('/.dockerenv').is_file()


class DockerImageRunContext(object):
  """A transient context for running docker images."""

  def __init__(self, image_name: str):
    self.image_name = image_name

  def _CommandLineInvocation(
      self, args: typing.List[str], flags: typing.Dict[str, str],
      volumes: typing.Dict[typing.Union[str, pathlib.Path], str]):
    volume_args = [f'-v{src}:{dst}' for src, dst in (volumes or {}).items()]
    flags_args = [f'--{k}={v}' for k, v in (flags or {}).items()]
    return (['docker', 'run'] + volume_args + [self.image_name] + args +
            flags_args)

  def CheckCall(
      self,
      args: typing.List[str],
      flags: typing.Dict[str, str] = None,
      volumes: typing.Dict[typing.Union[str, pathlib.Path], str] = None):
    """Run docker image."""
    cmd = self._CommandLineInvocation(args, flags, volumes)
    app.Log(2, "$ %s", " ".join(cmd))
    subprocess.check_call(cmd)

  def CheckOutput(
      self,
      args: typing.List[str],
      flags: typing.Dict[str, str] = None,
      volumes: typing.Dict[typing.Union[str, pathlib.Path], str] = None) -> str:
    cmd = self._CommandLineInvocation(args, flags, volumes)
    app.Log(2, "$ %s", " ".join(cmd))
    return subprocess.check_output(cmd, universal_newlines=True)


class BazelPy3Image(object):
  """TODO

  To use one a py3_image within a script, add the py3_image target with a
  ".tar" suffix as a data dependency of the script.
  """

  def __init__(self, data_path: str):
    """Constructor.

    Args:
      path: The path to the data, including the name of the workspace.

    Raises:
      FileNotFoundError: If path is not a file.
    """
    super(BazelPy3Image, self).__init__()
    self.data_path = data_path
    self.tar_path = bazelutil.DataPath(f'phd/{data_path}.tar')

    components = self.data_path.split('/')
    self.image_name = f'bazel/{"/".join(components[:-1])}:{components[-1]}'

  def _TemporaryImageName(self) -> str:
    basename = self.data_path.split('/')[-1]
    random_suffix = ''.join(
        random.choice('0123456789abcdef') for _ in range(32))
    return f'phd_{basename}_tmp_{random_suffix}'

  @contextlib.contextmanager
  def RunConext(self) -> DockerImageRunContext:
    subprocess.check_call(['docker', 'load', '-i', str(self.tar_path)])
    tmp_name = self._TemporaryImageName()
    subprocess.check_call(['docker', 'tag', self.image_name, tmp_name])
    subprocess.check_call(['docker', 'rmi', self.image_name])
    yield DockerImageRunContext(tmp_name)
    subprocess.check_call(['docker', 'rmi', tmp_name])


def test_entire_pipeline(tempdir: pathlib.Path):
  scrape_java_files_image = BazelPy3Image(
      'experimental/deeplearning/deepsmith/java_fuzz/scrape_java_files_image')
  split_contentfiles_image = BazelPy3Image(
      'experimental/deeplearning/deepsmith/java_fuzz/scrape_java_files_image')
  export_java_corpus_image = BazelPy3Image(
      'experimental/deeplearning/deepsmith/java_fuzz/scrape_java_files_image')

  # Step 1: Scrape a single repo from GitHub.
  with scrape_java_files_image.RunContext() as ctx:
    ctx.Run([], {
        "n": 1,
        "db": '////sqlite:////workdir/java.db',
    },
            volumes={
                tempdir: '/workdir',
                '/var/phd': '/var/phd',
            })

  # Check that contentfiles database is created.
  assert (tempdir / "java.db").is_file()

  # Step 2: Export a subset of the contentfiles database.
  with split_contentfiles_image.RunContext() as ctx:
    ctx.Run(
        [], {
            "n": 1,
            "input": '////sqlite:////workdir/java.db',
            "output": '////sqlite:////workdir/subset.db',
        },
        volumes={tempdir: '/workdir'})

  # Check that contentfiles database is created.
  assert (tempdir / "java.db").is_file()
  assert (tempdir / "subset.db").is_file()

  # Step 3: Export Java corpus.
  with export_java_corpus_image.RunContext() as ctx:
    ctx.Run(
        [], {
            "n":
            1,
            "db":
            '////sqlite:////workdir/java.db',
            "outdir":
            "/workdir/corpus",
            "preprocessors":
            "datasets.github.scrape_repos.preprocessors.extractors:JavaStaticMethods",
        },
        volumes={tempdir: '/workdir'})

  # Check that corpus is exported.
  assert (tempdir / "java.db").is_file()
  assert (tempdir / "corpus").is_dir()
