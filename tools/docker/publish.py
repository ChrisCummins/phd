"""Build and publish a docker image.

This is a work-in-progress script, it is not yet ready for prime-time!
"""
import pathlib
import subprocess

import getconfig
from labm8 import app
from labm8 import dockerutil

FLAGS = app.FLAGS

app.DEFINE_string('target', None, 'The bazel image target.')
app.DEFINE_string('tag', None, 'The docker image tag to export.')
app.DEFINE_boolean('push', False,
                   'Run `docker push` on the final tagged image.')

PHD_BUILD = dockerutil.BazelPy3Image('tools/docker/phd_build/phd_build')


def BuildAndLoad(target: str) -> str:
  assert target.startswith('//')
  target_without_prefix = target[len('//'):]

  target_components = target.split(':')
  assert len(target_components) == 2
  path = target_components[0] + '/' + target_components[1]
  tar_target = f'{target}.tar'

  app.Log(1, 'Building %s image', tar_target)
  with PHD_BUILD.RunContext() as ctx:
    ctx.CheckCall(['bazel', 'build', tar_target], timeout=600)

  phd_root = getconfig.GetGlobalConfig().paths.repo_root
  tar_path = pathlib.Path(f'{phd_root}/bazel-bin/{path}.tar')
  assert tar_path.is_file()

  # Load the tarfile build
  import_tag = f'bazel/{target_without_prefix}'
  app.Log(1, 'Loading docker image %s', import_tag)
  subprocess.check_call(
      ['timeout', '-s9',
       str(300), 'docker', 'load', '-i', tar_path])
  return import_tag


def RenameTag(src_tag: str, dst_tag: str) -> None:
  app.Log(1, 'Tagging %s', dst_tag)
  subprocess.check_call(
      ['timeout', '-s9',
       str(60), 'docker', 'tag', src_tag, dst_tag])
  subprocess.check_call(['timeout', '-s9', str(60), 'docker', 'rmi', src_tag])


def PushTag(tag: str):
  app.Log(1, 'Pushing docker image %s', tag)
  subprocess.check_call(['timeout', '-s9', str(360), 'docker', 'push', tag])


def main():
  """Main entry point."""
  loaded_tag = BuildAndLoad(FLAGS.target)
  if FLAGS.tag:
    RenameTag(loaded_tag, FLAGS.tag)
    if FLAGS.push:
      PushTag(FLAGS.tag)


if __name__ == '__main__':
  app.Run(main)
