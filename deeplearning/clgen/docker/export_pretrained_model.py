"""Script to build a docker image for a pre-trained CLgen instance."""
import pathlib
import subprocess
import tempfile

from deeplearning.clgen import clgen
from labm8 import app
from labm8 import fs
from labm8 import pbutil

FLAGS = app.FLAGS

app.DEFINE_string('export_instance_config', None,
                  'Path of CLgen instance proto to export.')
app.DEFINE_string('docker_base_image', 'chriscummins/clgen:190315',
                  'The name of the base docker image to build using.')


def main():
  """Main entry point."""
  config = clgen.ConfigFromFlags()
  instance = clgen.Instance(config)

  with tempfile.TemporaryDirectory(prefix='deeplearning_clgen_docker_') as d:
    export_dir = pathlib.Path(d)
    config = instance.ToProto()

    instance.ExportPretrainedModel(export_dir / 'model')
    config.pretrained_model = '/clgen/model'

    config.working_dir = '/clgen'
    pbutil.ToFile(config, export_dir / 'config.pbtxt')

    fs.Write(
        export_dir / 'Dockerfile', f"""
FROM {FLAGS.docker_base_image}
MAINTAINER Chris Cummins <chrisc.101@gmail.com>

ADD . /clgen
""".encode('utf-8'))
    subprocess.check_call(['docker', 'build', str(export_dir)])


if __name__ == '__main__':
  app.Run(main)
