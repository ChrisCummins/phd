import pathlib
import socket

from absl import app
from absl import flags

from lib.labm8 import pbutil

FLAGS = flags.FLAGS


def AssertLocalServiceHostname(service_config: pbutil.ProtocolBuffer):
  hostname = socket.gethostname()
  service_hostname = service_config.generator.service_hostname
  if (service_hostname != 'localhost' and
      service_hostname != hostname):
    raise app.UsageError(
      f'System hostname {hostname} does not match service hostname '
      f'{service_hostname}')


def ServiceConfigFromFlag(
    flag_name: str,
    service_config: pbutil.ProtocolBuffer) -> pbutil.ProtocolBuffer:
  if not getattr(FLAGS, flag_name):
    raise app.UsageError(f'--{flag_name} not set.')
  generator_config_path = pathlib.Path(FLAGS.generator_config)
  if not pbutil.ProtoIsReadable(
      generator_config_path, service_config):
    cls_name = type(service_config).__name__
    raise app.UsageError(f'--{flag_name} must be a {cls_name} proto.')

  return pbutil.FromFile(
    generator_config_path, service_config)
