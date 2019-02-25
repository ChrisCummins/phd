"""Unit tests for //system/machines:machine.py."""
import tempfile

import pytest
from absl import flags

from labm8 import test
from system.machines import machine
from system.machines.proto import machine_spec_pb2

FLAGS = flags.FLAGS


@pytest.fixture(scope='function')
def test_machine() -> machine.Machine:
  """A test fixture that returns a Machine instance."""
  with tempfile.TemporaryDirectory(prefix='tmp_system_machines_') as d:
    yield machine.Machine.FromProto(
        machine_spec_pb2.MachineSpec(
            name="test",
            host=[
                machine_spec_pb2.Host(host='localhost', port=22),
            ],
            mirrored_directory=[
                machine_spec_pb2.MirroredDirectory(
                    name='foo', local_path=d, remote_path='notused')
            ]))


def test_host(test_machine: machine.Machine):
  """Test host attribute."""
  assert test_machine.host == machine_spec_pb2.Host(host='localhost', port=22)


def test_MirroredDirectory_found(test_machine: machine.Machine):
  """Test that a mirrored directory can be found."""
  d = test_machine.MirroredDirectory('foo')
  assert d.LocalExists()


def test_MirroredDirectory_not_found(test_machine: machine.Machine):
  """Test that a mirrored directory cannot be found."""
  with pytest.raises(LookupError) as e_ctx:
    test_machine.MirroredDirectory('not_found')
  assert str(e_ctx.value) == "Cannot find mirrored directory 'not_found'"


if __name__ == '__main__':
  test.Main()
