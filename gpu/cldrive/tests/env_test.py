from absl import app

from gpu.cldrive.tests.testlib import *


def test_make_env_not_found():
  with pytest.raises(LookupError):
    cldrive.make_env(platform="not a real plat",
                     device="not a real dev")


def test_all_envs():
  min_num_devices = 0

  if cldrive.has_cpu():
    min_num_devices += 1
  if cldrive.has_gpu():
    min_num_devices += 1

  envs = list(cldrive.all_envs())
  assert len(envs) >= min_num_devices


@needs_cpu
def test_make_env_cpu():
  env = cldrive.make_env(devtype="cpu")
  assert env.device_type == "CPU"


@needs_gpu
def test_make_env_gpu():
  env = cldrive.make_env(devtype="gpu")
  assert env.device_type == "GPU"


def main(argv):  # pylint: disable=missing-docstring
  del argv
  sys.exit(pytest.main(
    [cldrive.env.__file__, __file__, "-v", "--doctest-modules"]))


if __name__ == "__main__":
  app.run(main)
