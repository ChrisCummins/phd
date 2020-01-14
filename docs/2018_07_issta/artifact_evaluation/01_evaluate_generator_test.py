"""Unit tests for :01_evaluate_generator.py."""
import pathlib
import subprocess

from deeplearning.deepsmith.proto import deepsmith_pb2
from deeplearning.deepsmith.proto import generator_pb2
from labm8.py import bazelutil
from labm8.py import pbutil
from labm8.py import test

BIN = bazelutil.DataPath(
  "phd/docs/2018_07_issta/artifact_evaluation/01_evaluate_generator"
)

pytest_plugins = ["deeplearning.clgen.tests.fixtures"]


def test_GenerateTestcases(abc_instance_config, tempdir: pathlib.Path):
  """Run a tiny end-to-end test."""
  generator_config = generator_pb2.ClgenGenerator(
    instance=abc_instance_config,
    testcase_skeleton=[
      deepsmith_pb2.Testcase(
        toolchain="opencl",
        harness=deepsmith_pb2.Harness(
          name="cldrive", opts={"timeout_seconds": "60"}
        ),
        inputs={"gsize": "1,1,1", "lsize": "1,1,1",},
      )
    ],
  )
  generator_path = tempdir / "generator.pbtxt"
  pbutil.ToFile(generator_config, generator_path)

  output_dir = tempdir / "outputs"

  subprocess.check_call(
    [
      str(BIN),
      "--generator",
      generator_path,
      "--output_directory",
      str(output_dir),
      "--num_testcases",
      str(3),
    ]
  )

  assert len(list((output_dir / "generated_testcases").iterdir())) >= 3
  assert len(list((output_dir / "generated_kernels").iterdir())) >= 3
  for f in (output_dir / "generated_testcases").iterdir():
    assert pbutil.ProtoIsReadable(f, deepsmith_pb2.Testcase())


if __name__ == "__main__":
  test.Main()
