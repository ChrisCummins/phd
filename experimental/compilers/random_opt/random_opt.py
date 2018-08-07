"""Random optimizer."""
import pathlib
import subprocess
import tempfile
import typing
from absl import app
from absl import flags
from absl import logging

from compilers.llvm import clang
from compilers.llvm import llvm_link
from datasets.benchmarks import bzip2
from experimental.compilers.random_opt.proto import random_opt_pb2
from lib.labm8 import crypto
from lib.labm8 import labdate


FLAGS = flags.FLAGS


def GetRuntime(cmd: typing.List[str],
               proto: random_opt_pb2.RandomOptStep,
               timeout_seconds: int = 5) -> None:
  for _ in range(3):
    start_ms = labdate.MillisecondsTimestamp()
    cmd = ['timeout', '-s9', str(timeout_seconds)] + cmd
    logging.debug('%s', ' '.join(cmd))
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                            universal_newlines=True)
    proc.communicate()
    if proc.returncode == 9:
      raise ValueError(f'{binary} timed out after {timeout_seconds} seconds')
    # TODO(cec): Continue from here.
    print(proc.stdout)
    print(proc.stderr)
    end_ms = labdate.MillisecondsTimestamp()
    logging.debug('%s: %d ms', cmd[0], end_ms - start_ms)
    proto.runtime_ms.extend([end_ms - start_ms])


def CompileBytecodeToBinary(input_path: pathlib.Path,
                            binary_path: pathlib.Path) -> pathlib.Path:
  proc = clang.Exec([str(input_path), '-o', str(binary_path)])
  if proc.returncode:
    raise ValueError(f'Failed to compile binary: {proc.stderr}')
  if not binary_path.is_file():
    raise ValueError(f'Binary file {out_path} not generated.')
  return binary_path


def EvaluateBytecodeFile(
    input_path: pathlib.Path,
    binary_path: pathlib.Path,
    binary_args: typing.List[str]) -> random_opt_pb2.RandomOptStep:
  proto = random_opt_pb2.RandomOptStep(
      start_time_epoch_ms=labdate.MillisecondsTimestamp()
  )
  CompileBytecodeToBinary(input_path, binary_path)
  GetRuntime([str(binary_path)] + binary_args, proto)
  return proto


def ProduceBitcode(
    input_src: pathlib.Path,
    output_path: pathlib.Path,
    cflags: typing.Optional[typing.List[str]] = None) -> pathlib.Path:
  """Generate bitcode for a source file.
  
  Args:
    input_src: The input source file. 
    output_path: The file to generate. 
  
  Returns:
    The output_path.
  """
  cflags = cflags or []
  proc = clang.Exec([str(input_src), '-o', str(output_path), '-emit-llvm',
                     '-S', '-c', '-O0'] + cflags)
  if proc.returncode:
    raise ValueError(f'Failed to compile bytecode: {proc.stderr}')
  if not output_path.is_file():
    raise ValueError(f'Bytecode file {out_path} not generated.')
  return output_path


def LinkBitcodeFilesToBytecode(input_paths: typing.List[pathlib.Path],
                               output_path: pathlib.Path) -> pathlib.Path:
  """Link multiple bitcode files to a single bytecode file.

  Args:
    input_paths: A list of input bitcode files.
    output_path: The bytecode file to generate.

  Returns:
    The output_path.
  """
  proc = llvm_link.Exec(
      [str(x) for x in input_paths] + ['-o', str(output_path), '-S'])
  if proc.returncode:
    raise ValueError(f'Failed to link bytecode: {proc.stderr}')
  if not output_path.is_file():
    raise ValueError(f'Bytecode file {output_path} not linked.')
  return output_path


def ProduceBytecodeFromSources(
    input_paths: typing.List[pathlib.Path],
    output_path: pathlib.Path,
    cflags: typing.Optional[typing.List[str]] = None) -> pathlib.Path:
  """Produce a single bytecode file for a set of sources.

  Args:
    input_paths: A list of input source files.
    output_path: The file to generate.

  Returns:
    The output_path.
  """
  if output_path.is_file():
    output_path.unlink()

  # Compile each input source to a bytecode file.
  with tempfile.TemporaryDirectory() as d:
    d = pathlib.Path(d)
    input_srcs = [
      d / (crypto.sha256_str(str(src)) + '.l') for src in input_paths]
    for src, input_src in zip(input_paths, input_srcs):
      ProduceBitcode(src, input_src, cflags)
    # Link the separate bytecode files.
    LinkBitcodeFilesToBytecode(input_srcs, output_path)
  return output_path


def RandomOpt(
    experiment: random_opt_pb2.RandomOptExperiment
) -> random_opt_pb2.RandomOptExperiment:
  """Run a random optimization experiment.

  Args:
    experiment: The experimental parameters.

  Returns:
    The experiment proto, with 'results' field set.
  """
  logging.debug('Beginning experiment:\n%s', str(experiment))
  srcs = [pathlib.Path(x) for x in experiment.input_src]
  for src in srcs:
    if not src.is_file():
      raise ValueError(f"RandomOptExperiment.input_src not found: '{src}'.")

  with tempfile.TemporaryDirectory(prefix='random_opt_') as d:
    d = pathlib.Path(d)
    bytecode_path = d / 'src.ll'
    binary_path = d / 'binary'
    ProduceBytecodeFromSources(srcs, bytecode_path)
    proto = EvaluateBytecodeFile(bytecode_path, binary_path,
                                 list(experiment.binary_arg))
    print(proto)
    for _ in range(experiment.max_steps):
      pass


def main(argv: typing.List[str]):
  """Main entry point."""
  if len(argv) > 1:
    raise app.UsageError("Unknown arguments: '{}'.".format(' '.join(argv[1:])))

  experiment = RandomOpt(random_opt_pb2.RandomOptExperiment(
      input_src=[str(x) for x in bzip2.BZIP2_SRCS],
      binary_arg=[],
      num_steps=10,
  ))
  print(experiment)
  logging.info('done')


if __name__ == '__main__':
  app.run(main)
