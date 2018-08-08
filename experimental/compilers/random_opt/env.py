"""A Gym environment for the LLVM optimizer."""
import filecmp
import gym
import os
import pathlib
import shutil
import subprocess
import sys
import tempfile
import typing
from absl import flags
from absl import logging
from gym import spaces
from gym.utils import seeding

from compilers.llvm import clang
from compilers.llvm import llvm_link
from compilers.llvm import opt
from experimental.compilers.random_opt.proto import random_opt_pb2
from lib.labm8 import crypto
from lib.labm8 import labdate
from lib.labm8 import pbutil


FLAGS = flags.FLAGS


class LlvmOptEnv(gym.Env):
  """A OpenAI gym environment for the LLVM optimizer."""

  def __init__(self, config: random_opt_pb2.Environment):
    """Instantiate an environment.

    Args:
      config: The Environment config proto.
    """
    self.working_dir = pathlib.Path(
        tempfile.mkdtemp(prefix='phd_llvm_opt_env_'))
    self.config = config

    # Validate the user configuration.
    for pass_name in self.config.candidate_pass:
      if pass_name not in opt.ALL_PASSES:
        raise ValueError(f"Unrecognized opt pass: '{pass_name}'")

    self.episodes = []
    self.action_space = spaces.Discrete(len(self.config.candidate_pass))
    # TODO(cec): Decide on observation space. For now we use a dummy variable.
    self.observation_space = spaces.Discrete(10)

    self.failure_reward = -10

    srcs = [pathlib.Path(x) for x in self.config.input_src]
    for src in srcs:
      if not src.is_file():
        raise ValueError(f"Environment.input_src not found: '{src}'.")

    # Produce the bytecode file.
    self.bytecode_path = self.working_dir / 'input.ll'
    self.base_bytecode_path = self.working_dir / 'base_input.ll'
    ProduceBytecodeFromSources(srcs, self.bytecode_path)
    logging.debug('$ cp %s %s', self.bytecode_path, self.base_bytecode_path)
    shutil.copyfile(self.bytecode_path, self.base_bytecode_path)

    self.binary_path = self.working_dir / 'binary'
    self.exec_cmd = self._MakeVariableSubstitution(self.config.exec_cmd)
    self.eval_cmd = None
    if self.config.HasField('eval_cmd'):
      self.eval_cmd = self._MakeVariableSubstitution(self.config.eval_cmd)

  def __del__(self):
    """Environment destructor. Clean up working dir."""
    shutil.rmtree(self.working_dir)

  def Reward(self, status: pbutil.Enum,
             speedup: typing.Optional[float]) -> float:
    """Get the reward for a step.

    Args:
      status: A Step.Status enum value.
      speedup: The speedup, if there is one.

    Returns:
      The immediate reward value.

    Raises:
      ValueError: If the status is not recognized.
    """
    if status == random_opt_pb2.Step.PASS:
      return speedup - 1
    elif status == random_opt_pb2.Step.OPT_FAILED:
      return -5
    elif status == random_opt_pb2.Step.COMPILE_FAILED:
      return -5
    elif status == random_opt_pb2.Step.EVAL_FAILED:
      return -5
    else:
      raise ValueError(f"Unrecognized Step.status value: '{status}'.")

  def _MakeVariableSubstitution(self, cmd: str) -> str:
    substitutions = {
      '$@': str(self.binary_path),
      '$<': str(self.bytecode_path),
      '@D': str(self.working_dir),
    }
    for src, dst in substitutions.items():
      cmd = cmd.replace(src, dst)
    return cmd

  def seed(self, seed=None):
    self.np_random, seed = seeding.np_random(seed)
    return [seed]

  def reset(self):
    logging.debug('$ cp %s %s', self.base_bytecode_path, self.bytecode_path)
    shutil.copyfile(self.base_bytecode_path, self.bytecode_path)
    CompileBytecodeToBinary(self.bytecode_path, self.binary_path)
    self.RunSetup()
    self.RunBinary()
    if not self.StepIsValid():
      raise ValueError(f"Failed to validate base binary.")
    start_time = labdate.MillisecondsTimestamp()
    self.episodes.append(random_opt_pb2.Episode(step=[
      random_opt_pb2.Step(
          start_time_epoch_ms=start_time,
          status=random_opt_pb2.Step.PASS,
          binary_runtime_ms=self.GetRuntimeMs(),
          reward=0,
          total_reward=0,
          speedup=1.0,
          total_speedup=1.0,
      )
    ]))
    self.episodes[-1].step[0].total_step_runtime_ms = (
        labdate.MillisecondsTimestamp() - start_time)

  def render(self, outfile=sys.stdout):
    """Render text representation of environment.

    Args:
      outfile: The text wrapper to write string representaiton to.

    Returns:
      The outfile.
    """
    bin_changed = (
      'changed' if self.episodes[-1].step[-1].binary_changed else 'unchanged')
    bytecode_changed = (
      'changed' if self.episodes[-1].step[-1].bytecode_changed else 'unchanged')
    outfile.write(f'''\
==================================================
EPISODE #{len(self.episodes)}, STEP #{len(self.episodes[-1].step) - 1}:

  Step time: {self.episodes[-1].step[-1].total_step_runtime_ms} ms.
  Passes Run: {self.episodes[-1].step[-1].opt_pass}.
  Binary {bin_changed}, bytecode {bytecode_changed}.
  Binary Runtimes: {self.episodes[-1].step[-1].binary_runtime_ms} ms.
  Reward: {self.episodes[-1].step[-1].reward:.3f} ({self.episodes[-1].step[-1].total_reward:.3f} total)
  Speedup: {self.episodes[-1].step[-1].speedup:.2f}x ({self.episodes[-1].step[-1].total_speedup:.2f}x total)
''')
    if self.episodes[-1].step[-1].status != random_opt_pb2.Step.PASS:
      last_status = random_opt_pb2.Step.Status.Name(
          self.episodes[-1].step[-1].status)
      outfile.write(f'  Status: {last_status}\n')
    outfile.write('\n')
    return outfile

  def Step(self, step: random_opt_pb2.Step) -> random_opt_pb2.Step:
    """Run a step.

    Args:
      step: A step proto with field 'opt_pass' set.

    Returns:
      The input step.
    """
    start_time = labdate.MillisecondsTimestamp()
    step.start_time_epoch_ms = start_time
    step.status = random_opt_pb2.Step.PASS
    temp_bytecode = self.working_dir / 'temp_src.ll'
    temp_binary = self.working_dir / 'temp_binary'

    # Run the pass.
    try:
      RunOptPassOnBytecode(self.bytecode_path, temp_bytecode,
                           list(step.opt_pass))
    except ValueError as e:
      step.status = random_opt_pb2.Step.OPT_FAILED
      step.status_msg = str(e)

    if step.status == random_opt_pb2.Step.PASS:
      # Update bytecode file.
      logging.debug('$ mv %s %s', temp_bytecode, self.bytecode_path)
      step.bytecode_changed = BytecodesAreEqual(temp_bytecode,
                                                self.bytecode_path)
      os.rename(str(temp_bytecode), str(self.bytecode_path))
      # Compile a new binary.
      try:
        CompileBytecodeToBinary(self.bytecode_path, temp_binary)
        step.binary_changed = BinariesAreEqual(temp_binary, self.binary_path)
        os.rename(str(temp_binary), str(self.binary_path))
      except ValueError as e:
        step.status = random_opt_pb2.Step.COMPILE_FAILED
        step.status_msg = str(e)

    if step.status == random_opt_pb2.Step.PASS:
      # Get the binary runtime.
      step.binary_runtime_ms.extend(self.GetRuntimeMs())
      if self.StepIsValid():
        step.speedup = (
            (sum(self.episodes[-1].step[-1].binary_runtime_ms) / len(
                self.episodes[-1].step[-1].binary_runtime_ms)) /
            (sum(step.binary_runtime_ms) / len(step.binary_runtime_ms)))
        step.total_speedup = (
            (sum(self.episodes[-1].step[0].binary_runtime_ms) / len(
                self.episodes[-1].step[0].binary_runtime_ms)) / (
                sum(step.binary_runtime_ms) / len(step.binary_runtime_ms)))
      else:
        step.status = random_opt_pb2.Step.EVAL_FAILED

    step.reward = self.Reward(step.status, step.speedup)
    step.total_reward = self.episodes[-1].step[-1].total_reward + step.reward
    step.total_step_runtime_ms = labdate.MillisecondsTimestamp() - start_time
    self.episodes[-1].step.extend([step])
    return step

  def StepIsValid(self, timeout_seconds: int = 60) -> bool:
    """Valid the current step. Should only be called after exec_cmd has run."""
    if self.eval_cmd:
      try:
        cmd = f"timeout -s9 {timeout_seconds} bash -c '{self.eval_cmd}'"
        logging.debug('$ %s', cmd)
        subprocess.check_call(cmd, shell=True)
        return True
      except subprocess.CalledProcessError:
        return False
    else:
      return True

  def RunSetup(self, timeout_seconds: int = 60) -> None:
    if self.config.HasField('setup_cmd'):
      cmd = self._MakeVariableSubstitution(self.config.setup_cmd)
      cmd = f"timeout -s9 {timeout_seconds} bash -c '{cmd}'"
      logging.debug('$ %s', cmd)
      subprocess.check_call(cmd, shell=True)

  def RunBinary(self, timeout_seconds: int = 60) -> None:
    """Run the binary. Requires that binary has been compiled."""
    exec_cmd = f"timeout -s9 {timeout_seconds} bash -c '{self.exec_cmd}'"
    logging.debug('$ %s', exec_cmd)
    proc = subprocess.Popen(exec_cmd, shell=True)
    proc.communicate()
    if proc.returncode == 9:
      raise ValueError(
          f"Command timed out after {timeout_seconds} seconds: '{cmd}'")
    elif proc.returncode:
      raise ValueError(
          f"Command exited with return code {proc.returncode}: '{cmd}'")

  def GetRuntimeMs(self, num_runs: int = 3) -> int:
    """Get runtime.

    Args:
      cmd: The command to execute.
      num_runs: The number of runs to execute.
      timeout_seconds: The maximum runtime of the command.

    Returns:
      The average runtime after num_runs executions, in milliseconds.
    """
    runtimes = []
    for _ in range(num_runs):
      start_ms = labdate.MillisecondsTimestamp()
      self.RunBinary()
      end_ms = labdate.MillisecondsTimestamp()
      runtimes.append(end_ms - start_ms)
    return runtimes

  def step(self, action: int):
    if not self.action_space.contains(action):
      raise ValueError(f"Unknown action: '{action}'")
    proto = self.Step(
        random_opt_pb2.Step(opt_pass=[self.config.candidate_pass[action]]))
    # TODO(cec): Calculate observation once observation space is implemented.
    obs = self.observation_space.sample()
    reward = proto.reward
    done = False if proto.status == random_opt_pb2.Step.PASS else True
    return obs, reward, done, {}

  def ToProto(self) -> random_opt_pb2.Experiment:
    """Return proto representation of environment."""
    return random_opt_pb2.Experiment(
        env=self.config,
        episode=self.episodes)


def CompileBytecodeToBinary(input_path: pathlib.Path,
                            binary_path: pathlib.Path) -> pathlib.Path:
  """Compile bytecode file to a binary.

  Args:
    input_path: The path of the input bytecode file.
    binary_path: The path of the binary to generate.

  Returns:
    The binary_path.
  """
  proc = clang.Exec([str(input_path), '-O0', '-o', str(binary_path)])
  if proc.returncode:
    raise ValueError(f'Failed to compile binary: {proc.stderr}')
  if not binary_path.is_file():
    raise ValueError(f'Binary file {binary_path} not generated.')
  return binary_path


def RunOptPassOnBytecode(input_path: pathlib.Path,
                         output_path: pathlib.Path,
                         opts: typing.List[str]) -> pathlib.Path:
  """Run opt pass on a bytecode file.

  Args:
    input_path: The input bytecode file.
    output_path: The file to generate.
    opts: Additional flags to pass to opt.

  Returns:
    The output_path.
  """
  proc = opt.Exec([str(input_path), '-o', str(output_path), '-S'] + opts)
  if proc.returncode:
    raise ValueError(f'Failed to run opt pass: {proc.stderr}')
  if not output_path.is_file():
    raise ValueError(f'Bytecode file {output_path} not generated')
  return output_path


def ProduceBitcode(
    input_src: pathlib.Path,
    output_path: pathlib.Path,
    copts: typing.Optional[typing.List[str]] = None) -> pathlib.Path:
  """Generate bitcode for a source file.

  Args:
    input_src: The input source file.
    output_path: The file to generate.
    copts: A list of additional flags to pass to clang.

  Returns:
    The output_path.
  """
  copts = copts or []
  proc = clang.Exec([str(input_src), '-o', str(output_path), '-emit-llvm',
                     '-S', '-c', '-O0'] + copts)
  if proc.returncode:
    raise ValueError(f'Failed to compile bytecode: {proc.stderr}')
  if not output_path.is_file():
    raise ValueError(f'Bytecode file {out_path} not generated.')
  return output_path


def LinkBitcodeFilesToBytecode(
    input_paths: typing.List[pathlib.Path],
    output_path: pathlib.Path,
    linkopts: typing.Optional[typing.List[str]] = None) -> pathlib.Path:
  """Link multiple bitcode files to a single bytecode file.

  Args:
    input_paths: A list of input bitcode files.
    output_path: The bytecode file to generate.
    linkopts: A list of additional flags to pass to llvm-link.

  Returns:
    The output_path.
  """
  if output_path.is_file():
    output_path.unlink()
  linkopts = linkopts or []
  proc = llvm_link.Exec(
      [str(x) for x in input_paths] + ['-o', str(output_path), '-S'] + linkopts)
  if proc.returncode:
    raise ValueError(f'Failed to link bytecode: {proc.stderr}')
  if not output_path.is_file():
    raise ValueError(f'Bytecode file {output_path} not linked.')
  return output_path


def ProduceBytecodeFromSources(
    input_paths: typing.List[pathlib.Path],
    output_path: pathlib.Path,
    copts: typing.Optional[typing.List[str]] = None,
    linkopts: typing.Optional[typing.List[str]] = None) -> pathlib.Path:
  """Produce a single bytecode file for a set of sources.

  Args:
    input_paths: A list of input source files.
    output_path: The file to generate.
    copts: A list of additional flags to pass to clang.
    linkopts: A list of additional flags to pass to llvm-link.

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
      ProduceBitcode(src, input_src, copts)
    # Link the separate bytecode files.
    LinkBitcodeFilesToBytecode(input_srcs, output_path, linkopts)
  return output_path


def BytecodesAreEqual(a: pathlib.Path, b: pathlib.Path) -> bool:
  # TODO(cec): Just ignoring the first line is not enough.
  with open(a) as f:
    a_src = '\n'.join(f.read().split('\n')[1:])
  with open(b) as f:
    b_src = '\n'.join(f.read().split('\n')[1:])
  return a_src == b_src


def BinariesAreEqual(a: pathlib.Path, b: pathlib.Path) -> bool:
  return filecmp.cmp(a, b)
