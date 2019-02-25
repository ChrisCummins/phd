"""A Gym environment for the LLVM optimizer."""
import filecmp
import os
import pathlib
import shutil
import subprocess
import sys
import tempfile
import time
import typing

import gym
import numpy as np
from absl import flags
from absl import logging
from gym import spaces
from gym.utils import seeding

from compilers.llvm import clang
from compilers.llvm import llvm
from compilers.llvm import llvm_link
from compilers.llvm import opt
from experimental.compilers.random_opt.proto import random_opt_pb2
from labm8 import crypto
from labm8 import jsonutil
from labm8 import labdate
from labm8 import pbutil
from labm8 import text

# WARNING: No flags can be defined in this file, because it is loaded at runtime
# by gym to resolve string entry points.
FLAGS = flags.FLAGS


class Environment(gym.Env):
  """Base class which defines the interface for LLVM opt environments."""
  step_t = typing.Tuple[np.ndarray, int, bool, jsonutil.JSON]

  def __init__(self, config: random_opt_pb2.Environment):
    """Instantiate an environment.

    Args:
      config: An Environment config proto.
    """
    super(Environment, self).__init__()

    self.config = config

    # Validate the requested candidate passes and set as the action space.
    for pass_name in self.config.candidate_pass:
      if pass_name not in opt.ALL_PASSES:
        raise ValueError(f"Unrecognized opt pass: '{pass_name}'")

    # Inheriting classes are required to provide their own action and
    # observation spaces.
    self.action_space = None
    self.observation_space = None

    # A working directory that an inheriting class can use as a scratch space
    # for reading and writing files. Destroyed during __del__ operator.
    self.working_dir = pathlib.Path(
        tempfile.mkdtemp(prefix='phd_llvm_opt_env_'))

    # A list of episode protocol buffers, ordered from first to most recent.
    self.episodes: typing.List[pbutil.ProtocolBuffer] = []

    # Validate the input sources.
    srcs = [pathlib.Path(x) for x in self.config.input_src]
    for src in srcs:
      if not src.is_file():
        raise ValueError(f"Environment.input_src not found: '{src}'.")

    # Produce the bytecode file.
    self.bytecode_path = self.working_dir / 'base_input.ll'
    self.working_bytecode_path = self.working_dir / 'working_input.ll'
    ProduceBytecodeFromSources(srcs, self.bytecode_path)

    self.binary_path = self.working_dir / 'binary'
    self.exec_cmd = self._MakeVariableSubstitution(self.config.exec_cmd)
    self.eval_cmd = None
    if self.config.HasField('eval_cmd'):
      self.eval_cmd = self._MakeVariableSubstitution(self.config.eval_cmd)

  def reset(self):
    """Reset the environment."""
    raise NotImplementedError

  def step(self, action: int) -> step_t:
    """Take a step using the given action and return a step_t tuple."""
    raise NotImplementedError

  def render(self, outfile=sys.stdout):
    """Render text representation of environment.

    Args:
      outfile: The text wrapper to write string representaiton to.

    Returns:
      The outfile.
    """
    raise NotImplementedError

  def RunSetupCommand(self, timeout_seconds: int = 60) -> None:
    if self.config.HasField('setup_cmd'):
      cmd = self._MakeVariableSubstitution(self.config.setup_cmd)
      cmd = f"timeout -s9 {timeout_seconds} bash -c '{cmd}'"
      logging.debug('$ %s', cmd)
      subprocess.check_call(cmd, shell=True)

  def RunBinary(self, timeout_seconds: int = 60) -> int:
    """Run the binary and return runtime. Requires that binary exists."""
    exec_cmd = f"timeout -s9 {timeout_seconds} bash -c '{self.exec_cmd}'"
    logging.debug('$ %s', exec_cmd)
    start_time = time.time()
    proc = subprocess.Popen(exec_cmd, shell=True)
    proc.communicate()
    end_time = time.time()
    if proc.returncode == 9:
      raise ValueError(f"Command timed out after {timeout_seconds} seconds: "
                       f"'{self.exec_cmd}'")
    elif proc.returncode:
      raise ValueError(f"Command exited with return code {proc.returncode}: "
                       f"'{self.exec_cmd}'")
    return int(round((end_time - start_time) * 1000))

  def GetRuntimes(self) -> typing.List[int]:
    """Get runtimes of binary.

    Returns:
      The runtime of the binary, in milliseconds.
    """
    return [self.RunBinary() for _ in range(FLAGS.runtime_num_runs)]

  def BinaryIsValid(self, timeout_seconds: int = 60) -> bool:
    """Validate the current step. Must be called after exec_cmd has run."""
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

  def ToProto(self) -> pbutil.ProtocolBuffer:
    """Return proto representation of environment."""
    raise NotImplementedError

  def seed(self, seed=None):
    """Re-seed the environment."""
    self.np_random, seed = seeding.np_random(seed)
    return [seed]

  def __del__(self):
    """Environment destructor. Clean up working dir."""
    shutil.rmtree(self.working_dir)

  def _MakeVariableSubstitution(self, cmd: str) -> str:
    """Perform make variable substitution on the given string."""
    substitutions = {
        '$@': str(self.binary_path),
        '$<': str(self.working_bytecode_path),
        '@D': str(self.working_dir),
    }
    for src, dst in substitutions.items():
      cmd = cmd.replace(src, dst)
    return cmd


class LlvmOptEnv(Environment):
  """A OpenAI gym environment for iterative learning of the LLVM optimizer.

  At each step, the requested pass is run on the bytecode, and is compiled to a
  binary. The difference in the runtime between the binary after the pass and
  before the pass determines the reward. This allows for immediate detection of
  actions which break the build or the binary, but requires O(n) compile-eval
  runs of the binary, where n is the number of steps in an episode.
  """

  def __init__(self, config: random_opt_pb2.Environment):
    """Instantiate an environment.

    Args:
      config: An Environment config proto.
    """
    super(LlvmOptEnv, self).__init__(config)
    self.action_space = spaces.Discrete(len(self.config.candidate_pass))
    self.observation_space = spaces.Discrete(10)

  def reset(self):
    logging.debug('$ cp %s %s', self.bytecode_path, self.working_bytecode_path)
    shutil.copyfile(self.bytecode_path, self.working_bytecode_path)
    clang.Compile([self.working_bytecode_path], self.binary_path, copts=['-O0'])
    start_time = labdate.MillisecondsTimestamp()
    self.RunSetupCommand()
    self.RunBinary()
    if not self.BinaryIsValid():
      raise ValueError(f"Failed to validate base binary.")
    self.episodes.append(
        random_opt_pb2.Episode(step=[
            random_opt_pb2.Step(
                start_time_epoch_ms=start_time,
                status=random_opt_pb2.Step.PASS,
                binary_runtime_ms=self.GetRuntimes(),
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
    bin_changed = ('changed' if self.episodes[-1].step[-1].binary_changed else
                   'unchanged')
    bytecode_changed = ('changed' if self.episodes[-1].step[-1].bytecode_changed
                        else 'unchanged')
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
      opt.RunOptPassOnBytecode(self.working_bytecode_path, temp_bytecode,
                               list(step.opt_pass))
    except llvm.LlvmError as e:
      step.status = random_opt_pb2.Step.OPT_FAILED
      step.status_msg = text.truncate(str(e), 255)

    if step.status == random_opt_pb2.Step.PASS:
      # Update bytecode file.
      logging.debug('$ mv %s %s', temp_bytecode, self.working_bytecode_path)
      step.bytecode_changed = BytecodesAreEqual(temp_bytecode,
                                                self.working_bytecode_path)
      os.rename(str(temp_bytecode), str(self.working_bytecode_path))
      # Compile a new binary.
      try:
        clang.Compile([self.working_bytecode_path], temp_binary, copts=['-O0'])
        step.binary_changed = BinariesAreEqual(temp_binary, self.binary_path)
        os.rename(str(temp_binary), str(self.binary_path))
      except llvm.LlvmError as e:
        step.status = random_opt_pb2.Step.COMPILE_FAILED
        step.status_msg = text.truncate(str(e), 255)

    if step.status == random_opt_pb2.Step.PASS:
      # Get the binary runtime.
      try:
        step.binary_runtime_ms.extend(self.GetRuntimes())
      except ValueError as e:
        step.status = random_opt_pb2.Step.EXEC_FAILED
        step.status_msg = text.truncate(str(e), 255)

    if step.status == random_opt_pb2.Step.PASS:
      if self.BinaryIsValid():
        step.speedup = (
            (sum(self.episodes[-1].step[-1].binary_runtime_ms) / len(
                self.episodes[-1].step[-1].binary_runtime_ms)) /
            (sum(step.binary_runtime_ms) / len(step.binary_runtime_ms)))
        step.total_speedup = (
            (sum(self.episodes[-1].step[0].binary_runtime_ms) / len(
                self.episodes[-1].step[0].binary_runtime_ms)) /
            (sum(step.binary_runtime_ms) / len(step.binary_runtime_ms)))
      else:
        step.status = random_opt_pb2.Step.EVAL_FAILED

    step.reward = self.Reward(step.status, step.speedup)
    step.total_reward = self.episodes[-1].step[-1].total_reward + step.reward
    step.total_step_runtime_ms = labdate.MillisecondsTimestamp() - start_time
    return step

  def step(self, action: int) -> Environment.step_t:
    """Perform the given action and return a step_t."""
    if not self.action_space.contains(action):
      raise ValueError(f"Unknown action: '{action}'")
    proto = self.Step(
        random_opt_pb2.Step(opt_pass=[self.config.candidate_pass[action]]))
    self.episodes[-1].step.extend([proto])
    # TODO(cec): Calculate observation once observation space is implemented.
    obs = self.observation_space.sample()
    reward = proto.reward
    done = False if proto.status == random_opt_pb2.Step.PASS else True
    return obs, reward, done, {}

  def ToProto(self) -> random_opt_pb2.Experiment:
    """Return proto representation of environment."""
    return random_opt_pb2.Experiment(env=self.config, episode=self.episodes)

  @staticmethod
  def Reward(status: pbutil.Enum, speedup: typing.Optional[float]) -> float:
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
    elif status == random_opt_pb2.Step.EXEC_FAILED:
      return -5
    elif status == random_opt_pb2.Step.EVAL_FAILED:
      return -5
    else:
      raise ValueError(f"Unrecognized Step.status value: '{status}'.")


class LlvmOptDelayedRewardEnv(LlvmOptEnv):
  """A delayed reward variation of the LLVM opt environment.

  Unlike the LlvmOptEnv, each step does not produce an intermediate compile-eval
  step. Instead, a binary is only produced at the end of the episode, at which
  point a single opt pass applies the full sequence of actions in one go and the
  resulting binary is executed. Reward is 0 - runtime.
  """

  def __init__(self, config: random_opt_pb2.Environment):
    super(LlvmOptDelayedRewardEnv, self).__init__(config)

    # An action is either an index into the candidate pass list, or a request to
    # end the episode. The end-of-episode action is the integer
    # len(candidate_pass).
    self.action_space = spaces.Discrete(len(self.config.candidate_pass) + 1)
    self.observation_space = spaces.Discrete(10)

    # Reward constants and functions.
    self.runtime_reward = lambda x: -int(round(x))
    self.bytecode_changed_reward = 0
    self.bytecode_unchanged_reward = -1
    self.opt_failed_reward = -5000
    self.exec_failed_reward = -5000
    self.eval_failed_reward = -5000
    self.compile_failed_reward = -5000

    # TODO(cec): Compile the binary and sanity check that it passes eval.

  def reset(self):
    """Reset the environment state."""
    logging.debug('$ cp %s %s', self.bytecode_path, self.working_bytecode_path)
    shutil.copyfile(self.bytecode_path, self.working_bytecode_path)
    clang.Compile([self.bytecode_path], self.binary_path, copts=['-O0'])
    self.RunSetupCommand()
    self.episodes.append(
        random_opt_pb2.DelayedRewardEpisode(step=[
            random_opt_pb2.DelayedRewardStep(
                start_time_epoch_ms=labdate.MillisecondsTimestamp(),)
        ]))

  def step(self, action: int) -> Environment.step_t:
    """Perform the given action and return a step_t."""
    if action < len(self.config.candidate_pass):
      return self.ActionStep(action)
    else:
      return self.EndEpisodeStep()

  def ActionStep(self, action: int) -> random_opt_pb2.DelayedRewardStep:
    if not self.action_space.contains(action):
      raise ValueError(f"Unknown action: '{action}'")
    start_ms = labdate.MillisecondsTimestamp()
    obs = self.observation_space.sample()
    opt_pass = self.config.candidate_pass[action]

    step = random_opt_pb2.DelayedRewardStep(
        start_time_epoch_ms=start_ms,
        opt_pass=opt_pass,
    )

    # Run the full list of passes and update working_bytecode file.
    try:
      all_passes = [step.opt_pass for step in self.episodes[-1].step[1:]]
      opt.RunOptPassOnBytecode(self.bytecode_path, self.working_dir / 'temp.ll',
                               all_passes)
      step.bytecode_changed = BytecodesAreEqual(self.working_dir / 'temp.ll',
                                                self.working_bytecode_path)
      shutil.copyfile(self.working_dir / 'temp.ll', self.working_bytecode_path)
      step.reward = (self.bytecode_changed_reward if step.bytecode_changed else
                     self.bytecode_unchanged_reward)
    except llvm.LlvmError as e:
      # Opt failed, set the error message.
      step.reward = self.opt_failed_reward
      step.opt_error_msg = text.truncate(str(e), 255)

    step.total_step_runtime_ms = labdate.MillisecondsTimestamp() - start_ms
    self.episodes[-1].step.extend([step])
    return obs, step.reward, False, {}

  def EndEpisodeStep(self) -> random_opt_pb2.DelayedRewardStep:
    start_ms = labdate.MillisecondsTimestamp()
    step = random_opt_pb2.DelayedRewardStep(start_time_epoch_ms=start_ms,)
    try:
      clang.Compile([self.working_bytecode_path],
                    self.binary_path,
                    copts=['-O0'])
      try:
        runtimes = self.GetRuntimes()
        self.episodes[-1].binary_runtime_ms.extend(runtimes)
        if self.BinaryIsValid():
          step.reward = self.runtime_reward(sum(runtimes) / len(runtimes))
        else:
          self.episodes[-1].outcome = (
              random_opt_pb2.DelayedRewardEpisode.EVAL_FAILED)
          step.reward = self.eval_failed_reward
      except ValueError as e:
        self.episodes[-1].outcome = random_opt_pb2.Step.EXEC_FAILED
        self.episodes[-1].outcome_error_msg = text.truncate(str(e), 255)
        step.reward = self.exec_failed_reward
    except clang.ClangException as e:
      self.episodes[-1].outcome = (
          random_opt_pb2.DelayedRewardEpisode.COMPILE_FAILED)
      self.episodes[-1].outcome_error_msg = text.truncate(str(e), 255)
      step.reward = self.compile_failed_reward

    obs = self.observation_space.sample()
    step.total_step_runtime_ms = labdate.MillisecondsTimestamp() - start_ms
    self.episodes[-1].step.extend([step])
    return obs, step.reward, True, {}

  def ToProto(self) -> random_opt_pb2.DelayedRewardExperiment:
    """Return proto representation of environment."""
    return random_opt_pb2.DelayedRewardExperiment(
        env=self.config, episode=self.episodes)

  def render(self, outfile=sys.stdout):
    """Render text representation of environment."""
    episode, step = self.episodes[-1], self.episodes[-1].step[-1]
    bytecode_changed = ('changed' if step.bytecode_changed else 'unchanged')
    outfile.write(f'''\
==================================================
EPISODE #{len(self.episodes)}, STEP #{len(episode.step) - 1}:

  Step time: {step.total_step_runtime_ms} ms.
  Opt pass: {step.opt_pass} (bytecode {bytecode_changed}).
  Reward: {step.reward}.
''')
    if step.opt_error_msg:
      outfile.write(f'  Opt error: {step.opt_error_msg}\n')
    if len(episode.step) and not step.opt_pass:
      outcome = random_opt_pb2.DelayedRewardEpisode.Outcome.Name(
          episode.outcome)
      total_reward = sum(step.reward for step in episode.step)
      outfile.write(f'''\
  Runtimes: {episode.binary_runtime_ms}
  Outcome: {outcome}
  Total reward: {total_reward}
''')
    outfile.write('\n')
    return outfile


def ProduceBytecodeFromSources(
    input_paths: typing.List[pathlib.Path],
    output_path: pathlib.Path,
    copts: typing.Optional[typing.List[str]] = None,
    linkopts: typing.Optional[typing.List[str]] = None,
    timeout_seconds: int = 60) -> pathlib.Path:
  """Produce a single bytecode file for a set of sources.

  Args:
    input_paths: A list of input source files.
    output_path: The file to generate.
    copts: A list of additional flags to pass to clang.
    linkopts: A list of additional flags to pass to llvm-link.
    timeout_seconds: The number of seconds to allow clang to run for.

  Returns:
    The output_path.
  """
  copts = copts or []
  if output_path.is_file():
    output_path.unlink()

  # Compile each input source to a bytecode file.
  with tempfile.TemporaryDirectory() as d:
    d = pathlib.Path(d)
    input_srcs = [
        d / (crypto.sha256_str(str(src)) + '.l') for src in input_paths
    ]
    for src, input_src in zip(input_paths, input_srcs):
      clang.Compile([src],
                    input_src,
                    copts=copts + ['-O0', '-emit-llvm', '-S', '-c'],
                    timeout_seconds=timeout_seconds)
    # Link the separate bytecode files.
    llvm_link.LinkBitcodeFilesToBytecode(
        input_srcs, output_path, linkopts, timeout_seconds=timeout_seconds)
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
