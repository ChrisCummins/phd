"""Random optimizer."""
import gym
import pathlib
import typing
from absl import app
from absl import flags
from absl import logging

from experimental.compilers.random_opt import env as llvm_env
from experimental.compilers.random_opt import envs
from lib.labm8 import pbutil


FLAGS = flags.FLAGS

flags.DEFINE_string(
    'env', envs.DEFAULT_ENV_ID,
    'The name of the environment to use.')
flags.DEFINE_integer(
    'num_episodes', 3,
    'The number of episodes to run for.')
flags.DEFINE_integer(
    'max_steps', 100,
    'The maximum number of steps per episode.')
flags.DEFINE_boolean(
    'render', True,
    'Render the environment after every step.')
flags.DEFINE_string(
    'proto_out', '/tmp/phd/experimental/compilers/random_opt/random_opt.pbtxt',
    'The output path to write experiment proto to.')


def Render(env: gym.Env) -> None:
  """Render environment if --render is set."""
  if FLAGS.render:
    env.render()


def ToFile(env: llvm_env.LlvmOptEnv) -> None:
  """Save environment to file --proto_out."""
  out_path = pathlib.Path(FLAGS.proto_out)
  out_path.parent.mkdir(parents=True, exist_ok=True)
  pbutil.ToFile(env.ToProto(), out_path)
  logging.info('Wrote experimental results to: %s', out_path)


def main(argv: typing.List[str]):
  """Main entry point."""
  if len(argv) > 1:
    raise app.UsageError("Unknown arguments: '{}'.".format(' '.join(argv[1:])))

  logging.info('Generating environment %s ...', FLAGS.env)
  env = gym.make(FLAGS.env)
  logging.info('Starting %d random walk episodes of %d steps each ...',
               FLAGS.num_episodes, FLAGS.max_steps)

  for i in range(FLAGS.num_episodes):
    env.reset()
    Render(env)
    for _ in range(FLAGS.max_steps):
      # We don't yet have an observation space, so all we can do is take random
      # choices.
      obs, reward, done, _ = env.step(env.action_space.sample())
      Render(env)
      if done:
        break

  ToFile(env)
  logging.info('Done.')


if __name__ == '__main__':
  app.run(main)
