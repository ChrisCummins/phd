"""Random optimizer."""
import gym
import pathlib
import typing
from absl import app
from absl import flags
from absl import logging
from gym.envs import registration

from compilers.llvm import opt
from datasets.benchmarks import bzip2
from experimental.compilers.random_opt.proto import random_opt_pb2
from lib.labm8 import pbutil


FLAGS = flags.FLAGS

flags.DEFINE_string(
    'env', 'LLVM-bzip2-512K-v0',
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

# Register the LLVM environments for use in gym.make().

registration.register(
    id='LLVM-bzip2-512K-v0',
    entry_point='phd.experimental.compilers.random_opt.env:LlvmOptEnv',
    kwargs={
      'config': random_opt_pb2.Environment(
          input_src=[str(x) for x in bzip2.BZIP2_SRCS],
          setup_cmd='head -c 512K </dev/urandom > @D/input.dat',
          exec_cmd=(f'$@ -z < @D/input.dat > @D/input.dat.bz2 && '
                    f'$@ -d < @D/input.dat.bz2 > @D/output.dat'),
          eval_cmd=f'cmp --silent @D/input.dat @D/output.dat',
          candidate_pass=list(opt.ALL_PASSES),
      )
    },
)


def Render(env: gym.Env) -> None:
  if FLAGS.render:
    env.render()


def main(argv: typing.List[str]):
  """Main entry point."""
  if len(argv) > 1:
    raise app.UsageError("Unknown arguments: '{}'.".format(' '.join(argv[1:])))

  logging.info('Generating environment %s ...', FLAGS.env)
  env = gym.make(FLAGS.env)
  logging.info('Starting %d random walk episodes of %d steps each ...',
               FLAGS.num_episodes, FLAGS.max_steps)

  for i in range(FLAGS.num_episodes):
    logging.info('EPISODE %d:', i + 1)
    env.reset()
    Render(env)
    for _ in range(FLAGS.max_steps):
      # We don't yet have an observation space, so all we can do is take random
      # choices.
      obs, reward, done, _ = env.step(env.action_space.sample())
      Render(env)
      if done:
        break

  out_path = pathlib.Path(FLAGS.proto_out)
  out_path.parent.mkdir(parents=True, exist_ok=True)
  pbutil.ToFile(env.ToProto(), out_path)
  logging.info('Wrote experimental results to: %s', out_path)
  logging.info('Done.')


if __name__ == '__main__':
  app.run(main)
