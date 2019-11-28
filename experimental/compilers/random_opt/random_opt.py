"""Random optimizer."""
import pathlib
import typing

import gym

from experimental.compilers.random_opt import environments
from experimental.compilers.random_opt import implementation as implementation
from labm8.py import app
from labm8.py import pbutil

FLAGS = app.FLAGS

app.DEFINE_string(
  "env", environments.DEFAULT_ENV_ID, "The name of the environment to use."
)
app.DEFINE_integer("num_episodes", 10, "The number of episodes to run for.")
app.DEFINE_integer("max_steps", 200, "The maximum number of steps per episode.")
app.DEFINE_boolean("render", True, "Render the environment after every step.")
app.DEFINE_string(
  "proto_out",
  "/tmp/phd/experimental/compilers/random_opt/random_opt.pbtxt",
  "The output path to write experiment proto to.",
)


def Render(env: gym.Env) -> None:
  """Render environment if --render is set."""
  if FLAGS.render:
    env.render()


def ToFile(env: implementation.Environment) -> None:
  """Save environment to file --proto_out."""
  out_path = pathlib.Path(FLAGS.proto_out)
  out_path.parent.mkdir(parents=True, exist_ok=True)
  pbutil.ToFile(env.ToProto(), out_path)
  app.Log(1, "Wrote experimental results to: %s", out_path)


def main(argv: typing.List[str]):
  """Main entry point."""
  if len(argv) > 1:
    raise app.UsageError("Unknown arguments: '{}'.".format(" ".join(argv[1:])))

  app.Log(1, "Generating environment %s ...", FLAGS.env)
  env = gym.make(FLAGS.env)
  app.Log(
    1,
    "Starting %d random walk episodes of %d steps each ...",
    FLAGS.num_episodes,
    FLAGS.max_steps,
  )

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
    else:
      # If we didn't naturally end the episode, explicitly stop it.
      env.step(len(env.config.candidate_pass))
      Render(env)

  ToFile(env)
  app.Log(1, "Done.")


if __name__ == "__main__":
  app.RunWithArgs(main)
