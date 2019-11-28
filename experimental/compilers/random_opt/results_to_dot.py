"""Output a tree of explored """
import pathlib
import typing

import graphviz

from experimental.compilers.random_opt.proto import random_opt_pb2
from labm8.py import app
from labm8.py import graph
from labm8.py import pbutil

FLAGS = app.FLAGS

app.DEFINE_string(
  "delayed_reward_experiment_path",
  "/tmp/phd/experimental/compilers/random_opt/random_opt.pbtxt",
  "Path to a DelayedRewardExperiment proto.",
)


def _SanitizeName(name):
  return name[1:].replace("-", "_")


def DelayedRewardExperimentToDot(data: random_opt_pb2.DelayedRewardExperiment):
  dot = graphviz.Digraph()
  dot.node("START", "START")
  for episode in data.episode:
    AddStepsToDot(dot, episode.step[1:], "START")
  return dot


def AddStepsToDot(
  dot, steps: typing.List[random_opt_pb2.DelayedRewardStep], uid: str
):
  if not steps:
    return

  name = _SanitizeName(steps[0].opt_pass) if steps[0].opt_pass else "END"
  new_uid = f"{uid}{name}"
  dot.node(new_uid, name)
  dot.edge(uid, new_uid)
  AddStepsToDot(dot, steps[1:], new_uid)


def DelayedRewardExperimentToGraph(
  data: random_opt_pb2.DelayedRewardExperiment,
) -> graph.Graph:
  root = graph.Graph("START")
  root.rewards = []
  for episode in data.episode:
    AddStepsToGraph(root, episode.step[1:])
  return root


def AddStepsToGraph(root, steps: typing.List[random_opt_pb2.DelayedRewardStep]):
  if not steps:
    return

  name = _SanitizeName(steps[0].opt_pass) if steps[0].opt_pass else "END"
  for child in root.children:
    if child.name == name:
      child.rewards.append(steps[0].reward)
      AddStepsToGraph(child, steps[1:])
      break
  else:
    new_node = graph.Graph(name)
    new_node.rewards = [steps[0].reward]
    root.children.add(new_node)
    AddStepsToGraph(new_node, steps[1:])


def main(argv: typing.List[str]):
  """Main entry point."""
  if len(argv) > 1:
    raise app.UsageError("Unknown arguments: '{}'.".format(" ".join(argv[1:])))

  path = pathlib.Path(FLAGS.delayed_reward_experiment_path)
  data = pbutil.FromFile(path, random_opt_pb2.DelayedRewardExperiment())
  # graph = DelayedRewardExperimentToGraph(data)
  # print(graph.ToDot())
  dot = DelayedRewardExperimentToDot(data)
  print(dot.source)


if __name__ == "__main__":
  app.RunWithArgs(main)
