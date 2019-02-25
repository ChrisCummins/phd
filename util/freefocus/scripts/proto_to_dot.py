"""Convert a protobuf format workspace to dot source."""
import sys
from argparse import ArgumentParser

from google.protobuf import json_format
from graphviz import Digraph

from util.freefocus import freefocus_pb2


def escape(string):
  return string.replace(":", "/")


def plot_groups(dot, parent_id, nodes):

  def plot_group_members(parent_id, members):
    for member in members:
      if member.type == freefocus_pb2.Group.Member.GROUP:
        dot.node(escape(member.group.id, escape(member.group.id)))
        dot.edge(escape(parent_id), escape(member.group.id))

        plot_group_members(member.id, member.members)
      else:
        dot.node(escape(member.person.uid), escape(member.person.uid))
        dot.edge(escape(parent_id), escape(member.person.uid))

  for node in nodes:
    dot.node(escape(node.id), escape(node.id))
    dot.edge(escape(parent_id), escape(node.id))

    if len(node.members):
      plot_group_members(node.id, node.members)


def plot_tags(dot, parent_id, nodes):
  for node in nodes:
    dot.node(escape(node.id), escape(node.id))
    dot.edge(escape(parent_id), escape(node.id))

    if len(node.children):
      plot_tags(dot, node.id, node.children)


def plot_comments(dot, parent_id, nodes):
  for node in nodes:
    dot.node(escape(node.id), escape(node.id))
    dot.edge(escape(parent_id), escape(node.id))

    if len(node.comments):
      plot_comments(dot, node.id, node.comments)


def plot_tasks(dot, parent_id, tasks):
  for task in tasks:
    dot.node(escape(task.id), escape(task.id))
    dot.edge(escape(parent_id), escape(task.id))

    if len(task.children):
      plot_tasks(dot, task.id, task.children)

    if len(task.comments):
      plot_comments(dot, task.id, task.comments)

    for tag in task.tags:
      dot.edge(escape(tag.id), escape(task.id))


if __name__ == "__main__":
  parser = ArgumentParser(description=__doc__)
  args = parser.parse_args()

  workspace = freefocus_pb2.Workspace()
  json_format.Parse(sys.stdin.read(), workspace)

  dot = Digraph(comment=workspace.uid)
  dot.graph_attr['rankdir'] = 'LR'

  dot.node(escape(workspace.uid), escape(workspace.uid))

  plot_groups(dot, workspace.uid, workspace.groups)
  plot_tags(dot, workspace.uid, workspace.tags)
  plot_tasks(dot, workspace.uid, workspace.tasks)

  print(dot.source)
