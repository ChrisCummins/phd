"""Convert a GitHub issue tracker into FreeFocus Protobuf messages."""
import calendar
import os
import sys
import time
from argparse import ArgumentParser
from configparser import ConfigParser

from github import Github
from google.protobuf.json_format import MessageToJson

from util.freefocus import freefocus_pb2


def now():
  return calendar.timegm(time.gmtime())


if __name__ == "__main__":
  parser = ArgumentParser(description=__doc__)
  parser.add_argument("owner")
  parser.add_argument("repo")
  parser.add_argument("--githubrc", default="~/.githubrc")
  args = parser.parse_args()

  config = ConfigParser()
  config.read(os.path.expanduser(args.githubrc))

  try:
    github_username = config['User']['Username']
    github_pw = config['User']['Password']
  except KeyError as e:
    print(
        f'config variable {e} not set. Check {args.githubrc}', file=sys.stderr)
    sys.exit(1)

  g = Github(github_username, github_pw)
  r = g.get_repo(f"{args.owner}/{args.repo}")

  # create person
  user = g.get_user(args.owner)
  p = freefocus_pb2.Person(uid=args.owner, name=user.name)

  workspace = freefocus_pb2.Workspace(
      uid="GitHub",
      created=calendar.timegm(time.strptime("1/4/08", "%d/%m/%y")))

  group = workspace.groups.add()
  group.id = f"@groups//{user.name}"

  groupmember = group.members.add()
  groupmember.type = freefocus_pb2.Group.Member.PERSON
  groupmember.person.uid = p.uid

  owner_subtree = workspace.tasks.add()
  owner_subtree.id = f"//:{args.owner}"

  owner = owner_subtree.owners.add()
  owner.id = group.id

  # owner_subtree.parent.type = freefocus_pb2.Task.Parent.WORKSPACE
  # owner_subtree.parent.workspace.uid = workspace.uid

  repo_subtree = owner_subtree.children.add()
  repo_subtree.id = f"//{args.owner}:{args.repo}"

  # repo_subtree.parent.type = freefocus_pb2.Task.Parent.TASK
  # repo_subtree.parent.task.id = owner_subtree.id

  owner_tagtree = workspace.tags.add()
  owner_tagtree.id = f"@tags//:{args.owner}"

  repo_tagtree = owner_tagtree.children.add()
  repo_tagtree.id = f"@tags//{args.owner}:{args.repo}"

  for label in r.get_labels():
    tag = repo_tagtree.children.add()
    tag.id = f"@tags//{args.owner}/{args.repo}/Labels:{label.name}"
    tag.body = label.name
    # TODO: tag.color

  for issue in r.get_issues(state="all"):
    # task.created = task.created_at
    # task.completed = task.closed_at

    # issues are children of their milestone
    if issue.milestone:
      milestone_id = f"//{args.owner}/{args.repo}:{issue.milestone.title}"

      if milestone_id not in [x.id for x in repo_subtree.children]:
        milestone = repo_subtree.children.add()
        milestone.id = milestone_id
        milestone.body = (
            f"{issue.milestone.title}\n\n{issue.milestone.description}"
        ).rstrip()
        milestone.status = freefocus_pb2.Task.ACTIVE

        milestone.parent.type = freefocus_pb2.Task.Parent.TASK
        milestone.parent.task.id = repo_subtree.id

      task = milestone.children.add()
      task.id = f"//{args.owner}/{args.repo}/{issue.milestone.title}:{issue.number}"

      task.parent.type = freefocus_pb2.Task.Parent.TASK
      task.parent.task.id = milestone_id
    else:
      task = repo_subtree.children.add()
      task.id = f"//{args.owner}/{args.repo}:{issue.number}"

      task.parent.type = freefocus_pb2.Task.Parent.TASK
      task.parent.task.id = repo_subtree.id

    task.body = (
        f"{issue.title}\n\n" + issue.body.replace('\r\n', '\n')).rstrip()
    task.status = freefocus_pb2.Task.ACTIVE

    for i, comment in enumerate(issue.get_comments()):
      j = i + 1
      comment_node = task.comments.add()
      comment_node.id = f"@comments//{args.owner}/{args.repo}/{issue.number}:{j}"
      comment_node.parent.type = freefocus_pb2.Comment.Parent.TASK
      comment_node.parent.task.id = task.id
      comment_node.body = issue.body

    for label in issue.get_labels():
      tag = task.tags.add()
      tag.id = f"@tags//{args.owner}:{args.repo}/Labels:{label.name}"

    for assignee in issue.assignees:
      pass

  print(MessageToJson(workspace))
