"""Convert a GitHub issue tracker into FreeFocus Protobuf messages."""
import os
import sys
from argparse import ArgumentParser
from configparser import ConfigParser

from github import Github

from util.freefocus.sql import *

if __name__ == "__main__":
  parser = ArgumentParser(description=__doc__)
  parser.add_argument("owner")
  parser.add_argument("repo")
  parser.add_argument("uri")
  parser.add_argument("--githubrc", default="~/.githubrc")
  parser.add_argument("-v", "--verbose", action="store_true")
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

  engine = sql.create_engine(args.uri, echo=args.verbose)

  Base.metadata.create_all(engine)
  Base.metadata.bind = engine
  make_session = sql.orm.sessionmaker(bind=engine)

  session = make_session()

  def get_or_create_usergroup(p: Person) -> Group:
    q = session.query(Group) \
      .filter(Group.body == p.name,
              Group.created == p.created)

    if q.count():
      return q.one()
    else:
      usergroup = Group(body=p.name, created=p.created)
      usergroup.members.append(p)
      session.add(usergroup)
      return usergroup

  def get_or_create_user(github_user) -> Person:
    q = session.query(Person).filter(Person.name == github_user.login,
                                     Person.created == github_user.created_at)
    if github_user.email:
      q = q.join(Email).filter(Email.address == github_user.email)

    if q.count():
      return q.one()
    else:
      p = Person(name=github_user.login, created=github_user.created_at)

      if github_user.email:
        p.emails.append(Email(address=github_user.email))

      session.add(p)
      get_or_create_usergroup(p)  # create group
      session.commit()

      return p

  # Create repo owner
  p = get_or_create_user(g.get_user(args.owner))
  usergroup = get_or_create_usergroup(p)

  # Create workspace
  workspace = Workspace(
      uid="GitHub", created=datetime.strptime("1/4/08", "%d/%m/%y"))
  session.add(workspace)

  # Import labels as Tags
  tagtree = Tag(body=args.owner, created_by=usergroup, created=r.created_at)
  tagtree.owners.append(usergroup)

  repo_tagtree = Tag(body=args.repo, created_by=usergroup, created=r.created_at)
  tagtree.children.append(repo_tagtree)

  for label in r.get_labels():
    tag = Tag(body=label.name, created_by=usergroup, created=r.created_at)
    # TODO: tag.color
    repo_tagtree.children.append(tag)

  session.add(tagtree)
  session.commit()

  tasktree = Task(body=args.owner, created_by=usergroup, created=r.created_at)
  tasktree.owners.append(usergroup)

  repo_tasktree = tasktree.add_subtask(
      Task(body=r.name, created_by=usergroup, created=r.created_at))

  # Import Milestones as Tasks
  for milestone in r.get_milestones(state='all'):
    a = get_or_create_usergroup(get_or_create_user(milestone.creator))
    m = repo_tasktree.add_subtask(
        Task(
            body=(f"{milestone.title}\n\n{milestone.description}").rstrip(),
            created_by=a,
            created=milestone.created_at,
            due=milestone.due_on,
            modified=milestone.updated_at,
        ))

    for label in milestone.get_labels():
      # lookup tag
      tag = session.query(Tag) \
        .filter(Tag.parent == repo_tagtree,
                Tag.body == label.name).one()
      m.tags.append(tag)

  session.add(repo_tasktree)
  session.commit()

  # Import Issues as Tasks
  for issue in r.get_issues(state="all"):
    if issue.milestone:
      # lookup milestone task
      milestone = session.query(Task) \
        .filter(Task.parent == repo_tasktree,
                Task.body == (
                  f"{issue.milestone.title}\n\n{issue.milestone.description}").rstrip()).one()
      task_parent = milestone
    else:
      task_parent = repo_tasktree

    a = get_or_create_usergroup(get_or_create_user(issue.user))
    task = task_parent.add_subtask(
        body=(f"{issue.title}\n\n" + issue.body.replace('\r\n', '\n')).rstrip(),
        completed=issue.closed_at,
        created_by=a,
        created=issue.created_at,
    )

    for comment in issue.get_comments():
      a = get_or_create_usergroup(get_or_create_user(comment.user))
      comment_node = TaskComment(
          body=comment.body,
          created_by=a,
          created=issue.created_at,
          modified=issue.updated_at,
      )
      task.comments.append(comment_node)

    for label in issue.get_labels():
      # lookup tag
      tag = session.query(Tag) \
        .filter(Tag.parent == repo_tagtree,
                Tag.body == label.name).one()

      task.tags.append(tag)

    for assignee in issue.assignees:
      a = get_or_create_usergroup(get_or_create_user(assignee))
      task.assigned.append(a)

    session.commit()

  session.add(tasktree)
  session.commit()
