"""Convert a SQL workspace to a js timeline."""
import json
from argparse import ArgumentParser

from util.freefocus.sql import *


def escape(string):
  line = str(string).strip().split('\n')[0]
  return line


def make_id(item, type):
  if type == "workspace":
    return escape(f'Workspace: {item.uid}')
  elif type == "person":
    return escape(f"Person: {item.name}")
  elif type == "group":
    return escape(f'Group: {item.body}')
  elif type == 'asset':
    return escape(f'Asset: {item.body}')
  elif type == 'tag':
    return escape(f'Tag: {item.body}')
  elif type == 'task':
    return escape(f'Task: {item.body}')
  elif type == 'comment':
    return escape(f'Comment: {item.body}')
  else:
    raise LookupError(type)


if __name__ == "__main__":
  parser = ArgumentParser(description=__doc__)
  parser.add_argument("uri")
  parser.add_argument("-v", "--verbose", action="store_true")
  args = parser.parse_args()

  engine = sql.create_engine(args.uri, echo=args.verbose)

  Base.metadata.bind = engine
  make_session = sql.orm.sessionmaker(bind=engine)

  session = make_session()

  timeline = []

  workspace = session.query(Workspace).one()

  def add_event(item, type):
    event = {
        'id': len(timeline) + 1,
        'content': make_id(item, type),
        'start': item.created.isoformat(),
        'group': type,
    }

    if hasattr(item, 'completed'):
      if item.completed != None:
        event['end'] = item.completed.isoformat()

    timeline.append(event)

  add_event(workspace, 'workspace')
  for person in session.query(Person):
    add_event(person, 'person')
  for group in session.query(Group):
    add_event(group, 'group')
  for task in session.query(Task):
    add_event(task, 'task')
  for tag in session.query(Tag):
    add_event(tag, 'tag')
  for asset in session.query(Asset):
    add_event(asset, 'asset')
  for table in [
      WorkspaceComment, GroupComment, TagComment, TaskComment, AssetComment
  ]:
    for comment in session.query(table):
      add_event(comment, 'comment')

  print("""\
<!DOCTYPE HTML>
<!-- http://visjs.org/docs/timeline/#Example -->
<html>
<head>
  <title>Timeline | Basic demo</title>

  <style type="text/css">
    body, html {
      font-family: sans-serif;
    }
  </style>

  <script src="https://cdnjs.cloudflare.com/ajax/libs/vis/4.19.1/vis.min.js"></script>
  <link href="https://cdnjs.cloudflare.com/ajax/libs/vis/4.19.1/vis.min.css" rel="stylesheet" type="text/css" />
</head>
<body>
<div id="visualization"></div>

<script type="text/javascript">
  // DOM element where the Timeline will be attached
  var container = document.getElementById('visualization');

  // Create a DataSet (allows two way data-binding)
  var items = new vis.DataSet(%s);

  // Configuration for the Timeline
  var options = {};

  // Create a Timeline
  var timeline = new vis.Timeline(container, items, options);
</script>
</body>
</html>\
""" % json.dumps(timeline))
