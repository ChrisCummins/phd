"""A flask server which renders test results."""
from typing import Any, Dict, List

import flask
import portpicker
import sqlalchemy as sql

from config import build_info
from labm8 import app
from labm8 import bazelutil
from labm8 import humanize
from labm8 import prof
from tools.continuous_integration import bazel_test_db as db

FLAGS = app.FLAGS

app.DEFINE_string(
    "db", "sqlite:////tmp/phd/tools/continuous_integration/buildbot.db",
    "Path to testlogs summary database.")
app.DEFINE_string("buildbot_url", "http://localhost:8010/#/builders",
                  "URL of buildbot server.")
app.DEFINE_string("hostname", "localhost", "Hostname of this server.")
app.DEFINE_integer('port', portpicker.pick_unused_port(),
                   'The port to launch the server on.')
app.DEFINE_boolean('debug_flask_server', True,
                   'Launch flask app with debugging enabled.')

flask_app = flask.Flask(
    __name__,
    template_folder=bazelutil.DataPath(
        'phd/tools/continuous_integration/buildbot/report_viewer/templates'),
    static_folder=bazelutil.DataPath(
        'phd/tools/continuous_integration/buildbot/report_viewer/static'))


def DeltaToTargets(delta: db.TestDelta) -> List[Dict[str, Any]]:
  return [{
      'bazel_target': result.bazel_target,
      'runtime': humanize.Duration(result.runtime_ms / 1000),
      'test_count': result.test_count,
      'git_branch': result.git_branch,
      'git_commit': result.git_commit,
      'log': result.log,
      'changed': True,
      'fail': True,
      'runtime_ms': result.runtime_ms,
  } for result in delta.broken] + [
      {
          'bazel_target': result.bazel_target,
          'runtime': humanize.Duration(result.runtime_ms / 1000),
          'test_count': result.test_count,
          'git_branch': result.git_branch,
          'git_commit': result.git_commit,
          'log': result.log,
          'changed': False,
          'fail': True,
          'runtime_ms': result.runtime_ms,
      } for result in delta.still_broken
  ] + [{
      'bazel_target': result.bazel_target,
      'runtime': humanize.Duration(result.runtime_ms / 1000),
      'test_count': result.test_count,
      'git_branch': result.git_branch,
      'git_commit': result.git_commit,
      'log': result.log,
      'changed': True,
      'fail': False,
      'runtime_ms': result.runtime_ms,
  } for result in delta.fixed
      ] + [{
          'bazel_target': result.bazel_target,
          'runtime': humanize.Duration(result.runtime_ms / 1000),
          'test_count': result.test_count,
          'git_branch': result.git_branch,
          'git_commit': result.git_commit,
          'log': result.log,
          'changed': False,
          'fail': False,
          'runtime_ms': result.runtime_ms,
      } for result in delta.still_pass]


def RenderInvocation(host, session, invocation):
  app.Log(1, 'Fetching for invocation %s', invocation)
  delta = db.GetTestDelta(session, invocation, to_return=[db.TestTargetResult])
  targets = DeltaToTargets(delta)

  urls = {
      "cache_tag":
      1,
      "self":
      f"http://{FLAGS.hostname}:{FLAGS.port}",
      "styles_css":
      flask.url_for('static', filename='bootstrap.css'),
      "site_js":
      flask.url_for('static', filename='site.js'),
      "buildbot":
      FLAGS.buildbot_url,
      "github_commit": ('https://github.com/ChrisCummins/phd-priv/commit/'
                        f'{targets[0]["git_commit"]}'),
  }

  hosts = [
      x[0] for x in session.query(sql.func.distinct(
          db.TestTargetResult.host)).order_by(db.TestTargetResult.host)
  ]
  invocations = [
      x[0] for x in session.query(
          sql.func.distinct(db.TestTargetResult.invocation_datetime)).filter(
              db.TestTargetResult.host == host).order_by(
                  db.TestTargetResult.invocation_datetime.desc())
  ]
  invocations = zip(range(len(invocations), 0, -1), invocations)

  return flask.render_template(
      "test_view.html",
      host=host,
      hosts=hosts,
      invocations=invocations,
      targets=targets,
      test_count=humanize.Commas(sum(t['test_count'] for t in targets)),
      test_duration=humanize.Duration(
          sum(t['runtime_ms'] for t in targets) / 1000),
      git_commit=targets[0]['git_commit'],
      git_branch=targets[0]['git_branch'],
      delta=db.TestDelta(
          broken=len(delta.broken),
          fixed=len(delta.fixed),
          still_broken=len(delta.still_broken),
          still_pass=len(delta.still_pass)),
      invocation_datetime=invocation,
      urls=urls,
      build_info=build_info.GetBuildInfo())


@flask_app.route("/<host>")
def test(host: str):
  with prof.Profile(f'Render /{host}'):
    database = db.Database(FLAGS.db)
    with database.Session() as session:
      invocation, = session.query(sql.func.max_(db.TestTargetResult.invocation_datetime))\
        .filter(db.TestTargetResult.host == host).one()
      template = RenderInvocation(host, session, invocation)
  return template


@flask_app.route("/<host>/<int:invocation_num>")
def index_invocation(host: str, invocation_num: int):
  with prof.Profile(f'Render /{host}/{invocation_num}'):
    database = db.Database(FLAGS.db)
    with database.Session() as session:
      invocation, = session.query(sql.func.distinct(db.TestTargetResult.invocation_datetime)) \
        .filter(db.TestTargetResult.host == host)\
        .order_by(db.TestTargetResult.invocation_datetime)\
        .offset(invocation_num - 1).limit(1).one()
      template = RenderInvocation(host, session, invocation)
  return template


def main():
  """Main entry point."""
  # TODO: Implement!
  flask_app.run(port=FLAGS.port, debug=FLAGS.debug_flask_server, host='0.0.0.0')


if __name__ == '__main__':
  app.Run(main)
