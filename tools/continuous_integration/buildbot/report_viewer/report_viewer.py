"""A flask server which renders test results."""
from typing import Any, Dict, List

import flask
import portpicker
import sqlalchemy as sql

from labm8 import app
from labm8 import bazelutil
from labm8 import humanize
from labm8 import prof
from tools.continuous_integration import bazel_test_db as db

FLAGS = app.FLAGS

app.DEFINE_string(
    "db", "sqlite:////tmp/phd/tools/continuous_integration/buildbot.db",
    "Path to testlogs summary database.")
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
      'log': result.log,
      'changed': True,
      'fail': True,
  } for result in delta.broken] + [
      {
          'bazel_target': result.bazel_target,
          'runtime': humanize.Duration(result.runtime_ms / 1000),
          'test_count': result.test_count,
          'log': result.log,
          'changed': False,
          'fail': True,
      } for result in delta.still_broken
  ] + [{
      'bazel_target': result.bazel_target,
      'runtime': humanize.Duration(result.runtime_ms / 1000),
      'test_count': result.test_count,
      'log': result.log,
      'changed': True,
      'fail': False,
  } for result in delta.fixed
      ] + [{
          'bazel_target': result.bazel_target,
          'runtime': humanize.Duration(result.runtime_ms / 1000),
          'test_count': result.test_count,
          'log': result.log,
          'changed': False,
          'fail': False,
      } for result in delta.still_pass]


def RenderInvocation(host, session, invocation):
  app.Log(1, 'Fetching for invocation %s', invocation)
  delta = db.GetTestDelta(session, invocation, to_return=[db.TestTargetResult])

  urls = {
      "cache_tag": 1,
      "styles_css": flask.url_for('static', filename='bootstrap.css'),
      "site_js": flask.url_for('static', filename='site.js'),
  }

  return flask.render_template(
      "index.html",
      host=host,
      targets=DeltaToTargets(delta),
      delta=db.TestDelta(
          broken=len(delta.broken),
          fixed=len(delta.fixed),
          still_broken=len(delta.still_broken),
          still_pass=len(delta.still_pass)),
      invocation_datetime=invocation,
      urls=urls)


@flask_app.route("/<host>")
def index(host: str):
  with prof.Profile(f'Render /{host}'):
    database = db.Database(FLAGS.db)
    with database.Session() as session:
      invocation, = session.query(sql.func.max_(db.TestTargetResult.invocation_datetime))\
        .filter(db.TestTargetResult.host == host).one()
      return RenderInvocation(host, session, invocation)


def main():
  """Main entry point."""
  # TODO: Implement!
  flask_app.run(port=FLAGS.port, debug=FLAGS.debug_flask_server, host='0.0.0.0')


if __name__ == '__main__':
  app.Run(main)
