"""A flask server which renders test results."""
import threading

import flask
import portpicker

import build_info
from deeplearning.clgen.dashboard import dashboard_db
from labm8 import app
from labm8 import bazelutil

FLAGS = app.FLAGS

app.DEFINE_database('clgen_dashboard_db', dashboard_db.DashboardDatabase,
                    'sqlite:////tmp/phd/deeplearning/clgen/dashboard.db',
                    'URL of the dashboard database.')
app.DEFINE_integer('clgen_dashboard_port', portpicker.pick_unused_port(),
                   'The port to launch the server on.')

flask_app = flask.Flask(
    __name__,
    template_folder=bazelutil.DataPath(
        'phd/deeplearning/clgen/dashboard/templates'),
    static_folder=bazelutil.DataPath('phd/deeplearning/clgen/dashboard/static'),
)


@flask_app.route('/')
def index():
  assert bazelutil.DataPath(
      'phd/deeplearning/clgen/dashboard/static/bootstrap.css')
  app.Log(1, 'Rendering index')
  urls = {
      'cache_tag': build_info.BuildTimestamp(),
      'styles_css': flask.url_for('static', filename='bootstrap.css'),
      'site_js': flask.url_for('static', filename='site.js'),
  }
  return flask.render_template(
      'dashboard.html',
      urls=urls,
      build_info=build_info.FormatShortBuildDescription(html=True))


def GetDatabase() -> dashboard_db.DashboardDatabase:
  return FLAGS.clgen_dashboard_db()


def Launch(debug: bool = False):
  """Launch dashboard in a separate thread."""
  app.Log(1, 'Launching dashboard on http://127.0.0.1:%d',
          FLAGS.clgen_dashboard_port)
  kwargs = {
      'port': FLAGS.clgen_dashboard_port,
      # Debugging must be disabled when run in a separate thread.
      'debug': debug,
      'host': '0.0.0.0',
  }
  if debug:
    flask_app.run(**kwargs)
  else:
    thread = threading.Thread(target=flask_app.run, kwargs=kwargs)
    thread.start()
    return thread
