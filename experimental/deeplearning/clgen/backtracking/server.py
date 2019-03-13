"""A flask server which renders test results."""

import json
import os
import time
from typing import Any, Dict

import flask
import portpicker

from config import build_info
from deeplearning.clgen import samplers
from experimental.deeplearning.clgen.backtracking import backtracking_model
from experimental.deeplearning.clgen.backtracking import logger
from labm8 import app
from labm8 import bazelutil
from labm8 import prof
from research.cummins_2017_cgo import generative_model

FLAGS = app.FLAGS

app.DEFINE_integer('port', portpicker.pick_unused_port(),
                   'The port to launch the server on.')
app.DEFINE_boolean('debug_flask_server', True,
                   'Launch flask app with debugging enabled.')
app.DEFINE_integer('sample_seed', 0, 'Random seed.')

flask_app = flask.Flask(
    __name__,
    template_folder=bazelutil.DataPath(
        'phd/experimental/deeplearning/clgen/backtracking/templates'),
    static_folder=bazelutil.DataPath(
        'phd/experimental/deeplearning/clgen/backtracking/static'))


@flask_app.route("/")
def index():
  with prof.Profile('Render /'):
    template_args = {
        'build_info': build_info.GetBuildInfo(),
        'urls': {
            "cache_tag": 1,
            "styles_css": flask.url_for('static', filename='bootstrap.css'),
            "site_js": flask.url_for('static', filename='site.js'),
        }
    }
    return flask.render_template("index.html", **template_args)


def Data(data: Dict[str, Any]):
  return f"retry: 100\ndata: {json.dumps(data)}\n\n"


def SampleStream():
  config = generative_model.CreateInstanceProtoFromFlags()

  os.environ['CLGEN_CACHE'] = config.working_dir

  logger = MyLogger()

  model = backtracking_model.BacktrackingModel(config.model, logger=logger)
  sampler = samplers.Sampler(config.sampler)

  model.Sample(sampler, FLAGS.clgen_min_sample_count, FLAGS.sample_seed)

  yield Data({'text': f'time now: {int(time.time())}\n'})


@flask_app.route("/append_state")
def stream():
  return flask.Response(SampleStream(), mimetype="text/event-stream")


class MyLogger(logger.BacktrackingLogger):

  def __init__(self):
    self._step_count = 0
    self._start_time = None

  def OnSampleStart(self,
                    backtracker: backtracking_model.OpenClBacktrackingHelper):
    self._step_count = 0
    self._start_time = time.time()

  def OnSampleStep(self,
                   backtracker: backtracking_model.OpenClBacktrackingHelper,
                   attempt_count: int, token_count: int):
    self._step_count += 1

    runtime_ms = int((time.time() - self._start_time) * 1000)
    app.Log(1, 'Reached step %d after %d attempts, %d tokens', self._step_count,
            attempt_count, token_count)

  def OnSampleEnd(self,
                  backtracker: backtracking_model.OpenClBacktrackingHelper):
    del backtracker
    self._step_count += 1
    app.Log(1, "Sampling concluded at step %d", self._step_count)
    self._step_count = 0


def main():
  """Main entry point."""
  flask_app.run(port=FLAGS.port, debug=FLAGS.debug_flask_server, host='0.0.0.0')


if __name__ == '__main__':
  app.Run(main)
