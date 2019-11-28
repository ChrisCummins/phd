"""A flask server which renders test results."""
import os
import threading

import flask
import flask_sqlalchemy
import portpicker
import sqlalchemy as sql

import build_info
from deeplearning.clgen.corpuses import encoded
from deeplearning.clgen.dashboard import dashboard_db
from labm8.py import app
from labm8.py import bazelutil
from labm8.py import humanize

FLAGS = app.FLAGS

app.DEFINE_integer(
  "clgen_dashboard_port",
  portpicker.pick_unused_port(),
  "The port to launch the server on.",
)

flask_app = flask.Flask(
  __name__,
  template_folder=bazelutil.DataPath(
    "phd/deeplearning/clgen/dashboard/templates"
  ),
  static_folder=bazelutil.DataPath("phd/deeplearning/clgen/dashboard/static"),
)
flask_app.config["SQLALCHEMY_DATABASE_URI"] = os.environ.get(
  "CLGEN_DASHBOARD", "sqlite:////tmp/phd/deeplearning/clgen/dashboard.db"
)
db = flask_sqlalchemy.SQLAlchemy(flask_app)


def GetBaseTemplateArgs():
  return {
    "urls": {
      "cache_tag": build_info.BuildTimestamp(),
      "styles_css": flask.url_for("static", filename="bootstrap.css"),
      "site_css": flask.url_for("static", filename="site.css"),
      "site_js": flask.url_for("static", filename="site.js"),
    },
    "build_info": {
      "html": build_info.FormatShortBuildDescription(html=True),
      "version": build_info.Version(),
    },
    "dashboard_info": {"db": flask_app.config["SQLALCHEMY_DATABASE_URI"],},
  }


@flask_app.route("/")
def index():
  corpuses = db.session.query(
    dashboard_db.Corpus.id,
    dashboard_db.Corpus.encoded_url,
    dashboard_db.Corpus.summary,
  ).all()
  models = db.session.query(
    dashboard_db.Model.id,
    dashboard_db.Model.cache_path,
    dashboard_db.Model.corpus_id,
    dashboard_db.Model.summary,
  ).all()

  data = {
    "corpuses": {
      x.id: {"name": x.encoded_url, "summary": x.summary, "models": {}}
      for x in corpuses
    },
  }

  for model in sorted(models, key=lambda x: x.id):
    data["corpuses"][model.corpus_id]["models"][model.id] = {
      "name": model.cache_path,
      "summary": model.summary,
    }

  return flask.render_template(
    "dashboard.html", data=data, **GetBaseTemplateArgs()
  )


@flask_app.route("/corpus/<int:corpus_id>/model/<int:model_id>/")
def report(corpus_id: int, model_id: int):
  corpus, corpus_config_proto, preprocessed_url, encoded_url = (
    db.session.query(
      dashboard_db.Corpus.summary,
      dashboard_db.Corpus.config_proto,
      dashboard_db.Corpus.preprocessed_url,
      dashboard_db.Corpus.encoded_url,
    )
    .filter(dashboard_db.Corpus.id == corpus_id)
    .one()
  )
  model, model_config_proto = (
    db.session.query(
      dashboard_db.Model.summary, dashboard_db.Model.config_proto
    )
    .filter(dashboard_db.Model.id == model_id)
    .one()
  )

  telemetry = (
    db.session.query(
      dashboard_db.TrainingTelemetry.timestamp,
      dashboard_db.TrainingTelemetry.epoch,
      dashboard_db.TrainingTelemetry.step,
      dashboard_db.TrainingTelemetry.training_loss,
    )
    .filter(dashboard_db.TrainingTelemetry.model_id == model_id)
    .all()
  )

  q1 = (
    db.session.query(sql.func.max(dashboard_db.TrainingTelemetry.id))
    .filter(dashboard_db.TrainingTelemetry.model_id == model_id)
    .group_by(dashboard_db.TrainingTelemetry.epoch)
  )

  q2 = (
    db.session.query(
      dashboard_db.TrainingTelemetry.timestamp,
      dashboard_db.TrainingTelemetry.epoch,
      dashboard_db.TrainingTelemetry.step,
      dashboard_db.TrainingTelemetry.learning_rate,
      dashboard_db.TrainingTelemetry.training_loss,
      dashboard_db.TrainingTelemetry.pending,
    )
    .filter(dashboard_db.TrainingTelemetry.id.in_(q1))
    .order_by(dashboard_db.TrainingTelemetry.id)
  )

  q3 = (
    db.session.query(
      sql.sql.expression.cast(
        sql.func.avg(dashboard_db.TrainingTelemetry.ns_per_batch), sql.Integer
      ).label("us_per_step"),
    )
    .group_by(dashboard_db.TrainingTelemetry.epoch)
    .filter(dashboard_db.TrainingTelemetry.model_id == model_id)
    .order_by(dashboard_db.TrainingTelemetry.id)
  )

  epoch_telemetry = [
    {
      "timestamp": r2.timestamp.strftime("%Y-%m-%d %H:%M:%S"),
      "epoch": r2.epoch,
      "step": humanize.Commas(r2.step),
      "learning_rate": f"{r2.learning_rate:.5E}",
      "training_loss": f"{r2.training_loss:.6f}",
      "pending": r2.pending,
      "us_per_step": humanize.Duration(r3.us_per_step / 1e6),
    }
    for r2, r3 in zip(q2, q3)
  ]

  data = GetBaseTemplateArgs()
  data["corpus_id"] = corpus_id
  data["model_id"] = model_id
  data["corpus"] = corpus
  data["model"] = model
  data["data"] = {
    "corpus_config_proto": corpus_config_proto,
    "model_config_proto": model_config_proto,
    "telemetry": telemetry,
    "epoch_telemetry": epoch_telemetry,
    "preprocessed_url": preprocessed_url,
    "encoded_url": encoded_url,
  }
  data["urls"]["view_encoded_file"] = f"/corpus/{corpus_id}/encoded/random/"

  return flask.render_template("report.html", **data)


@flask_app.route("/corpus/<int:corpus_id>/encoded/random/")
def random_encoded_contentfile(corpus_id: int):
  (encoded_url,) = (
    db.session.query(dashboard_db.Corpus.encoded_url)
    .filter(dashboard_db.Corpus.id == corpus_id)
    .one()
  )

  encoded_db = encoded.EncodedContentFiles(encoded_url, must_exist=True)

  with encoded_db.Session() as session:
    (random_id,) = (
      session.query(encoded.EncodedContentFile.id)
      .order_by(encoded_db.Random())
      .limit(1)
      .one()
    )

  return flask.redirect(f"/corpus/{corpus_id}/encoded/{random_id}/", code=302)


@flask_app.route("/corpus/<int:corpus_id>/encoded/<int:encoded_id>/")
def encoded_contentfile(corpus_id: int, encoded_id: int):
  (encoded_url,) = (
    db.session.query(dashboard_db.Corpus.encoded_url)
    .filter(dashboard_db.Corpus.id == corpus_id)
    .one()
  )

  encoded_db = encoded.EncodedContentFiles(encoded_url, must_exist=True)

  with encoded_db.Session() as session:
    cf = (
      session.query(encoded.EncodedContentFile)
      .filter(encoded.EncodedContentFile.id == encoded_id)
      .limit(1)
      .one()
    )
    indices = cf.indices_array
    vocab = {
      v: k
      for k, v in encoded.EncodedContentFiles.GetVocabFromMetaTable(
        session
      ).items()
    }
    tokens = [vocab[i] for i in indices]
    text = "".join(tokens)
    encoded_cf = {
      "id": cf.id,
      "tokencount": humanize.Commas(cf.tokencount),
      "indices": indices,
      "text": text,
      "tokens": tokens,
    }
    vocab = {
      "table": [(k, v) for k, v in vocab.items()],
      "size": len(vocab),
    }

  data = GetBaseTemplateArgs()
  data["encoded"] = encoded_cf
  data["vocab"] = vocab
  data["urls"]["view_encoded_file"] = f"/corpus/{corpus_id}/encoded/random/"
  return flask.render_template("encoded_contentfile.html", **data)


@flask_app.route(
  "/corpus/<int:corpus_id>/model/<int:model_id>/samples/<int:epoch>"
)
def samples(corpus_id: int, model_id: int, epoch: int):
  samples = (
    db.session.query(
      dashboard_db.TrainingSample.sample,
      dashboard_db.TrainingSample.token_count,
      dashboard_db.TrainingSample.sample_time,
    )
    .filter(
      dashboard_db.TrainingSample.model_id == model_id,
      dashboard_db.TrainingSample.epoch == epoch,
    )
    .all()
  )

  data = {
    "samples": samples,
  }

  opts = GetBaseTemplateArgs()
  opts["urls"]["back"] = f"/corpus/{corpus_id}/model/{model_id}/"

  return flask.render_template(
    "samples.html", data=data, corpus_id=corpus_id, model_id=model_id, **opts
  )


def Launch(debug: bool = False):
  """Launch dashboard in a separate thread."""
  app.Log(
    1,
    "Launching CLgen dashboard on http://127.0.0.1:%d",
    FLAGS.clgen_dashboard_port,
  )
  kwargs = {
    "port": FLAGS.clgen_dashboard_port,
    # Debugging must be disabled when run in a separate thread.
    "debug": debug,
    "host": "0.0.0.0",
  }
  db.create_all()
  if debug:
    flask_app.run(**kwargs)
  else:
    thread = threading.Thread(target=flask_app.run, kwargs=kwargs)
    thread.setDaemon(True)
    thread.start()
    return thread
