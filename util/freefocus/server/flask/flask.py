#!/usr/bin/env python
"""Run a REST server."""
import typing
from argparse import ArgumentParser
from contextlib import contextmanager

import flask
import flask_cors
import sqlalchemy
from flask import abort
from flask import request

from util.freefocus import freefocus
from util.freefocus import sql


app = flask.Flask(__name__)
flask_cors.CORS(app)
app.config.from_object("config")

make_session = None


@contextmanager
def Session(commit: bool = False) -> sqlalchemy.orm.session.Session:
  """Provide a transactional scope around a series of operations."""
  session = make_session()
  try:
    yield session
    if commit:
      session.commit()
  except:
    session.rollback()
    raise
  finally:
    session.close()


API_BASE = f"/api/v{freefocus.SPEC_MAJOR}.{freefocus.SPEC_MINOR}"
URL_STUB = "http://" + app.config.get("SERVER_NAME", "") + API_BASE


def active_task_graph():
  def build_graph(session, task: sql.Task):
    return {
      "id": task.id,
      "body": task.body.split("\n")[0],
      "completed": True if task.completed else False,
      "children": [
        build_graph(session, child)
        for child in session.query(sql.Task)
        .filter(sql.Task.parent_id == task.id)
        .order_by(sql.Task.created.desc())
      ],
    }

  with Session() as session:
    # List 'root' tasks
    q = (
      session.query(sql.Task)
      .filter(sql.Task.parent_id == None)
      .order_by(sql.Task.created.desc())
    )

    r = [build_graph(session, t) for t in q]
    return r


@app.route("/")
def index():
  data = {
    "freefocus": {
      "version": f"{freefocus.SPEC_MAJOR}.{freefocus.SPEC_MINOR}.{freefocus.SPEC_MICRO}",
    },
    "assets": {
      "cache_tag": 1,
      "bootstrap_css": flask.url_for("static", filename="bootstrap.css"),
      "styles_css": flask.url_for("static", filename="styles.css"),
      "site_js": flask.url_for("static", filename="site.js"),
    },
    "tasks": active_task_graph(),
  }

  return flask.render_template("lists.html", **data)


def response(data):
  """ make an API response """
  return jsonify(data)


def paginated_response(iterable: typing.Iterable):
  """ make a paginated API response """
  # TODO: chunk and paginate
  return response(list(iterable))


def truncate(string: str, maxlen=144):
  suffix = "..."

  if len(string) > maxlen:
    truncated = string[: maxlen - len(suffix)] + suffix
    return {"data": truncated, "truncated": True}
  else:
    return {"data": string, "truncated": False}


def task_url(task: sql.Task):
  return URL_STUB + f"/tasks/{task.id}"


def group_url(group: sql.Group):
  return URL_STUB + f"/groups/{group.id}"


def asset_url(asset: sql.Asset):
  return URL_STUB + f"/assets/{group.id}"


def date(d):
  if d:
    return d.isoformat()
  else:
    return None


@app.errorhandler(404)
def not_found(error):
  """ 404 error handler """
  return make_response(jsonify({"error": "Not found"}), 404)


@app.errorhandler(400)
def not_found(error):
  """ 400 Bad Request """
  return make_response(jsonify({"error": "Bad Request"}), 400)


@app.route(API_BASE + "/persons", methods=["GET"])
def get_persons():
  with Session() as session:
    q = session.query(sql.Person)
    return paginated_response(p.json() for p in q)


@app.route(API_BASE + "/persons/<int:person_uid>", methods=["GET"])
def get_person(person_uid: int):
  with Session() as session:
    p = session.query(sql.Person).filter(sql.Person.uid == person_uid).first()
    if not p:
      abort(404)
    return response(p.json())


@app.route(API_BASE + "/persons/<int:person_uid>/groups", methods=["GET"])
def get_person_groups(person_uid: int):
  with Session() as session:
    p = session.query(sql.Person).filter(sql.Person.uid == person_uid).first()
    if not p:
      abort(404)
    return paginated_response(g.json() for g in p.groups)


@app.route(API_BASE + "/tasks", methods=["GET"])
def get_tasks():
  def build_graph(session, task: sql.Task = None):
    parent = None if task is None else task.id
    q = (
      session.query(sql.Task)
      .filter(sql.Task.parent_id == parent)
      .order_by(sql.Task.created.desc())
    )

    # "Completed" request parameter
    completed = request.args.get("completed", None)
    if completed is not None:
      if completed == "true":
        q = q.filter(sql.Task.completed)
      elif completed == "false":
        q = q.filter(sql.Task.completed == None)
      else:
        abort(400)

    children = [build_graph(session, t) for t in q]

    if task is None:
      return children
    else:
      return {
        "url": task_url(task),
        "body": truncate(task.body),
        "status": task.status,
        "assigned": [g.id for g in task.assigned],
        "children": children,
      }

  with Session() as session:
    return paginated_response(build_graph(session))


@app.route(API_BASE + "/tasks/<int:task_id>", methods=["GET"])
def get_task(task_id: int):
  with Session() as session:
    t = session.query(sql.Task).filter(sql.Task.id == task_id).first()
    if not t:
      abort(404)
    return response(
      {
        "body": t.body,
        "assigned": t.is_assigned,
        "blocked": t.is_blocked,
        "defer_until": date(t.defer_until),
        "start_on": date(t.start_on),
        "estimated_duration": t.duration,
        "due": date(t.due),
        "started": date(t.started),
        "completed": date(t.completed),
        "created": {"at": date(t.created), "by": group_url(t.created_by),},
      }
    )


@app.route(API_BASE + "/tasks/<int:task_id>/owners", methods=["GET"])
def get_task_owners(task_id: int):
  with Session() as session:
    t = session.query(sql.Task).filter(sql.Task.id == task_id).first()
    if not t:
      abort(404)
    # TODO: summary
    return paginated_response(group_url(g) for g in t.owners)


@app.route(API_BASE + "/tasks/<int:task_id>/assigned", methods=["GET"])
def get_task_assigned(task_id: int):
  with Session() as session:
    t = session.query(sql.Task).filter(sql.Task.id == task_id).first()
    if not t:
      abort(404)
    # TODO: summary
    return paginated_response(group_url(g) for g in t.assigned)


@app.route(API_BASE + "/tasks", methods=["POST"])
def add_task():
  with Session(commit=True) as session:
    pass


def main():
  global make_session

  parser = ArgumentParser(description=__doc__)
  parser.add_argument("uri")
  parser.add_argument("-v", "--verbose", action="store_true")
  args = parser.parse_args()

  engine = sqlalchemy.create_engine(args.uri, echo=args.verbose)

  sql.Base.metadata.create_all(engine)
  sql.Base.metadata.bind = engine
  make_session = sqlalchemy.orm.sessionmaker(bind=engine)

  app.RunWithArgs(debug=True, host="0.0.0.0")


if __name__ == "__main__":
  main()
