"""Import OmniFocus database into FreeFocus Protobuf messages."""
import os
from argparse import ArgumentParser

from util.freefocus.sql import *


OMNIFOCUS_DB_PATH = (
  "~/Library/Containers/com.omnigroup.OmniFocus2/Data/Library/Caches/"
  "com.omnigroup.OmniFocus2/OmniFocusDatabase2"
)


class Task(object):
  pass


def load_of_session(path):
  import sqlalchemy as sql
  from sqlalchemy import create_engine, MetaData, Table

  engine = create_engine("sqlite:///%s" % path, echo=True)
  metadata = MetaData(engine)

  sql.orm.mapper(Task, Table("Task", metadata, autoload=True))

  return sql.orm.sessionmaker(bind=engine)()


if __name__ == "__main__":
  parser = ArgumentParser(description=__doc__)
  parser.add_argument("uri")
  parser.add_argument("--db", default=OMNIFOCUS_DB_PATH)
  parser.add_argument("-v", "--verbose", action="store_true")
  args = parser.parse_args()

  engine = sql.create_engine(args.uri, echo=args.verbose)
  Base.metadata.create_all(engine)
  Base.metadata.bind = engine
  make_session = sql.orm.sessionmaker(bind=engine)

  session = make_session()
  of_session = load_of_session(os.path.expanduser(args.db))

  for task in of_session.query(Task):
    print(task)
