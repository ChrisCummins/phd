"""
The datastore acts as the bridge between the RPC frontend and the db backend.
"""
import typing

from contextlib import contextmanager
from datetime import datetime

from absl import flags
from labm8 import crypto
from sqlalchemy import orm

from deeplearning.deepsmith import db
from deeplearning.deepsmith import dbutil
from deeplearning.deepsmith.protos import deepsmith_pb2 as pb

FLAGS = flags.FLAGS

class DataStore(object):
  """ The centralized data store. """

  def __init__(self, **db_opts):
    self.opts = db_opts
    self._engine, _ = dbutil.make_engine(**self.opts)
    db.Base.metadata.create_all(self._engine)
    db.Base.metadata.bind = self._engine
    self._make_session = orm.sessionmaker(bind=self._engine)

  @contextmanager
  def session(self, commit: bool = False) -> db.session_t:
    session = self._make_session()
    try:
      yield session
      if commit:
        session.commit()
    except:
      session.rollback()
      raise
    finally:
      session.close()

  def add_testcases(self, testcases: typing.List[pb.Testcase]) -> None:
    """
    Add a sequence of testcases to the datastore.

    TODO: Optimize to reduce the number of SQL queries by batching lookups/inserts.
    """
    with self.session(commit=True) as session:
      for testcase in testcases:
        self._add_one_testcase(session, testcase)

  def _add_one_testcase(self, session: db.session_t,
                        testcase_pb: pb.Testcase) -> None:
    # Add generator:
    generator = dbutil.get_or_add(
        session, db.Generator,
        name=testcase_pb.generator.name,
        version=testcase_pb.generator.name,
    )

    # Add harness:
    harness = dbutil.get_or_add(
        session, db.Harness,
        name=testcase_pb.harness.name,
        version=testcase_pb.harness.version,
    )

    # Add testcase:
    testcase = dbutil.get_or_add(
        session, db.Testcase,
        generator=generator,
        harness=harness,
    )

    # Add inputs:
    for input_pb in testcase_pb.inputs:
      name = dbutil.get_or_add(
          session, db.TestcaseInputName,
          name=input_pb.name,
      )
      sha1 = crypto.sha1_str(input_pb.text)
      input = dbutil.get_or_add(
          session, db.TestcaseInput,
          name=name,
          sha1=sha1,
          linecount=len(input_pb.text.split("\n")),
          charcount=len(input_pb.text),
          input=input_pb.text,
      )
      dbutil.get_or_add(
          session, db.TestcaseInputAssociation,
          testcase=testcase,
          input=input,
      )

    # Add options:
    for opt_ in testcase_pb.opts:
      opt = dbutil.get_or_add(
          session, db.TestcaseOpt,
          opt=opt_
      )
      dbutil.get_or_add(
          session, db.TestcaseOptAssociation,
          testcase=testcase, opt=opt
      )

    # Add timings:
    for timings_ in testcase_pb.timings:
      client = dbutil.get_or_add(
          session, db.Client,
          name=timings_.client
      )
      event = dbutil.get_or_add(
          session, db.Event,
          name=timings_.event
      )
      timing = dbutil.get_or_add(
          session, db.TestcaseTiming,
          testcase=testcase,
          event=event,
          client=client,
          duration_seconds=timings_.duration,
          date=datetime.fromtimestamp(timings_.event_epoch_seconds)
      )
