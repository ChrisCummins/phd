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
from deeplearning.deepsmith.protos import deepsmith_pb2

FLAGS = flags.FLAGS


class InvalidRequest(ValueError):
  """Exception raised if request cannot be served."""
  pass


class DataStore(object):
  """ The centralized data store. """

  def __init__(self, **db_opts):
    self.opts = db_opts
    self._engine, _ = dbutil.MakeEngine(**self.opts)
    db.Base.metadata.create_all(self._engine)
    db.Base.metadata.bind = self._engine
    self._make_session = orm.sessionmaker(bind=self._engine)

  @contextmanager
  def Session(self, commit: bool = False) -> db.session_t:
    """Provide a transactional scope around a session.

    Args:
      commit: If true, commit session at the end of scope.

    Returns:
      A database session.
    """
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

  def SubmitTestcases(self, request: deepsmith_pb2.SubmitTestcasesRequest,
                      response: deepsmith_pb2.SubmitTestcasesResponse) -> None:
    """Add a sequence of testcases to the datastore.
    """
    with self.Session(commit=True) as session:
      for testcase in request.testcases:
        self._AddOneTestcase(session, testcase)

  def _AddOneTestcase(self, session: db.session_t,
                      testcase_pb: deepsmith_pb2.Testcase) -> None:
    """Record a single Testcase in the database.
    """
    # Add language:
    language = dbutil.GetOrAdd(
        session, db.Language,
        name=testcase_pb.language,
    )

    # Add generator:
    generator = dbutil.GetOrAdd(
        session, db.Generator,
        name=testcase_pb.generator.name,
        version=testcase_pb.generator.name,
    )

    # Add harness:
    harness = dbutil.GetOrAdd(
        session, db.Harness,
        name=testcase_pb.harness.name,
        version=testcase_pb.harness.version,
    )

    # Add testcase:
    testcase = dbutil.GetOrAdd(
        session, db.Testcase,
        language=language,
        generator=generator,
        harness=harness,
    )

    # Add inputs:
    for input_pb in testcase_pb.inputs:
      text = testcase_pb.inputs[input_pb]
      name = dbutil.GetOrAdd(
          session, db.TestcaseInputName,
          name=input_pb,
      )
      sha1 = crypto.sha1_str(text)

      input = dbutil.GetOrAdd(
          session, db.TestcaseInput,
          name=name,
          sha1=sha1,
          linecount=len(text.split("\n")),
          charcount=len(text),
          input=text,
      )
      dbutil.GetOrAdd(
          session, db.TestcaseInputAssociation,
          testcase=testcase,
          input=input,
      )

    # TODO(cec): If the testcase already exists, don't add timings.

    # Add timings:
    for timings_ in testcase_pb.timings:
      client = dbutil.GetOrAdd(
          session, db.Client,
          name=timings_.client
      )
      profiling_event_name = dbutil.GetOrAdd(
          session, db.ProfilingEventName,
          name=timings_.name
      )
      timing = dbutil.GetOrAdd(
          session, db.TestcaseTiming,
          testcase=testcase,
          name=profiling_event_name,
          client=client,
          duration_seconds=timings_.duration_seconds,
          date=datetime.fromtimestamp(timings_.date_epoch_seconds)
      )

  def _BuildTestcaseRequestQuery(self, session, request) -> db.query_t:
    def _FilterLanguageGeneratorHarness(q):
      if request.has_language:
        language = dbutil.GetOrAdd(db.Language, name=request.language)
        if not language:
          raise LookupError
        q = q.filter(db.Testcase.language_id == language.id)

      # Filter by generator.
      if request.has_generator:
        generator = session.query(db.Generator) \
          .filter(db.Generator.name == request.generator.name,
                  db.Generator.version == request.generator.version) \
          .first()
        if not generator:
          raise LookupError
        q = q.filter(db.Testcase.generator_id == generator.id)

      # Filter by generator.
      if request.has_harness:
        harness = session.query(db.Harness) \
          .filter(db.Harness.name == request.harness.name,
                  db.Harness.version == request.harness.version) \
          .first()
        if not harness:
          raise LookupError
        q = q.filter(db.Testcase.harness_id == harness.id)

      return q

    q = session.query(db.Testcase)
    q = _FilterLanguageGeneratorHarness(q)

    testbed_id = None
    if request.has_testbed():
      language = dbutil.GetOrAdd(db.Language, name=request.testbed)
      testbed = dbutil.GetOrAdd(db.Testbed, language=language,
                                name=request.testbed.language,
                                version=request.testbed.version)
      testbed_id = testbed.id

    if testbed_id and not request.include_testcases_with_results:
      q2 = session.query(db.Result.testcase_id) \
        .join(db.Testcase) \
        .filter(db.Result.testbed_id == testbed_id)
      q2 = _FilterLanguageGeneratorHarness(q2)
      q = q.filter(~db.Testcase.id.in_(q2))

    if testbed_id and not request.include_testcases_with_pending_results:
      q2 = session.query(db.PendingResult.testcase_id) \
        .join(db.Testcase)\
        .filter(db.PendingResult.testbed_id == testbed_id)
      q2 = _FilterLanguageGeneratorHarness(q2)
      q = q.filter(~db.Testcase.id.in_(q2))

    return q

  def _GeneratorObjectToProto(self, generator: db.Generator) -> deepsmith_pb2.Generator:
    return deepsmith_pb2.Generator(name=generator.name,
                                   version=generator.version)

  def _HarnessObjectToProto(self, harness: db.Harness) -> deepsmith_pb2.Harness:
    return deepsmith_pb2.Harness(name=harness.name,
                                 version=harness.version)

  def _TestcaseObjectToProto(self, testcase: db.Testcase) -> deepsmith_pb2.Testcase:
    return deepsmith_pb2.Testcase(
        language=testcase.language,
        generator=self._GeneratorObjectToProto(testcase.generator),
        harness=self._HarnessObjectToProto(testcase.harness),
        inputs={i.name: i.input for i in testcase.inputs},
        # TODO(cec) optionally set timings field.
    )

  def RequestTestcases(
      self, request: deepsmith_pb2.RequestTestcasesRequest,
      response: deepsmith_pb2.RequestTestcasesResponse) -> None:
    """Request testcases.
    """
    with self.Session(commit=False) as session:
      # Validate request parameters.
      if request.max_num_testcases < 1:
        raise InvalidRequest(
            f"max_num_testcases must be >= 1, not {request.max_num_testcases}")

      q = self._BuildTestcaseRequestQuery(session, request)
      q.limit(request.max_num_testcases)

      if request.return_testcases:
        response.testcases = [
          self._TestcaseObjectToProto(testcase) for testcase in q]

      if request.return_total_matching_count:
        q2 = self._BuildTestcaseRequestQuery(session, request)
        response.total_matching_count = q2.count()
