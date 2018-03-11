"""
The datastore acts as the bridge between the RPC frontend and the db backend.
"""
import contextlib

from absl import flags
from sqlalchemy import orm

import deeplearning.deepsmith.client
import deeplearning.deepsmith.generator
import deeplearning.deepsmith.harness
import deeplearning.deepsmith.result
import deeplearning.deepsmith.testbed
import deeplearning.deepsmith.testcase
import deeplearning.deepsmith.testcase
import deeplearning.deepsmith.toolchain
from deeplearning.deepsmith import db
from deeplearning.deepsmith.proto import datastore_pb2
from deeplearning.deepsmith.proto import deepsmith_pb2

FLAGS = flags.FLAGS

flags.DEFINE_string('datastore', None, 'Path to Datastore config proto.')


class InvalidRequest(ValueError):
  """Exception raised if request cannot be served."""
  pass


class DataStore(object):
  """ The centralized data store. """

  def __init__(self, config: datastore_pb2.DataStore):
    self._config = config
    self._engine = db.MakeEngine(self._config)
    db.Table.metadata.create_all(self._engine)
    db.Table.metadata.bind = self._engine
    self._make_session = orm.sessionmaker(bind=self._engine)

  @contextlib.contextmanager
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
                      proto: deepsmith_pb2.Testcase) -> None:
    """Record a single Testcase in the database.
    """
    deeplearning.deepsmith.testcase.Testcase.GetOrAdd(session, proto)

  def _BuildTestcaseRequestQuery(self, session, request) -> db.query_t:
    def _FilterToolchainGeneratorHarness(q):
      if request.HasField("toolchain"):
        toolchain = db.GetOrAdd(
            session, deeplearning.deepsmith.toolchain.Toolchain,
            name=request.toolchain
        )
        if not toolchain:
          raise LookupError
        q = q.filter(
            deeplearning.deepsmith.testcase.Testcase.toolchain_id == toolchain.id)

      # Filter by generator.
      if request.HasField("generator"):
        generator = session.query(deeplearning.deepsmith.generator.Generator) \
          .filter(deeplearning.deepsmith.generator.Generator.name == request.generator.name,
                  deeplearning.deepsmith.generator.Generator.version == request.generator.version) \
          .first()
        if not generator:
          raise LookupError
        q = q.filter(deeplearning.deepsmith.testcase.Testcase.generator_id == generator.id)

      # Filter by generator.
      if request.has_harness:
        harness = session.query(deeplearning.deepsmith.harness.Harness) \
          .filter(deeplearning.deepsmith.harness.Harness.name == request.harness.name,
                  deeplearning.deepsmith.harness.Harness.version == request.harness.version) \
          .first()
        if not harness:
          raise LookupError
        q = q.filter(
            deeplearning.deepsmith.testcase.Testcase.harnessid == harness.id)

      return q

    q = session.query(deeplearning.deepsmith.testcase.Testcase)
    q = _FilterToolchainGeneratorHarness(q)

    testbed_id = None
    if request.HasField("testbed"):
      toolchain = db.GetOrAdd(
          session, deeplearning.deepsmith.toolchain.Toolchain,
          name=request.testbed
      )
      testbed = db.GetOrAdd(
          session, deeplearning.deepsmith.testbed.Testbed,
          toolchain=toolchain,
          name=request.testbed.toolchain,
          version=request.testbed.version
      )
      testbed_id = testbed.id

    if testbed_id and not request.include_testcases_with_results:
      q2 = session.query(deeplearning.deepsmith.result.Result.testcase_id) \
        .join(deeplearning.deepsmith.testcase.Testcase) \
        .filter(deeplearning.deepsmith.result.Result.testbed_id == testbed_id)
      q2 = _FilterToolchainGeneratorHarness(q2)
      q = q.filter(~deeplearning.deepsmith.testcase.Testcase.id.in_(q2))

    if testbed_id and not request.include_testcases_with_pending_results:
      q2 = session.query(deeplearning.deepsmith.result.PendingResult.testcase_id) \
        .join(deeplearning.deepsmith.testcase.Testcase) \
        .filter(deeplearning.deepsmith.result.PendingResult.testbed_id == testbed_id)
      q2 = _FilterToolchainGeneratorHarness(q2)
      q = q.filter(~deeplearning.deepsmith.testcase.Testcase.id.in_(q2))

    return q

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
        response.testcases = [testcase.ToProto() for testcase in q]

      if request.return_total_matching_count:
        q2 = self._BuildTestcaseRequestQuery(session, request)
        response.total_matching_count = q2.count()
