"""
The datastore acts as the bridge between the RPC frontend and the db backend.
"""
import contextlib
import datetime

from absl import flags
from sqlalchemy import orm

from labm8 import crypto

import deeplearning.deepsmith.client
import deeplearning.deepsmith.generator
import deeplearning.deepsmith.harness
import deeplearning.deepsmith.toolchain
import deeplearning.deepsmith.result
import deeplearning.deepsmith.testbed
import deeplearning.deepsmith.testcase

from deeplearning.deepsmith import db
from deeplearning.deepsmith import profiling_event
from deeplearning.deepsmith.protos import deepsmith_pb2

FLAGS = flags.FLAGS


class InvalidRequest(ValueError):
  """Exception raised if request cannot be served."""
  pass


class DataStore(object):
  """ The centralized data store. """

  def __init__(self, **db_opts):
    self.opts = db_opts
    self._engine, _ = db.MakeEngine(**self.opts)
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
                      testcase_proto: deepsmith_pb2.Testcase) -> None:
    """Record a single Testcase in the database.
    """
    # Add toolchain:
    toolchain = db.GetOrAdd(
        session, deeplearning.deepsmith.toolchain.Toolchain,
        name=testcase_proto.toolchain,
    )

    # Add generator:
    generator = db.GetOrAdd(
        session, deeplearning.deepsmith.generator.Generator,
        name=testcase_proto.generator.name,
        version=testcase_proto.generator.name,
    )

    # Add harness:
    harness = db.GetOrAdd(
        session, deeplearning.deepsmith.harness.Harness,
        name=testcase_proto.harness.name,
        version=testcase_proto.harness.version,
    )

    # Add testcase:
    testcase = db.GetOrAdd(
        session, deeplearning.deepsmith.testcase.Testcase,
        toolchain=toolchain,
        generator=generator,
        harness=harness,
    )

    # Add inputs:
    for input_pb in testcase_proto.inputs:
      text = testcase_proto.inputs[input_pb]
      name = db.GetOrAdd(
          session, deeplearning.deepsmith.testcase.TestcaseInputName,
          name=input_pb,
      )
      sha1 = crypto.sha1_str(text)

      input = db.GetOrAdd(
          session, deeplearning.deepsmith.testcase.TestcaseInput,
          name=name,
          sha1=sha1,
          linecount=len(text.split("\n")),
          charcount=len(text),
          input=text,
      )
      db.GetOrAdd(
          session, deeplearning.deepsmith.testcase.TestcaseInputAssociation,
          testcase=testcase,
          input=input,
      )

    # TODO(cec): If the testcase already exists, don't add timings.

    # Add timings:
    for timing in testcase_proto.timings:
      client_ = db.GetOrAdd(
          session, deeplearning.deepsmith.client.Client,
          name=timing.client
      )
      profiling_event_name = db.GetOrAdd(
          session, profiling_event.ProfilingEventName,
          name=timing.name
      )
      timing = db.GetOrAdd(
          session, profiling_event.TestcaseTiming,
          testcase=testcase,
          name=profiling_event_name,
          client=client_,
          duration_seconds=timing.duration_seconds,
          date=datetime.datetime.fromtimestamp(timing.date_epoch_seconds)
      )

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

  def _HarnessObjectToProto(self, harness: deeplearning.deepsmith.harness.Harness) -> deepsmith_pb2.Harness:
    return deepsmith_pb2.Harness(name=harness.name,
                                 version=harness.version)

  def _TestcaseObjectToProto(self, testcase: deeplearning.deepsmith.testcase.Testcase) -> deepsmith_pb2.Testcase:
    return deepsmith_pb2.Testcase(
        toolchain=testcase.toolchain,
        generator=testcase.generator.ToProto(),
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
