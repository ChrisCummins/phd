#
# Copyright 2017 Chris Cummins <chrisc.101@gmail.com>.
#
# This file is part of DeepSmith.
#
# DeepSmith is free software: you can redistribute it and/or modify it under the
# terms of the GNU General Public License as published by the Free Software
# Foundation, either version 3 of the License, or (at your option) any later
# version.
#
# DeepSmith is distributed in the hope that it will be useful, but WITHOUT ANY
# WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR
# A PARTICULAR PURPOSE.  See the GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License along with
# DeepSmith.  If not, see <http://www.gnu.org/licenses/>.
#
from contextlib import contextmanager
from tempfile import TemporaryDirectory

from dsmith import test as tests
from dsmith import db
from dsmith import dsmith_pb2 as pb
from dsmith import server


@contextmanager
def TestSession():
    """Provide a transactional scope around a series of operations."""
    with TemporaryDirectory(prefix="dsmith-test-", suffix=".db") as tmpdir:
        with db.Session(db.init(engine="sqlite", db_dir=tmpdir)) as session:
            yield session


def test_service_empty():
    service = server.TestingService()

    request = pb.SubmitTestcasesRequest(testcases=[])
    response = service.SubmitTestcases(request, None)
    assert type(response) == pb.SubmitTestcasesResponse

    with TestSession() as s:
        assert s.query(db.Client).count() == 0
        assert s.query(db.Event).count() == 0
        assert s.query(db.Testcase).count() == 0
        assert s.query(db.TestcaseInput).count() == 0
        assert s.query(db.TestcaseOpt).count() == 0
        assert s.query(db.TestcaseTiming).count() == 0


def test_service_add_one():
    service = server.TestingService()
    testcases = [
        pb.Testcase(
            generator="foo",
            input="input",
            opts=["1", "2", "3"],
            timings=[
                pb.Timing(client="c", event="a", time=1),
                pb.Timing(client="c", event="b", time=2),
                pb.Timing(client="c", event="c", time=3),
            ],
        ),
    ]
    request = pb.SubmitTestcasesRequest(testcases=testcases)
    service.SubmitTestcases(request, None)

    with TestSession() as s:
        assert s.query(db.Client).count() == 1
        assert s.query(db.Event).count() == 3
        assert s.query(db.Testcase).count() == 1
        assert s.query(db.TestcaseInput).count() == 1
        assert s.query(db.TestcaseOpt).count() == 3
        assert s.query(db.TestcaseTiming).count() == 3


def test_service_add_two():
    service = server.TestingService()
    testcases = [
        pb.Testcase(
            generator="foo",
            input="input",
            opts=["1", "2", "3"],
            timings=[
                pb.Timing(client="c", event="a", time=1),
                pb.Timing(client="c", event="b", time=2),
                pb.Timing(client="c", event="c", time=3),
            ],
        ),
        pb.Testcase(
            generator="bar",
            input="abc",
            opts=["1", "2", "3", "4"],
            timings=[
                pb.Timing(client="b", event="a", time=1),
                pb.Timing(client="c", event="b", time=2),
                pb.Timing(client="c", event="c", time=3),
            ],
        ),
    ]
    request = pb.SubmitTestcasesRequest(testcases=testcases)
    service.SubmitTestcases(request, None)

    with TestSession() as s:
        assert s.query(db.Client).count() == 2
        assert s.query(db.Event).count() == 4
        assert s.query(db.Testcase).count() == 2
        assert s.query(db.TestcaseInput).count() == 2
        assert s.query(db.TestcaseOpt).count() == 4
        assert s.query(db.TestcaseTiming).count() == 6


def test_service_add_duplicate():
    service = server.TestingService()
    testcases = [
        pb.Testcase(generator="foo", input="input"),
        pb.Testcase(generator="foo", input="input"),
    ]
    request = pb.SubmitTestcasesRequest(testcases=testcases)
    service.SubmitTestcases(request, None)

    with TestSession() as s:
        assert s.query(db.Testcase).count() == 1
