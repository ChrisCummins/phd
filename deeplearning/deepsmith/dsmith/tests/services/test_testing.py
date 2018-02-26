#
# Copyright 2017, 2018 Chris Cummins <chrisc.101@gmail.com>.
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
import pytest
import random
import string

from tempfile import TemporaryDirectory

from dsmith import test as tests
from dsmith import datastore
from dsmith import db
from dsmith import dsmith_pb2 as pb
from dsmith.services import testing as testing_service


@pytest.fixture
def ds():
    with TemporaryDirectory(prefix="dsmith-test-db-") as tmpdir:
        yield datastore.DataStore(engine="sqlite", db_dir=tmpdir)


def random_generator():
    return pb.Generator(
        name=random.choice(["foo", "bar", "baz"]),
        version=random.choice(["1", "1", "1", "2"]),
    )


def random_harness():
    return pb.Harness(
        name=random.choice(["a", "b", "c"]),
        version=random.choice(["1", "1", "1", "2"]),
    )


def random_input():
    return ''.join(random.choices(string.ascii_uppercase + string.digits,
                                  k=int(random.random() * 1000) + 1))


def random_inputs():
    return [pb.NamedText(name="src", text=random_input())]


def random_opt():
    return "foo=(bar)"


def random_opts():
    return [random_opt()] * (int(random.random() * 5) + 1)


def random_testcase():
    return pb.Testcase(
        generator=random_generator(),
        harness=random_harness(),
        inputs=random_inputs(),
        opts=random_opts(),
        timings=[
            pb.Timing(client="c", event="a", duration=1),
            pb.Timing(client="c", event="b", duration=2),
            pb.Timing(client="c", event="c", duration=3),
        ],
    )


def test_service_empty(ds):
    service = testing_service.TestingService(ds)
    request = pb.SubmitTestcasesRequest(testcases=[])
    response = service.SubmitTestcases(request, None)
    assert type(response) == pb.SubmitTestcasesResponse

    with ds.session() as s:
        assert s.query(db.Client).count() == 0
        assert s.query(db.Event).count() == 0
        assert s.query(db.Testcase).count() == 0
        assert s.query(db.TestcaseInput).count() == 0
        assert s.query(db.TestcaseOpt).count() == 0
        assert s.query(db.TestcaseTiming).count() == 0


def test_service_add_one(ds):
    service = testing_service.TestingService(ds)
    testcases = [
        pb.Testcase(
            generator=pb.Generator(name="foo", version="foo"),
            harness=pb.Harness(name="foo", version="bar"),
            inputs=[pb.NamedText(name="src", text="foo")],
            opts=["1", "2", "3"],
            timings=[
                pb.Timing(client="c", event="a", duration=1),
                pb.Timing(client="c", event="b", duration=2),
                pb.Timing(client="c", event="c", duration=3),
            ],
        ),
    ]
    request = pb.SubmitTestcasesRequest(testcases=testcases)
    service.SubmitTestcases(request, None)

    with ds.session() as s:
        assert s.query(db.Client).count() == 1
        assert s.query(db.Event).count() == 3
        assert s.query(db.Testcase).count() == 1
        assert s.query(db.TestcaseInput).count() == 1
        assert s.query(db.TestcaseOpt).count() == 3
        assert s.query(db.TestcaseTiming).count() == 3


def test_service_add_two(ds):
    service = testing_service.TestingService(ds)
    testcases = [
        pb.Testcase(
            generator=pb.Generator(name="foo", version="foo"),
            harness=pb.Harness(name="foo", version="bar"),
            inputs=[pb.NamedText(name="src", text="foo")],
            opts=["1", "2", "3"],
            timings=[
                pb.Timing(client="c", event="a", duration=1),
                pb.Timing(client="c", event="b", duration=2),
                pb.Timing(client="c", event="c", duration=3),
            ],
        ),
        pb.Testcase(
            generator=pb.Generator(name="bar", version="foo"),
            harness=pb.Harness(name="foo", version="bar"),
            inputs=[pb.NamedText(name="src", text="abc")],
            opts=["1", "2", "3", "4"],
            timings=[
                pb.Timing(client="b", event="a", duration=1),
                pb.Timing(client="c", event="d", duration=2),
                pb.Timing(client="c", event="c", duration=3),
            ],
        ),
    ]
    request = pb.SubmitTestcasesRequest(testcases=testcases)
    service.SubmitTestcases(request, None)

    with ds.session() as s:
        assert s.query(db.Client).count() == 2
        assert s.query(db.Event).count() == 4
        assert s.query(db.Testcase).count() == 2
        assert s.query(db.TestcaseInput).count() == 2
        assert s.query(db.TestcaseOpt).count() == 4
        assert s.query(db.TestcaseTiming).count() == 6


def serve_request(service, request):
    service.SubmitTestcases(request, None)


def test_add_one(ds, benchmark):
    service = testing_service.TestingService(ds)
    request = pb.SubmitTestcasesRequest(testcases=[random_testcase()])
    benchmark(serve_request, service, request)


def test_add_many(ds, benchmark):
    service = testing_service.TestingService(ds)
    testcases = []
    for _ in range(1, 100):
        testcases.append(random_testcase())
    request = pb.SubmitTestcasesRequest(testcases=testcases)
    benchmark(serve_request, service, request)
