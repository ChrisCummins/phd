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
import logging

from labm8 import crypto

import dsmith

from dsmith import db
from dsmith import dsmith_pb2 as pb
from dsmith.db_base import get_or_add


def add_testcase(session: db.session_t, testcase_pb: pb.Testcase):
    # Add generator:
    generator = get_or_add(session, db.Generator, generator=testcase_pb.generator)

    # Add input:
    sha1 = crypto.sha1_str(testcase_pb.input)
    input = get_or_add(
        session, db.TestcaseInput, sha1=sha1, input=testcase_pb.input)

    # Add testcase:
    testcase = get_or_add(
        session, db.Testcase, generator=generator, input=input)

    # Add options:
    for opt_ in testcase_pb.opts:
        opt = get_or_add(session, db.TestcaseOpt, opt=opt_)
        get_or_add(session, db.TestcaseOptAssociation, testcase=testcase, opt=opt)

    # Add timings:
    for timings_ in testcase_pb.timings:
        client = get_or_add(session, db.Client, client=timings_.client)
        event = get_or_add(session, db.Event, event=timings_.event, client=client)
        timing = get_or_add(session, db.TestcaseTiming, testcase=testcase,
                            event=event, time=timings_.time)
