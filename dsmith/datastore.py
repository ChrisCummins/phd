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

from contextlib import contextmanager
from labm8 import crypto
from typing import List
from sqlalchemy.orm import sessionmaker

import dsmith

from dsmith import db
from dsmith import dbutil
from dsmith import dsmith_pb2 as pb


class DataStore(object):
    """ The centralized data store. """
    def __init__(self, **db_opts):
        self.opts = db_opts
        self._engine, _ = dbutil.make_engine(**self.opts)
        db.Base.metadata.create_all(self._engine)
        db.Base.metadata.bind = self._engine
        self._make_session = sessionmaker(bind=self._engine)

    @contextmanager
    def session(self, commit: bool=False) -> db.session_t:
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

    def add_testcases(self, testcases: List[pb.Testcase]) -> None:
        """
        Add a sequence of testcases to the datastore.

        TODO: Optimize to reduce the number of SQL queries by batching lookups/inserts.
        """
        with self.session(commit=True) as session:
            for testcase in testcases:
                self._add_one_testcase(session, testcase)

    def _add_one_testcase(self, session: db.session_t, testcase_pb: pb.Testcase) -> None:
        # Add generator:
        generator = dbutil.get_or_add(session, db.Generator, generator=testcase_pb.generator)

        # Add input:
        sha1 = crypto.sha1_str(testcase_pb.input)
        input = dbutil.get_or_add(session, db.TestcaseInput, sha1=sha1, input=testcase_pb.input)

        # Add testcase:
        testcase = dbutil.get_or_add(session, db.Testcase, generator=generator, input=input)

        # Add options:
        for opt_ in testcase_pb.opts:
            opt = dbutil.get_or_add(session, db.TestcaseOpt, opt=opt_)
            dbutil.get_or_add(session, db.TestcaseOptAssociation, testcase=testcase, opt=opt)

        # Add timings:
        for timings_ in testcase_pb.timings:
            client = dbutil.get_or_add(session, db.Client, client=timings_.client)
            event = dbutil.get_or_add(session, db.Event, event=timings_.event, client=client)
            timing = dbutil.get_or_add(session, db.TestcaseTiming, testcase=testcase,
                                       event=event, time=timings_.time)
