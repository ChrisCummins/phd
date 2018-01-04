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
"""
Differential test soldity results.
"""
import threading

import dsmith

from dsmith import Colors
from dsmith.sol.db import *


def difftest():
    with Session() as s:
        create_results_metas(s)


def create_results_metas(s: session_t):
    """
    Create total time and cumulative time for each test case evaluated on each
    testbed using each harness.
    """
    class Worker(threading.Thread):
        """ worker thread to run testcases asynchronously """
        def __init__(self, testbeds_harnesses: List[Tuple['Testbed.id_t', 'Harnesses.column_t']]):
            self.ndone = 0
            self.testbeds_harnesses = testbeds_harnesses
            super(Worker, self).__init__()

        def run(self):
            """ main loop"""
            with Session() as s:
                for testbed_id, harness in self.testbeds_harnesses:
                    self.ndone += 1
                    testbed = s.query(Testbed).filter(Testbed.id == testbed_id).scalar()

                    # FIXME: @cumtime variable is not supported by SQLite.
                    s.execute(f"""
INSERT INTO {ResultMeta.__tablename__} (id, total_time, cumtime)
SELECT  results.id,
        results.runtime + programs.generation_time AS total_time,
        @cumtime := @cumtime + results.runtime + programs.generation_time AS cumtime
FROM {Result.__tablename__} results
INNER JOIN {Testcase.__tablename__} testcases ON results.testcase_id = testcases.id
INNER JOIN {Program.__tablename__} programs ON testcases.program_id = programs.id
JOIN (SELECT @cumtime := 0) r
WHERE results.testbed_id = {testbed.id}
AND testcases.harness = {harness}
ORDER BY programs.date""")
                    s.commit()

    # break early if we can
    num_results = s.query(func.count(Result.id)).scalar()
    num_metas = s.query(func.count(ResultMeta.id)).scalar()
    if num_results == num_metas:
        return

    print("creating results metas ...")
    s.execute(f"DELETE FROM {ResultMeta.__tablename__}")
    testbeds_harnesses = s.query(Result.testbed_id, Testcase.harness)\
        .join(Testcase)\
        .group_by(Result.testbed_id, Testcase.harness)\
        .order_by(Testcase.harness, Result.testbed_id)\
        .all()

    bar = progressbar.ProgressBar(initial_value=0,
                                  max_value=len(testbeds_harnesses),
                                  redirect_stdout=True)
    worker = Worker(testbeds_harnesses)
    worker.start()
    while worker.is_alive():
        bar.update(min(worker.ndone, len(testbeds_harnesses)))
        worker.join(0.5)
