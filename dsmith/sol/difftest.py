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
    # break early if we can
    num_results = s.query(func.count(Result.id)).scalar()
    num_metas = s.query(func.count(ResultMeta.id)).scalar()
    if num_results == num_metas:
        return

    print("creating results metas ...")
    s.execute(f"DELETE FROM {ResultMeta.__tablename__}")
    logging.debug("deleted existing result metas")
    testbeds_harnesses = s.query(Result.testbed_id, Testcase.harness)\
        .join(Testcase)\
        .group_by(Result.testbed_id, Testcase.harness)\
        .order_by(Testcase.harness, Result.testbed_id)\
        .all()

    bar = progressbar.ProgressBar(redirect_stdout=True)
    for testbed_id, harness in bar(testbeds_harnesses):
        # FIXME: @cumtime variable is not supported by SQLite.
        s.execute(sql_query(f"""
INSERT INTO {ResultMeta.__tablename__} (id, total_time, cumtime)
SELECT  results.id,
        results.runtime + programs.generation_time AS total_time,
        @cumtime := @cumtime + results.runtime + programs.generation_time AS cumtime
FROM {Result.__tablename__} results
INNER JOIN {Testcase.__tablename__} testcases ON results.testcase_id = testcases.id
INNER JOIN {Program.__tablename__} programs ON testcases.program_id = programs.id
JOIN (SELECT @cumtime := 0) r
WHERE results.testbed_id = {testbed_id}
AND testcases.harness = {harness}
ORDER BY programs.date"""))
        s.commit()
