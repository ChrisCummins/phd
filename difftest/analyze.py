import math
import sys
import sqlalchemy as sql

from collections import Counter
from signal import Signals

import db
import util
from db import *

OUTCOMES = [
    "bf",    # build failure
    "bc",    # build crash
    "bto",   # compilation timeout
    "c",     # runtime crash (after compilation)
    "to",    # timeout (after compilation)
    "pass",  # execution completed with zero returncode
    "fail",  # misc testing error - null result
]


CLSMITH_CLASSIFICATIONS = [
    "w",     # program computes result which disagrees with majority
    "bf",    # compilation fails
    "c",     # program exists non-zero returncode
    "to",    # program killed due to timeout
    "pass",  # program computes result which agrees with majority
    "fail",  # misc testing error - null result
]


OUR_CLASSIFICATIONS = [
    "w",     # program computes result wich disagrees with majority
    "bf",    # minority compilation fails
    "bc",    # compiler crashed
    "c",     # minority runtime crash
    "to",    # minority timeout
    "pass",  # same outcome as majority: {pass,bf,c,to}
    "fail",  # misc testing error - null result
]


def get_majority(lst):
    """ get the majority value of the list elements, and the majority count """
    return Counter(lst).most_common(1)[0]


def get_majority_output(session, tables, result):
    q = session.query(tables.results.stdout)\
        .filter(tables.results.program_id == result.program.id,
                tables.results.params_id == result.params.id)
    majority_output, majority_count = get_majority([r[0] for r in q])
    return majority_output


def get_cl_launcher_outcome(result) -> None:
    """
    Given a cl_launcher result, determine and set it's outcome.

    See OUTCOMES for list of possible outcomes.
    """
    def crash_or_build_failure():
        return "c" if "Compilation terminated successfully..." in result.stderr else "bf"
    def crash_or_build_crash():
        return "c" if "Compilation terminated successfully..." in result.stderr else "bc"
    def timeout_or_build_timeout():
        return "to" if "Compilation terminated successfully..." in result.stderr else "bto"

    if result.status == 0:
        return "pass"
    # 139 is SIGSEV
    elif result.status == 139 or result.status == -11:
        result.status = 139
        return crash_or_build_crash()
    # Preproccessor or Unicode error
    elif result.status == 1024 or result.status == 1025:
        return "fail"
    # SIGTRAP
    elif result.status == -5:
        return crash_or_build_crash()
    # SIGKILL
    elif result.status == -9 and result.runtime >= 60:
        return timeout_or_build_timeout()
    # SIGILL
    elif result.status == -4:
        return crash_or_build_crash()
    # SIGABRT
    elif result.status == -6:
        return crash_or_build_crash()
    # SIGFPE
    elif result.status == -8:
        return crash_or_build_crash()
    # SIGBUS
    elif result.status == -7:
        return crash_or_build_crash()
    # cl_launcher error
    elif result.status == 1:
        return crash_or_build_failure()
    else:
        print(result)
        try:
            print(Signals(-result.status).name)
        except ValueError:
            print(result.status)
        raise LookupError(f"failed to determine outcome of cl_launcher {result.id}")


def set_cl_launcher_outcomes(session, results_table, rerun: bool=False) -> None:
    """ Set all cl_launcher outcomes. Set `rerun' to recompute outcomes for all results """
    print("Determining CLSmith outcomes ...")
    q = session.query(results_table)
    if not rerun:
        q = q.filter(results_table.outcome == None)
    ntodo = q.count()
    for result in util.NamedProgressBar('cl_launcher outcomes')(q, max_value=ntodo):
        result.outcome = get_cl_launcher_outcome(result)


def get_cldrive_outcome(result):
    """
    Given a cldrive result, determine and set it's outcome.

    See OUTCOMES for list of possible outcomes.
    """
    def crash_or_build_failure():
        return "c" if "[cldrive] Kernel: " in result.stderr else "bf"
    def crash_or_build_crash():
        return "c" if "[cldrive] Kernel: " in result.stderr else "bc"
    def timeout_or_build_timeout():
        return "to" if "[cldrive] Kernel: " in result.stderr else "bto"

    if result.status == 0:
        return "pass"
    # 401 is bad harness
    elif result.status == 401:
        return "fail"
    # 139 is SIGSEV
    elif result.status == 139 or result.status == -11:
        result.status = 139
        return crash_or_build_crash()
    # SIGTRAP
    elif result.status == -5:
        return crash_or_build_crash()
    # SIGKILL
    elif result.status == -9 and result.runtime >= 60:
        return timeout_or_build_timeout()
    # SIGILL
    elif result.status == -4:
        return crash_or_build_crash()
    # SIGFPE
    elif result.status == -8:
        return crash_or_build_crash()
    # SIGBUS
    elif result.status == -7:
        return crash_or_build_crash()
    # SIGABRT
    elif result.status == -6:
        return crash_or_build_crash()
    # cl_launcher error
    elif result.status == 1:
        if 'cldrive.driver.Timeout: 60' in result.stderr:
            return timeout_or_build_timeout()
        else:
            return crash_or_build_failure()
    # file not found (check the stderr on this one):
    elif result.status == 127:
        return crash_or_build_failure()
    else:
        print(result)
        try:
            print(Signals(-result.status).name)
        except ValueError:
            print(result.status)
        raise LookupError(f"failed to determine outcome of cldrive result #{result.id}")


def set_cldrive_outcomes(session, results_table, rerun: bool=False) -> None:
    """ Set all cldrive outcomes. Set `rerun' to recompute outcomes for all results """
    print("Determining CLgen outcomes ...")
    q = session.query(results_table)
    if not rerun:
        q = q.filter(results_table.outcome == None)
    ntodo = q.count()
    for result in util.NamedProgressBar('cldrive outcomes')(q, max_value=ntodo):
        result.outcome = get_cldrive_outcome(result)


def set_clsmith_classifications(session, results_table, params_table,
                                programs_table, rerun: bool=True) -> None:
    """
    Run results classification algorithm of paper:

        Lidbury, C., Lascu, A., Chong, N., & Donaldson, A. (2015). Many-Core
        Compiler Fuzzing. In PLDI. https://doi.org/10.1145/2737924.2737986

    Requires that result outcomes have been computed.

    Set `rerun' to recompute classifications for all results. You must do this
    whenver changing classification algorithm, or when new results are added, as
    they may change existing outcomes.
    """
    q = session.query(results_table)
    tablename = results_table.__tablename__

    # reset any existing classifications
    if rerun:
        print(f"Reseting {tablename} classifications ...")
        session.query(results_table).update({"classification": None})

    # direct mappings from outcome to classification
    print(f"Classifying {tablename} timeouts ...")
    session.query(results_table)\
        .filter(sql.or_(results_table.outcome == "to",
                        results_table.outcome == "bto"))\
        .update({"classification": "to"})
    print(f"Classifying {tablename} build failures ...")
    session.query(results_table)\
        .filter(sql.or_(results_table.outcome == "bf",
                        results_table.outcome == "bc"))\
        .update({"classification": "bf"})
    print(f"Classifying {tablename} crashes ...")
    session.query(results_table)\
        .filter(results_table.outcome == "c")\
        .update({"classification": "c"})
    print(f"Classifying {tablename} test failures ...")
    session.query(results_table)\
        .filter(results_table.outcome == "fail")\
        .update({"classification": "fail"})

    # Go program-by-program, looking for wrong-code outputs
    ok = session.query(results_table.program_id).filter(
        results_table.outcome == "pass").distinct()
    q = session.query(programs_table).filter(programs_table.id.in_(ok))
    for program in util.NamedProgressBar('classify')(q, max_value=q.count()):
        # treat param combinations independently
        # TODO: iterate over pairs of opt on/off params
        for params in session.query(params_table):
            # select all results for this test case
            q = session.query(results_table)\
                .filter(results_table.program_id == program.id,
                        results_table.params_id == params.id,
                        results_table.outcome == "pass")

            if q.count() <= 3:
                # Too few results for a majority, so everything passed.
                for result in q:
                    result.classification = "pass"
            else:
                # Determine the majority output, and majority size.
                majority_output, majority_count = get_majority([r.stdout for r in q])

                if majority_count < 3:
                    # No majority, so everything passed.
                    for result in q:
                        result.classification = "pass"
                else:
                    # There is a majority conensus, so compare individual
                    # outputs to majority
                    for result in q:
                        if result.stdout == majority_output:
                            result.classification = "pass"
                        else:
                            result.classification = "w"


def set_our_classifications(session, results_table, params_table,
                            programs_table, rerun: bool=True) -> None:
    """
    Our methodology for classifying results.
    """
    q = session.query(results_table)
    tablename = results_table.__tablename__

    # reset any existing classifications
    if rerun:
        print(f"Reseting {tablename} classifications ...")
        session.query(results_table).update({"classification": "pass"})

    # Go program-by-program, looking for wrong-code outputs
    q = session.query(results_table.program_id).distinct()
    for row in util.NamedProgressBar('classify')(q, max_value=q.count()):
        program_id = row[0]
        # treat param combinations independently
        for params in session.query(params_table):
            # select all results for this test case
            q = session.query(results_table)\
                .filter(results_table.program_id == program_id,
                        results_table.params_id == params.id)
            n = q.count()

            if n < 6:
                # Too few results for a majority
                continue

            min_majority_count = math.ceil(n / 2)
            majority_outcome, majority_count = get_majority([r.outcome for r in q])
            if majority_outcome != "bf":
                q.filter(results_table.outcome == "bf")\
                    .update({"classification": "bf"})
            if majority_outcome != "c":
                q.filter(results_table.outcome == "c")\
                    .update({"classification": "c"})
            if majority_outcome != "to":
                q.filter(results_table.outcome == "to")\
                    .update({"classification": "to"})

            # Look for wrong-code bugs
            q2 = q.filter(results_table.outcome == "pass")
            n2 = q2.count()
            if n2 < 6:
                # Too few results for a majority
                continue

            majority_output, output_majority_count = get_majority(
                [r.stdout for r in q2])

            min_output_majority_count = max(n2 - 1, 7)
            # min_output_majority_count = math.ceil(n2 / 2)
            if output_majority_count < min_output_majority_count:
                # No majority
                print("output_majority_count <", min_output_majority_count, " = ", output_majority_count)
                continue

            # There is a majority conensus, so compare individual
            # outputs to majority
            q2.filter(results_table.stdout != majority_output)\
                .update({"classification": "w"})


set_classifications = set_our_classifications
