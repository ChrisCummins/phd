#!/usr/bin/env python

import clgen
import math
import sys
import sqlalchemy as sql

from argparse import ArgumentParser
from collections import Counter
from signal import Signals
from progressbar import ProgressBar

import db
import oclgrind
import util
from db import *


def get_majority(lst):
    """ get the majority value of the list elements, and the majority count """
    return Counter(lst).most_common(1)[0]


def get_majority_output(session, tables: Tableset, result):
    q = session.query(tables.results.stdout)\
        .filter(tables.results.program_id == result.program.id,
                tables.results.params_id == result.params.id)
    outputs = [r[0] for r in q]
    majority_output, majority_count = get_majority(outputs)
    return majority_output, majority_count, len(outputs)


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


def set_cl_launcher_outcomes(session, tables: Tableset, rerun: bool=False) -> None:
    """ Set all cl_launcher outcomes. Set `rerun' to recompute outcomes for all results """
    print("Determining CLSmith outcomes ...")
    q = session.query(tables.results)
    if not rerun:
        q = q.filter(tables.results.outcome == None)
    ntodo = q.count()
    for result in util.NamedProgressBar('cl_launcher outcomes')(q, max_value=ntodo):
        result.outcome = get_cl_launcher_outcome(result)


def get_cldrive_outcome(result):
    """
    Given a cldrive result, determine its outcome.

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
    elif result.status == -9:
        print(f"SIGKILL, but only ran for {result.runtime:.2f}s")
        return crash_or_build_crash()
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


def set_cldrive_outcomes(session, tables: Tableset, rerun: bool=False) -> None:
    """ Set all cldrive outcomes. Set `rerun' to recompute outcomes for all results """
    print("Determining CLgen outcomes ...")
    q = session.query(tables.results)
    if not rerun:
        q = q.filter(tables.results.outcome == None)
    ntodo = q.count()
    for result in util.NamedProgressBar('cldrive outcomes')(q, max_value=ntodo):
        result.outcome = get_cldrive_outcome(result)


def set_clsmith_classifications(session, tables: Tableset, rerun: bool=True) -> None:
    """
    Run results classification algorithm of paper:

        Lidbury, C., Lascu, A., Chong, N., & Donaldson, A. (2015). Many-Core
        Compiler Fuzzing. In PLDI. https://doi.org/10.1145/2737924.2737986

    Requires that result outcomes have been computed.

    Set `rerun' to recompute classifications for all results. You must do this
    whenver changing classification algorithm, or when new results are added, as
    they may change existing outcomes.
    """
    # TODO: Update to meta table layout
    q = session.query(tables.results)

    # reset any existing classifications
    if rerun:
        print(f"Reseting {tables.name} classifications ...")
        session.query(tables.results).update({"classification": None})

    # direct mappings from outcome to classification
    print(f"Classifying {tables.name} timeouts ...")
    session.query(tables.results)\
        .filter(sql.or_(tables.results.outcome == "to",
                        tables.results.outcome == "bto"))\
        .update({"classification": "to"})
    print(f"Classifying {tables.name} build failures ...")
    session.query(tables.results)\
        .filter(sql.or_(tables.results.outcome == "bf",
                        tables.results.outcome == "bc"))\
        .update({"classification": "bf"})
    print(f"Classifying {tables.name} crashes ...")
    session.query(tables.results)\
        .filter(tables.results.outcome == "c")\
        .update({"classification": "c"})
    print(f"Classifying {tables.name} test failures ...")
    session.query(tables.results)\
        .filter(tables.results.outcome == "fail")\
        .update({"classification": "fail"})

    # Go program-by-program, looking for wrong-code outputs
    ok = session.query(tables.results.program_id).filter(
        tables.results.outcome == "pass").distinct()
    q = session.query(tables.programs).filter(tables.programs.id.in_(ok))
    for program in util.NamedProgressBar('classify')(q, max_value=q.count()):
        # treat param combinations independently
        # TODO: iterate over pairs of opt on/off params
        for params in session.query(tables.params):
            # select all results for this test case
            q = session.query(tables.results)\
                .filter(tables.results.program_id == program.id,
                        tables.results.params_id == params.id,
                        tables.results.outcome == "pass")

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


def set_our_classifications(session, tables: Tableset, rerun: bool=True) -> None:
    """
    Our methodology for classifying results.
    """
    q = session.query(tables.results)

    all_params = [x[0] for x in session.query(tables.params.id).all()]

    # Go program-by-program:
    # programs = session.query(tables.results.program_id).join(tables.meta).distinct().all()
    programs = session.query(tables.programs.id).all()
    for i, (program_id,) in enumerate(util.NamedProgressBar("classify")(programs)):
        # Reset existing classifications
        session.query(tables.results)\
            .filter(tables.results.program_id == program_id)\
            .update({"classification": "pass"})

        program_ok = None

        # Treat each param combination independently:
        for params_id in all_params:
            # Select all non-bc results for this test case:
            q = session.query(tables.results)\
                .filter(tables.results.program_id == program_id,
                        tables.results.params_id == params_id,
                        tables.results.outcome != "bc")

            # Check that there are enough non-bc results for a majority:
            n = session.query(sql.sql.func.count(tables.results.id))\
                .filter(tables.results.program_id == program_id,
                        tables.results.params_id == params_id,
                        tables.results.outcome != "bc").scalar() or 0
            if n < 6:
                continue

            # Determine majority outcome:
            min_majority_count = math.ceil(n / 2)
            majority_outcome, majority_count = get_majority([r.outcome for r in q])

            if majority_count < min_majority_count:
                continue

            # # If majority outcome resulted in binaries, mark anomalous build
            # # failures:
            # if majority_outcome != "bf":
            #     q.filter(tables.results.outcome == "bf")\
            #         .update({"classification": "bf"})

            # # If majority outcome did not crash, mark anomalous crashes:
            # if majority_outcome != "c":
            #     q.filter(tables.results.outcome == "c")\
            #         .update({"classification": "c"})

            # # If majority outcome did not timeout, mark anomalous timeouts:
            # if majority_outcome != "to":
            #     q.filter(tables.results.outcome == "to")\
            #         .update({"classification": "to"})

            # Look for wrong-code bugs:
            #
            # If the majority did not produce outputs, then we're done:
            if majority_outcome != "pass":
                continue

            # Pruning of programs whose outputs should not be difftested:
            if tables.name == "CLgen":
                # If we haven't checked the program yet, do so now:
                if program_ok == None:
                    program_ok = True

                    program = session.query(tables.programs)\
                            .filter(tables.programs.id == program_id).first()

                    # Run GPUverify on kernel
                    if program.gpuverified == None:
                        try:
                            clgen.gpuverify(program.src, ["--local_size=64", "--num_groups=128"])
                            program.gpuverified = 1
                        except clgen.GPUVerifyException:
                            program.gpuverified = 0

                    if not program.gpuverified:
                        print("skipping kernel which failed GPUVerify")
                        program_ok = False
                    if "float" in program.src:
                        print("skipping floating point kernel")
                        program_ok = False

                # If program can be skipped, do so:
                if not program_ok:
                    continue

                # TODO: Run oclgrind on test harness

            # Get "pass" outcome results:
            q2 = q.filter(tables.results.outcome == "pass")
            n2 = q2.count()

            majority_output, output_majority_count = get_majority([r.stdout for r in q2])

            # Ensure that the majority of configurations agree on the output:
            # min_output_majority_count = n2 - 1
            min_output_majority_count = math.ceil(n2 / 2)
            if output_majority_count < min_output_majority_count:
                # No majority
                print("skipping output_majority_count <", min_output_majority_count, " = ", output_majority_count)
                continue

            # There is a majority conensus, so compare individual
            # outputs to majority
            q2.filter(tables.results.stdout != majority_output)\
                .update({"classification": "w"})

        if not i % 100:
            session.commit()

    session.commit()


def verify_clgen_w_result(session: session_t, result: CLgenResult) -> None:
    print(f"Verifying CLgen w-result {result.id} ...")

    def fail():
        result.classification = "pass"
        session.commit()

    if "float" in result.program.src:
        print(f"retracted CLgen w-result {result.id}: contains float")
        return fail()

    # Run GPUverify on kernel
    if result.program.gpuverified == None:
        try:
            clgen.gpuverify(program.src, ["--local_size=64", "--num_groups=128"])
            program.gpuverified = 1
        except clgen.GPUVerifyException:
            program.gpuverified = 0
        session.commit()

    if not result.program.gpuverified:
        print(f"retracted CLgen w-result {result.id}: failed GPUVerify")
        return fail()

    harness = session.query(CLgenHarness)\
                .filter(CLgenHarness.program_id == result.program_id,
                        CLgenHarness.params_id == result.params_id)\
                .first()

    if harness.oclverified == None:
        harness.oclverified = oclgrind.oclgrind_verify_clgen(harness)
        session.commit()

    if not harness.oclverified:
        print(f"retracted CLgen w-result {result.id}: failed OCLgrind verification")
        return fail()

    # majority_output, majority_count, count = get_majority_output(
    #     session, CLGEN_TABLES, result)
    # if majority_count < count - 1:
    #     print(f"retracting CLgen w-result {result.id}: not a large enough majority (only {majority_count} of {count} agree)")
    #     return fail()


def verify_clsmith_w_result(session: session_t, result: CLgenResult) -> None:
    verified = oclgrind.oclgrind_verify_clsmith(result)

    if not verified:
        print(f"retracted CLSmith w-result {result.id}: failed OCLgrind verification")
        result.classification = "pass"
        session.commit()


def set_throws_warnings(session: session_t, tables: Tableset, program_id: int):
    """
    Determine if the program produces relevant compiler warnings, and if so,
    mark it as such, and remove the wrong-code classification from any results.
    """
    def fail():
        program = session.query(tables.programs)\
            .filter(tables.programs.id == program_id).first()

        # Mark program as throwing warnings:
        program.throws_warnings = True
        # Mark any wrong-code classifications for this program as passes:
        session.query(tables.results)\
            .filter(tables.results.program_id == program.id,
                    tables.results.classification == "w")\
            .update({"classification": "pass"})
        session.commit()

    print(f"Checking program {program_id} for compiler warnings")
    stderrs = session.query(tables.results.stderr)\
        .filter(tables.programs.id == program_id)

    for stderr in stderrs:
        if "incompatible pointer to integer conversion" in stderr:
            print(f"marking program {program_id} as throws errors: incompatible pointer to to integer conversion")
            return fail()

        if "ordered comparison between pointer and integer" in stderr:
            print(f"marking program {program_id} as throws errors: ordered comparison between pointer and integer")
            return fail()

        if "warning: incompatible" in stderr:
            print(f"marking program {program_id} as throws errors: incompatible")
            return fail()

        if "warning" in stderr:
            print("WARNINGS:")
            print("\n".join(f">> {line}" for line in stderr.split("\n")))
            # break


def prune_w_classifications(session: session_t, tables: Tableset, _):
    # Verify existing w-classifications:
    #
    q = session.query(tables.results.id)\
        .filter(tables.results.classification == "w")

    for i, (result_id,) in enumerate(q):
        result = session.query(tables.results)\
            .filter(tables.results.id == result_id).first()
        if tables.name == "CLgen":
            verify_clgen_w_result(session, result)
        elif tables.name == "CLSmith":
            verify_clsmith_w_result(session, result)

    # Check if programs riase compiler warnings:
    #
    program_ids = session.query(tables.results.program_id)\
        .join(tables.programs)\
        .filter(tables.results.classification == "w",
                tables.programs.throws_warnings == None)\
        .distinct().all()

    for program_id, in ProgressBar()(program_ids):
        set_throws_warnings(session, tables, program_id)

    # Check for ADL errors:
    #
    q = session.query(tables.results)\
        .filter(tables.results.testbed_id == 20,
                tables.classification == "w")


def get_classifications(session, tables: Tableset) -> None:
    """
    Our methodology for classifying results.
    """
    # Go testcase-by-testcase:
    num_results = session.query(sql.sql.func.count(tables.results.id)).scalar()
    num_classifications = session.query(sql.sql.func.count(tables.classifications.id)).scalar()

    # Nothing to do
    if num_classifications == num_results:
        return

    print(f"Resetting {tables.name} classifications ...")
    session.execute(f"DELETE FROM {tables.classifications.__tablename__}")
    session.commit()

    min_majority_outcome = 6
    print(f"Classifying testcases with majority outcomes smaller than {min_majority_outcome} as passes ...")
    session.execute(f"""
INSERT INTO {tables.classifications.__tablename__} (id, classification)
SELECT id, {CLASSIFICATIONS_TO_INT['pass']}
FROM {tables.results.__tablename__}
WHERE testcase_id IN
    (SELECT testcase_id
     FROM (
        SELECT testcase_id, COUNT(*) as majority_outcome
        FROM {tables.results.__tablename__}
        WHERE outcome <> {OUTCOMES_TO_INT['bc']}
        GROUP BY testcase_id) r
     WHERE majority_outcome < {min_majority_outcome})
""")

    session.commit()
    # Go testcase-by-testcase
    testcases = session.query(tables.testcases)\
        .filter(tables.testcases.id.in_(
            session.query(tables.results.testcase_id)\
                .outerjoin(tables.classifications)\
                .filter(tables.classifications.id == None))).all()

    # We count compiler crashes and timeouts as passes, since there's no way of
    # voting on crashes/timeouts:
    print("Classifying {bc,bto}-outcomes as passes ...")
    session.execute(f"""
INSERT INTO {tables.classifications.__tablename__} (id, classification)
SELECT {tables.results.__tablename__}.id, {CLASSIFICATIONS_TO_INT['pass']}
FROM {tables.results.__tablename__}
LEFT JOIN {tables.classifications.__tablename__} ON {tables.results.__tablename__}.id = {tables.classifications.__tablename__}.id
WHERE {tables.classifications.__tablename__}.id IS NULL
AND outcome IN ({OUTCOMES_TO_INT['bc']}, {OUTCOMES_TO_INT['bto']})
""")
    session.commit()

    to_add = []

    print("Applying voting heuristics ...")
    for i, testcase in enumerate(ProgressBar()(testcases)):

        if i and not i % 1000:
            session.bulk_save_objects(to_add)
            to_add = []
            session.commit()

        testcase_id = testcase.id

        results = session.query(tables.results)\
            .outerjoin(tables.classifications)\
            .filter(tables.results.testcase_id == testcase_id,
                    tables.classifications.id == None).all()
        n = len(results)

        if n < min_majority_outcome:
            to_add += [tables.classifications(id=r.id, classification=CLASSIFICATIONS_TO_INT['pass'])
                       for r in results]
            continue

        # Determine majority outcome:
        min_majority_count = math.ceil(n / 2)
        majority_outcome, majority_count = get_majority([r.outcome for r in results])

        # If majority outcome resulted in binaries, mark anomalous build
        # failures:
        minority_outcomes = set([OUTCOMES_TO_INT["bf"], OUTCOMES_TO_INT["c"], OUTCOMES_TO_INT["to"]])
        if majority_outcome not in minority_outcomes:
            to_add += [tables.classifications(id=r.id, classification=CLASSIFICATIONS_TO_INT[OUTCOMES[r.outcome]])
                       for r in results if r.outcome in minority_outcomes]
            results[:] = [r for r in results if r.outcome not in minority_outcomes]

        # If the majority did not produce outputs, then we're done:
        if majority_outcome != OUTCOMES_TO_INT["pass"]:
            to_add += [tables.classifications(id=r.id, classification=CLASSIFICATIONS_TO_INT['pass'])
                       for r in results]
            continue

        # Look for wrong-code bugs:
        majority_output, output_majority_count = get_majority([r.stdout_id for r in results])

        # Ensure that the majority of configurations agree on the output:
        min_output_majority_count = math.ceil(len(results) / 2)
        # min_output_majority_count = len(results) - 1
        if output_majority_count == len(results):
            # Everyone agreed on the output:
            to_add += [tables.classifications(id=r.id, classification=CLASSIFICATIONS_TO_INT["pass"])
                       for r in results]
        elif output_majority_count < min_output_majority_count:
            # No majority:
            print("skipping output_majority_count <", min_output_majority_count, " = ", output_majority_count)
            to_add += [tables.classifications(id=r.id, classification=CLASSIFICATIONS_TO_INT["pass"])
                       for r in results]
        else:
            # At least one result disagreed:
            to_add += [tables.classifications(id=r.id, classification=CLASSIFICATIONS_TO_INT["pass"] if r.stdout_id == majority_output else CLASSIFICATIONS_TO_INT["w"])
                       for r in results]
    session.commit()


def verify_testcase(session: session_t, tables: Tableset, testcase) -> None:

    def fail():
        q1 = session.query(tables.classifications.id)\
            .join(tables.results)\
            .filter(tables.results.testcase_id == testcase.id,
                    tables.classifications == CLASSIFICATIONS_TO_INT["w"])
        q2 = session.query(tables.classification)\
                .filter(tables.classification.id.in_(q1))\
                .update({"classification": "pass"})
        n = q2.count()
        if n:
            print("retracting w-classification on {n} results")
            q2.update({"classification": "pass"})
        session.commit()

    # TODO: Consider checking stderrs for warnings

    if testcase.oclverified == None:
        if tables.name == "CLSmith":
            testcase.oclverified = oclgrind.oclgrind_verify_clsmith(testcase)
        else:
            testcase.oclverified = oclgrind.oclgrind_verify_clgen(testcase)

    if not testcase.oclverified:
        print(f"testcase {testcase.id}: failed OCLgrind verification")
        fail()

    if tables.name == "CLgen":
        if testcase.contains_floats == None:
            testcase.contains_floats = "float" in testcase.program.src

        if testcase.contains_floats:
            print(f"testcase {testcase.id}: contains floats")
            fail()

        # Run GPUverify on kernel
        if testcase.gpuverified == None:
            try:
                clgen.gpuverify(testcase.program.src, ["--local_size=64", "--num_groups=128"])
                testcase.gpuverified = 1
            except clgen.GPUVerifyException:
                testcase.gpuverified = 0

        if not testcase.gpuverified:
            print(f"testcase {testcase.id}: failed GPUVerify check")
            fail()

    session.commit()


def verify_w_classifications(session: session_t, tables: Tableset) -> None:
    q = session.query(tables.results.testcase_id)\
            .join(tables.classifications)\
            .filter(tables.classifications.classification == CLASSIFICATIONS_TO_INT["w"])\
            .distinct()
    testcases_to_verify = session.query(tables.testcases)\
                            .filter(tables.results.testcase_id.in_(q))\
                            .distinct().all()

    for testcase in ProgressBar()(testcases_to_verify):
        # testcase = session.query(tables.testcases)\
        #     .filter(tables.testcases.id == testcase_id)
        verify_testcase(session, tables, testcase)


if __name__ == "__main__":
    parser = ArgumentParser(description="Collect difftest results for a device")
    parser.add_argument("-H", "--hostname", type=str, default="cc1",
                        help="MySQL database hostname")
    parser.add_argument("--clsmith", action="store_true",
                        help="analyze only clsmith results")
    parser.add_argument("--clgen", action="store_true",
                        help="analyze only clgen results")
    parser.add_argument("--prune", action="store_true")
    parser.add_argument("-t", "--time-limit", type=int, default=48,
                        help="time limit in hours (default: 48)")
    args = parser.parse_args()

    tables = []
    if not args.clgen:
        tables.append(CLSMITH_TABLES)
    if not args.clsmith:
        tables.append(CLGEN_TABLES)

    # Connect to database
    db_hostname = args.hostname
    print("connected to", db.init(db_hostname))

    set_classifications = set_our_classifications
    # set_classifications = set_clsmith_classifications

    with Session(commit=True) as s:
        for tableset in tables:
            if args.prune:
                verify_w_classifications(s, tableset)
            else:
                get_classifications(s, tableset)
