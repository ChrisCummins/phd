#!/usr/bin/env python
"""
Analyze results using voting heuristics.
"""
import cldrive
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
from db import *


def set_classifications(s: session_t) -> None:
    """
    Determine anomalous results.
    """
    s.execute(f"DELETE FROM {Classification.__tablename__}")

    min_majsize = 7

    print(f"setting {{bc,bto}} classifications ...", file=sys.stderr)
    s.execute(f"""
INSERT INTO {Classification.__tablename__}
SELECT results.id, {Classifications.BC}
FROM {Result.__tablename__} results
WHERE outcome = {Outcomes.BC}
""")
    s.execute(f"""
INSERT INTO {Classification.__tablename__}
SELECT results.id, {Classifications.BTO}
FROM {Result.__tablename__} results
WHERE outcome = {Outcomes.BTO}
""")

    print(f"determining anomalous build-failures ...", file=sys.stderr)
    s.execute(f"""
INSERT INTO {Classification.__tablename__}
SELECT results.id, {Classifications.ABF}
FROM {Result.__tablename__} results
INNER JOIN {Majority.__tablename__} majorities ON results.testcase_id = majorities.id
WHERE outcome = {Outcomes.BF}
AND outcome_majsize >= {min_majsize}
AND maj_outcome = {Outcomes.PASS}
""")

    print(f"determining anomalous runtime crashes ...", file=sys.stderr)
    s.execute(f"""
INSERT INTO {Classification.__tablename__}
SELECT {Result.__tablename__}.id, {Classifications.ARC}
FROM {Result.__tablename__}
INNER JOIN {Majority.__tablename__} majorities ON results.testcase_id = majorities.id
WHERE outcome = {Outcomes.RC}
AND outcome_majsize >= {min_majsize}
AND maj_outcome = {Outcomes.PASS}
""")

    print(f"determining anomylous wrong output classifications ...", file=sys.stderr)
    s.execute(f"""
INSERT INTO {Classification.__tablename__}
SELECT results.id, {Classifications.AWO}
FROM {Result.__tablename__} results
INNER JOIN {Majority.__tablename__} majorities ON results.testcase_id = majorities.id
WHERE outcome = {Outcomes.PASS}
AND maj_outcome = {Outcomes.PASS}
AND outcome_majsize >= {min_majsize}
AND stdout_majsize >= CEILING((2 * outcome_majsize) / 3)
AND stdout_id <> maj_stdout_id
""")
    s.commit()


def verify_w_testcase(s: session_t) -> None:

    # CLgen-specific analysis. We can omit these checks for CLSmith, as they
    # will always pass.
    if tables.name == "CLgen":
        if testcase.contains_floats == None:
            testcase.contains_floats = "float" in testcase.program.src

        if testcase.contains_floats:
            print(f"testcase {testcase.id}: contains floats")
            return fail()

        if testcase_raises_compiler_warnings(session, tables, testcase):
            print(f"testcase {testcase.id}: redflag compiler warnings")
            return fail()

        for arg in cldrive.extract_args(testcase.program.src):
            if arg.is_vector:
                # An error in my implementation of vector types:
                print(f"testcase {testcase.id}: contains vector types")
                return fail()

        # Run GPUverify on kernel
        if testcase.gpuverified == None:
            try:
                clgen.gpuverify(testcase.program.src, ["--local_size=64", "--num_groups=128"])
                testcase.gpuverified = 1
            except clgen.GPUVerifyException:
                testcase.gpuverified = 0

        if not testcase.gpuverified:
            print(f"testcase {testcase.id}: failed GPUVerify check")
            return fail()

    # Check that program runs with Oclgrind without error:
    if not oclgrind.verify_testcase(session, tables, testcase):
        print(f"testcase {testcase.id}: failed OCLgrind verification")
        return fail()


# def verify_optimization_sensitive(session: session_t, tables: Tableset, result) -> None:
#     """
#     Check if an anomylous wrong output result is optimization-sensitive, and
#     ignore it if not.
#     """
#     params = result.testcase.params
#     complement_params_id = s.query(tables.params.id)\
#         .filter(tables.params.gsize_x == params.gsize_x,
#                 tables.params.gsize_y == params.gsize_y,
#                 tables.params.gsize_z == params.gsize_z,
#                 tables.params.optimizations == (not params.optimizations))

#     complement_testcase = s.query(Testcase.id)\
#         .filter(Testcase.program_id == result.testcase.program_id,
#                 Testcase.params_id == complement_params_id)

#     q = s.query(
#             Result.id,
#             Classification.classification)\
#         .join(Testcase)\
#         .filter(Result.testbed_id == result.testbed_id,
#                 Result.testcase_id == complement_testcase)\
#         .first()

#     if q:
#         complement_id, complement_classification = q
#         if complement_classification == Classifications.AWO:
#             print(f"retracting w-classification on 2 results {{{result.id},{complement_id}}} - optimization insensitive")
#             s.query(Classification)\
#                 .filter(Classification.id.in_([result.id, complement_id]))\
#                 .delete(synchronize_session=False)
#     else:
#         print(f"no complement result for result {result.id}")


def prune_w_classifications(s: session_t) -> None:
    print(f"Verifying w-classified testcases ...", file=sys.stderr)
    q = s.query(Result.testcase_id)\
        .join(Classification)\
        .filter(Classification.classification == Classifications.AWO)\
        .distinct()
    testcases_to_verify = s.query(Testcase)\
        .filter(Testcase.id.in_(q))\
        .distinct()\
        .all()

    for testcase in ProgressBar()(testcases_to_verify):
        if not testcase.verify_awo(s):
            q = s.query(Result.id)\
                .join(Classification)\
                .filter(Result.testcase_id == testcase.id,
                        Classification.classification == Classifications.AWO)
            ids_to_update = [x[0] for x in q.all()]
            n = len(ids_to_update)
            assert n > 0
            ids_str = ",".join(str(x) for x in ids_to_update)
            print(f"retracting awo-classification on {n} results: {ids_str}")
            s.query(Classification)\
                .filter(Classification.id.in_(ids_to_update))\
                .delete(synchronize_session=False)


# def verify_opencl_version(session: session_t, tables: Tableset, testcase) -> None:
#     """
#     OpenCL 2.0
#     """
#     opencl_2_0_testbeds = s.query(Testbed.id).filter(Testbed.opencl == "2.0")

#     passes_2_0 = s.query(sql.sql.func.count(Result.id))\
#         .filter(Result.testcase_id == testcase.id,
#                 Result.testbed_id.in_(opencl_2_0_testbeds),
#                 Result.outcome != Outcomes.BF)\
#         .scalar()

#     if not passes_2_0:
#         # If it didn't build on OpenCL 2.0, we're done.
#         return

#     passes_1_2 = s.query(sql.sql.func.count(Result.id))\
#         .filter(Result.testcase_id == testcase.id,
#                 ~Result.testbed_id.in_(opencl_2_0_testbeds),
#                 Result.outcome == Outcomes.PASS)\
#         .scalar()

#     if passes_1_2:
#         # If it *did* build on OpenCL 1.2, we're done.
#         return

#     ids_to_update = [
#         x[0] for x in
#         s.query(Result.id)\
#             .join(Classification)\
#             .filter(Result.testcase_id == testcase.id,
#                     ~Result.testbed_id.in_(opencl_2_0_testbeds),
#                     Classification.classification == Classifications.ABF).all()
#     ]
#     n = len(ids_to_update)
#     if n:
#         ids_str = ",".join(str(x) for x in ids_to_update)
#         print(f"retracting bf-classification for testcase {testcase.id} - only passes on OpenCL 2.0. {n} results: {ids_str}")
#         s.query(Classification)\
#             .filter(Classification.id.in_(ids_to_update))\
#             .delete(synchronize_session=False)


# def prune_bf_classifications(session: session_t, tables: Tableset) -> None:

#     def prune_stderr_like(like):
#         ids_to_delete = [x[0] for x in s.query(Result.id)\
#             .join(Classification)\
#             .join(tables.stderrs)\
#             .filter(Classification.classification == Classifications.ABF,
#                     tables.stderrs.stderr.like(f"%{like}%"))]

#         n = len(ids_to_delete)
#         if n:
#             print(f"retracting {n} bf-classified results with msg {like[:30]}")
#             s.query(Classification)\
#                 .filter(Classification.id.in_(ids_to_delete))\
#                 .delete(synchronize_session=False)

#     prune_stderr_like("use of type 'double' requires cl_khr_fp64 extension")
#     prune_stderr_like("implicit declaration of function")
#     prune_stderr_like("function cannot have argument whose type is, or contains, type size_t")
#     prune_stderr_like("unresolved extern function")
#     # prune_stderr_like("error: declaration does not declare anything")
#     prune_stderr_like("error: cannot increment value of type%")
#     prune_stderr_like("subscripted access is not allowed for OpenCL vectors")
#     prune_stderr_like("Images are not supported on given device")
#     prune_stderr_like("error: variables in function scope cannot be declared")
#     prune_stderr_like("error: implicit conversion ")
#     # This is fine: prune_stderr_like("error: automatic variable qualified with an address space ")
#     prune_stderr_like("Could not find a definition ")

#     # Verify results
#     q = s.query(Result.testcase_id)\
#         .join(Classification)\
#         .join(Testbed)\
#         .filter(Classification.classification == Classifications.ABF,
#                 Testbed.opencl == "1.2")\
#         .distinct()
#     testcases_to_verify = s.query(Testcase)\
#         .filter(Testcase.id.in_(q))\
#         .distinct()\
#         .all()

#     print(f"Verifying bf-classified testcases ...", file=sys.stderr)
#     for testcase in ProgressBar()(testcases_to_verify):
#         verify_opencl_version(session, tables, testcase)

#     s.commit()


# def verify_c_testcase(session: session_t, tables: Tableset, testcase) -> None:
#     """
#     Verify that a testcase is sensible.
#     """

#     def fail():
#         ids_to_update = [
#             x[0] for x in
#             s.query(Result.id)\
#                 .join(Classification)\
#                 .filter(Result.testcase_id == testcase.id,
#                         Classification.classification == Classifications.ARC)\
#                 .all()
#         ]
#         n = len(ids_to_update)
#         assert n > 0
#         ids_str = ",".join(str(x) for x in ids_to_update)
#         print(f"retracting c-classification on {n} results: {ids_str}")
#         s.query(Classification)\
#             .filter(Classification.id.in_(ids_to_update))\
#             .delete(synchronize_session=False)

#     # CLgen-specific analysis. We can omit these checks for CLSmith, as they
#     # will always pass.
#     if tables.name == "CLgen":
#         if testcase_raises_compiler_warnings(session, tables, testcase):
#             print(f"testcase {testcase.id}: redflag compiler warnings")
#             return fail()

#         # Run GPUverify on kernel
#         if testcase.gpuverified == None:
#             try:
#                 clgen.gpuverify(testcase.program.src, ["--local_size=64", "--num_groups=128"])
#                 testcase.gpuverified = 1
#             except clgen.GPUVerifyException:
#                 testcase.gpuverified = 0

#         if not testcase.gpuverified:
#             print(f"testcase {testcase.id}: failed GPUVerify check")
#             return fail()

#     # Check that program runs with Oclgrind without error:
#     if not oclgrind.verify_testcase(session, tables, testcase):
#         print(f"testcase {testcase.id}: failed OCLgrind verification")
#         return fail()


# def prune_c_classifications(session: session_t, tables: Tableset) -> None:

#     def prune_stderr_like(like):
#         ids_to_delete = [x[0] for x in s.query(Result.id)\
#             .join(Classification)\
#             .join(tables.stderrs)\
#             .filter(Classification.classification == Classifications.ARC,
#                     tables.stderrs.stderr.like(f"%{like}%"))]

#         n = len(ids_to_delete)
#         if n:
#             print(f"retracting {n} c-classified results with msg {like[:30]}")
#             s.query(Classification)\
#                 .filter(Classification.id.in_(ids_to_delete))\
#                 .delete(synchronize_session=False)

#     prune_stderr_like("clFinish CL_INVALID_COMMAND_QUEUE")

#     # Verify testcases
#     q = s.query(Result.testcase_id)\
#             .join(Classification)\
#             .filter(Classification.classification == Classifications.ARC)\
#             .distinct()
#     testcases_to_verify = s.query(Testcase)\
#             .filter(Testcase.id.in_(q))\
#             .distinct()\
#             .all()

#     print(f"Verifying c-classified testcases ...", file=sys.stderr)
#     for testcase in ProgressBar()(testcases_to_verify):
#         verify_c_testcase(session, tables, testcase)

#     s.commit()


if __name__ == "__main__":
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("-H", "--hostname", type=str, default="cc1",
                        help="MySQL database hostname")
    args = parser.parse_args()

    # Connect to database
    db_hostname = args.hostname
    print("connected to", db.init(db_hostname), file=sys.stderr)

    with Session(commit=True) as s:
        # set_classifications(s)
        prune_w_classifications(s)
        # prune_bf_classifications(s)
        # prune_c_classifications(s)
