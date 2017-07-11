#!/usr/bin/env python
import sqlalchemy as sql
import sys

from collections import Counter
from progressbar import ProgressBar
from signal import Signals

import db
from db import *

CO_TABLE_NAMES = ["CLgen"]
CO_TABLES = [coCLgenResult]

# NOTE: excludes GPUverified tables
CL_LAUNCHER_TABLE_NAMES = ["CLSmith"]
CL_LAUNCHER_TABLES = [CLSmithResult]

CLDRIVE_TABLE_NAMES = ["CLSmith w. cldrive", "GitHub"]
CLDRIVE_TABLES = [cldriveCLSmithResult, GitHubResult]


# Group outcomes into less-granular classifications.
# This table maps <outcome>: <classification>
CLASSIFICATIONS = {
    'CL_BUILD_PROGRAM_FAILURE': 'Build failure',
    'CL_INVALID_COMMAND_QUEUE': 'Runtime crash',
    'CL_INVALID_KERNEL_ARGS': 'Invalid testcase',
    'CL_INVALID_KERNEL_NAME': 'Build failure',
    'CL_INVALID_VALUE': 'Runtime crash',
    'CL_OUT_OF_HOST_MEMORY': 'Runtime crash',
    'CL_OUT_OF_RESOURCES': 'Runtime crash',
    'cldrive Error': 'Invalid testcase',
    'cldrive.driver.PorcelainError: 1': 'Runtime crash',
    'cldrive.driver.PorcelainError: 127': 'Runtime crash',
    'cldrive.driver.PorcelainError: SIGFPE': 'Runtime crash',
    'clWaitForEvents()': 'Runtime crash',
    'Error building program: -42': 'Invalid testcase',
    'Error enqueueing kernel': 'Invalid testcase',
    'Error setting kernel argument': 'Invalid testcase',
    'Error: recursive call in function(s):': 'Invalid testcase',
    'Error: unresolved pointer calls in function(s):': 'Invalid testcase',
    'INVALID_WORK_GROUP_SIZE': 'Invalid testcase',
    'Multiple OpenCL kernels': 'Invalid testcase',
    'No OpenCL kernel': 'Invalid testcase',
    'Preprocessing Failed': 'Invalid testcase',
    'Returncode 127': 'Runtime crash',
    'Segmentation Fault': 'Build failure',
    'ParseError': 'Invalid testcase',
    'SIGABRT': 'Runtime crash',
    'SIGBUS': 'Runtime crash',
    'SIGFPE': 'Runtime crash',
    'SIGILL': 'Runtime crash',
    'Signal 5': 'Runtime crash',
    'SIGTRAP': 'Runtime crash',
    'Timeout': 'Timeout',
    'UnicodeError': 'Invalid testcase',
    'Unsupported Program': 'Invalid testcase',
    'z_Okay': 'Okay',
}


def lookup_returncode(status):
    try:
        return {
            -11: "Segmentation Fault",
            -9: "Timeout",
            0: "z_Okay",
            139: "Segmentation Fault",
            1024: "Preprocessing Failed",
            1025: "UnicodeError",
        }[status]
    except KeyError:
        try:
            return Signals(-status).name
        except ValueError:
            return f'Returncode {status}'


def get_cl_launcher_outcome(result):
    if result.status == 1:
        # exit status of 1 indicates that cl_launcher exited after an OpenCL API call error
        prev = None
        for line in reversed(result.stderr.split('\n')):
            if line == "Error found (callback):":
                # Interpret CLSmith error callback messages: the following line describes the error
                raise LookupError(prev.split()[0])
                return prev.split()[0]
            elif line.startswith("Error setting kernel argument"):
                return "bf"
            elif line.startswith("Error enqueueing kernel"):
                return "c"
            elif line.startswith("Error"):
                # Interpret CLSmith error messages
                return {
                    "Error building program: -11": "bf",
                    "Error creating kernel: -46": "bf",
                    "Error sending finish command: -36": "c",
                }.get(line, line)
            prev = line
        else:  # should never happen
            print("\nI could not understand this output:")
            print(result.stderr)
            print("Return code:", result.status)
            print("stdout:")
            print(result.stdout)
            print("RESULT ID:", result.id)
            raise LookupError
    else:
        return lookup_returncode(result.status)


def get_co_outcome(result, session):
    if result.status == 1:
        if result.stderr.find("pycparser.plyparser.ParseError"):
            return "ParseError"
        try:
            return {
                "clBuildProgram CL_BUILD_PROGRAM_FAILURE": "CL_BUILD_PROGRAM_FAILURE",
                "clBuildProgram CL_INVALID_BINARY": "CL_INVALID_BINARY",
            }[result.stderr.rstrip().split('\n')[-1]]
        except KeyError:
            print("\n\n=============\nI could not understand this output:")
            print(result.stderr)
            print("Return code:", result.status)
            print("RESULT ID:", result.id)
            print("PROGRAM ID:", result.program.id)
            print()
            print(result.program.src)
            print()
            raise KeyError
    else:
        return lookup_returncode(result.status)


def get_cldrive_outcome(result):
    if result.status == 1:
        # exit status of 1
        line = result.stderr.split('\n')[-2]  # last line contains the Exception line
        if line.startswith('cldrive.args.OpenCLValueError'):
            return 'Unsupported Program'
        elif line.startswith('ValueError: unsupported data type'):
            return 'cldrive Error'
        elif line.startswith('TypeError: unsupported operand type'):
            return 'cldrive Error'
        elif line.startswith('TypeError: _set_args() takes'):
            return 'cldrive Error'
        elif line.startswith('ValueError: could not broadcast input array from shape'):
            return 'cldrive Error'
        else:
            # This table maps <exception line>: <outcome>
            return {
                'AssertionError': 'cldrive Error',
                'cldrive.driver.PorcelainError: SIGBUS': 'SIGBUS',
                'cldrive.driver.PorcelainError: SIGSEGV': 'Segmentation Fault',
                'cldrive.driver.Timeout: 60': 'Timeout',
                'EOFError: Ran out of input': 'cldrive Error',
                'KeyError: \'outputs\'': 'cldrive Error',
                'LookupError: source contains more than one kernel definition': 'Multiple OpenCL kernels',
                'LookupError: source contains no kernel definitions.': 'No OpenCL kernel',
                'pyopencl.cffi_cl.LogicError: clEnqueueNDRangeKernel failed: INVALID_WORK_GROUP_SIZE': 'INVALID_WORK_GROUP_SIZE',
                'pyopencl.cffi_cl.LogicError: clWaitForEvents failed: <unknown error -9999>': 'clWaitForEvents()',
                'pyopencl.cffi_cl.RuntimeError: clEnqueueNDRangeKernel failed: OUT_OF_HOST_MEMORY': 'CL_OUT_OF_HOST_MEMORY',
                'pyopencl.cffi_cl.RuntimeError: clEnqueueNDRangeKernel failed: OUT_OF_RESOURCES': 'CL_OUT_OF_RESOURCES',
                'RuntimeError': 'cldrive Error',
                'ValueError: failed to set kernel args': 'cldrive Error',
            }.get(line, line)
    else:
        return lookup_returncode(result.status)


def analyze_co_result(result, table, session) -> None:
    result.outcome = get_co_outcome(result, session)
    result.classification = CLASSIFICATIONS[result.outcome]


def analyze_cl_launcher_result(result, table, session,
                               require_gpuverified=False,
                               require_handchecked=False):
    result.outcome = get_cl_launcher_outcome(result)
    # result.classification = CLASSIFICATIONS[result.outcome]

    # # determine if output differs from the majority (if there is one)
    # is_okay = result.status == 0
    # if is_okay and require_gpuverified:
    #     is_okay |= result.program.gpuverified == 1
    # if is_okay and require_handchecked:
    #     is_okay |= result.program.handchecked == 1

    # if is_okay:
    #     outputs = [x[0] for x in session.query(table.stdout)\
    #         .filter(table.program == result.program,
    #                 table.params == result.params,
    #                 table.status == 0)]
    #     if len(outputs) > 3:
    #         # Use voting to pick oracle.
    #         majority_output, majority_count = Counter(outputs).most_common(1)[0]
    #         if majority_count < 3:  # no majority
    #             result.classification = "No majority"
    #         elif result.stdout != majority_output:
    #             result.classification = "Wrong code"
    #     elif len(outputs) == 2:
    #         if outputs[0] != outputs[1]:
                # result.classification = "No majority"


def analyze_cldrive_result(result, table, session, require_gpuverified=False,
                           require_handchecked=False):
    result.outcome = get_cldrive_outcome(result)
    result.classification = CLASSIFICATIONS[result.outcome]

    # determine if output differs from the majority (if there is one)
    is_okay = result.status == 0
    if is_okay and require_gpuverified:
        is_okay |= result.program.gpuverified == 1
    if is_okay and require_handchecked:
        is_okay |= result.program.handchecked == 1

    if is_okay:
        outputs = [x[0] for x in session.query(table.stdout)\
            .filter(table.program == result.program,
                    table.params == result.params,
                    table.status == 0)]
        if len(outputs) > 2:
            # Use voting to pick oracle.
            majority_output, majority_count = Counter(outputs).most_common(1)[0]
            if majority_count < 3:  # no majority
                result.classification = "No majority"
            elif result.stdout != majority_output:
                result.classification = "Wrong code"
        elif len(outputs) == 2:
            if outputs[0] != outputs[1]:
                result.classification = "No majority"

    result.outcome = get_cldrive_outcome(result)
    result.classification = CLASSIFICATIONS[result.outcome]



def main():
    db.init("cc1")
    session = db.make_session()

    print("Possible classifications:", ", ".join(f"'{s}'" for s in sorted(set(CLASSIFICATIONS.values()))))

    # for name, table in zip(CO_TABLE_NAMES, CO_TABLES):
    #     print(f"{name} ...")
    #     for result in ProgressBar()(session.query(table).all()):
    #         analyze_co_result(result, table, session)
    #     session.commit()

    # CLgen programs have special treatment because we require that they be
    # gpuverified before labeling as wrong code
    # name = "CLgen w. cl_launcher"
    # table = cl_launcherCLgenResult
    # print(f"{name} ...")
    # for result in ProgressBar()(session.query(table).all()):
    #     analyze_cl_launcher_result(result, table, session, require_gpuverified=True)
    # session.commit()

    # name = "CLgen"
    # table = CLgenResult
    # print(f"{name} ...")
    # for result in ProgressBar()(session.query(table).all()):
    #     analyze_cldrive_result(result, table, session, require_gpuverified=True)
    # session.commit()

    # cl_launcher result outcomes and classifications:
    for name, table in zip(CL_LAUNCHER_TABLE_NAMES, CL_LAUNCHER_TABLES):
        print(f"{name} outcomes ...")
        num_results = session.query(table).count()
        for result in ProgressBar()(session.query(table), max_value=num_results):
            result.outcome = get_cl_launcher_outcome(result)
        session.commit()

    # # cldrive results outcomes and classifications:
    # for name, table in zip(CLDRIVE_TABLE_NAMES, CLDRIVE_TABLES):
    #     print(f"{name} ...")
    #     for result in ProgressBar()(session.query(table).all()):
    #         analyze_cldrive_result(result, table, session)
    #     session.commit()


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(type(e).__name__, e, file=sys.stderr)
        sys.exit(1)
