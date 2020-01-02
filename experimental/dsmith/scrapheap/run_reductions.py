#!/usr/bin/env python3
from collections import namedtuple
from typing import NewType

from dsmith.db import *
from dsmith.lib import *

from labm8.py import fs
from third_party.py.pyopencl import pyopencl as cl

# paths to clreduce library
CLREDUCE_DIR = fs.abspath("..", "lib", "clreduce")
CLSMITH_DIR = fs.abspath("..", "lib", "CLSmith", "build")

# FIXME:
# CL_LAUNCHER = fs.abspath('../lib/clsmith/build/cl_launcher')
# CLSMITH_HEADERS = [path for path in fs.ls(CLSMITH_DIR, abspaths=True) if path.endswith('.h')]
# CLSMITH_RUNTIME_DIR = fs.abspath('..', 'lib', 'CLSmith', 'runtime')
# CREDUCE = fs.abspath(CLREDUCE_DIR, 'build_creduce', 'creduce', 'creduce')
# INTERESTING_TEST = fs.abspath(CLREDUCE_DIR, 'interestingness_tests', 'wrong_code_bug.py')
# OCLGRIND = fs.abspath('../lib/clgen/native/oclgrind/c3760d07365b74ccda04cd361e1b567a6d99dd8c/install/bin/oclgrind')

status_t = NewType("status_t", int)
return_t = namedtuple("return_t", ["runtime", "status", "log", "src"])


def get_platform_name(platform_id):
  platform = cl.get_platforms()[platform_id]
  return platform.get_info(cl.platform_info.NAME)


def get_device_name(platform_id, device_id):
  platform = cl.get_platforms()[platform_id]
  device = platform.get_devices()[device_id]
  return device.get_info(cl.device_info.NAME)


# def get_num_results_to_reduce(session: db.session_t, tables: Tableset, testbed: Testbed):
#     num_ran = session.query(sql.sql.func.count(tables.reductions.id))\
#         .join(tables.results)\
#         .filter(tables.results.testbed_id == testbed.id)\
#         .scalar()
#     total = session.query(sql.sql.func.count(tables.results.id))\
#         .join(tables.classifications)\
#         .filter(tables.classifications.classification == CLASSIFICATIONS_TO_INT['w'],
#                 tables.results.testbed_id == testbed.id)\
#         .scalar()
#     return num_ran, total

# def remove_preprocessor_comments(test_case_name):
#     """ written by the CLreduce folks """
#     for line in fileinput.input(test_case_name, inplace=True):
#         if re.match(r'^# \d+ "[^"]*"', line):
#             continue
#         print(line, end="")

# def run_reduction(s: session_t, tables: Tableset,
#                   result: Union[CLSmithResult, CLgenResult]) -> Union[CLSmithReduction, CLgenReduction]:
#     """
#     Note as a side effect this method modifies environment variables.
#     """
#     with TemporaryDirectory(prefix='clreduce-') as tmpdir:
#         path = fs.path(tmpdir, "kernel.cl")

#         # move headers into place
#         for header in CLSMITH_HEADERS:
#             fs.cp(header, tmpdir)

#         # create kernel file
#         kernel = fs.path(tmpdir, "kernel.cl")
#         with open(kernel, 'w') as outfile:
#             print(result.testcase.program.src, file=outfile)

#         # Preprocess to inline headers
#         cmd = ["clang", "-I", CLSMITH_DIR, "-I", CLSMITH_RUNTIME_DIR,
#                "-E", "-CC", "-o", kernel, path]
#         pp = subprocess.run(cmd, timeout=60, check=True)
#         remove_preprocessor_comments(kernel)

#         # Get OpenCL device and platform indexes
#         env = cldrive.make_env(platform=result.testbed.platform,
#                                device=result.testbed.device)
#         platform_id, device_id = env.ids()

#         # Setup env
#         os.chdir(tmpdir)
#         optimized = 'optimised' if result.testcase.params.optimizations else 'unoptimised'
#         os.environ['CREDUCE_TEST_OPTIMISATION_LEVEL'] = optimized
#         os.environ['CREDUCE_TEST_CASE'] = path
#         os.environ['OCLGRIND'] = OCLGRIND
#         os.environ['CREDUCE_TEST_PLATFORM'] = str(platform_id)
#         os.environ['CREDUCE_TEST_DEVICE'] = str(device_id)

#         if tables.name == "CLSmith":
#             # If reducing a CLSmith program, use the default cl_launcher
#             os.environ['CREDUCE_TEST_CL_LAUNCHER'] = CL_LAUNCHER
#         else:
#             # Else, generate a harness which will serve as a stand-in for cl_launcher
#             driver_path = fs.path(tmpdir, "driver")
#             generation_time, compile_only, src = clgen_mkharness.mkharness_src(env, result.testcase)
#             clgen_mkharness.compile_harness(src, driver_path)
#             os.environ['CREDUCE_TEST_CL_LAUNCHER'] = driver_path

#         # Log the interestingness test once to check that everything is okay
#         out = []
#         cmd = ['timeout', '-s9', '60s', INTERESTING_TEST]
#         process = subprocess.Popen(cmd, universal_newlines=True, bufsize=1,
#                                    stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
#         with process.stdout:
#             for line in process.stdout:
#                 sys.stdout.write(line)
#                 out.append(line)
#         process.wait()

#         # Run the actual reduction
#         start_time = time()
#         cmd = ['perl', '--', CREDUCE, '--n', '4', '--timing', INTERESTING_TEST, path]
#         process = subprocess.Popen(cmd, universal_newlines=True, bufsize=1,
#                                    stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
#         with process.stdout:
#             for line in process.stdout:
#                 sys.stdout.write(line)
#                 out.append(line)
#         process.wait()

#         status = process.returncode
#         runtime = time() - start_time

#         with open(kernel) as infile:
#             src = infile.read()

#     if tables.name == "CLSmith":
#         reduction_t = CLSmithReduction
#     else:
#         reduction_t = CLgenReduction

#     return reduction_t(result=result, runtime=runtime, status=status,
#                        linecount=len(src.split('\n')), src=src, log='\n'.join(out))

# if __name__ == "__main__":
#     parser = ArgumentParser()
#     parser.add_argument(
#         "-H", "--hostname", type=str, default="cc1",
#         help="MySQL database hostname")
#     parser.add_argument(
#         "platform_id", type=int, metavar="<platform id>", help="OpenCL platform ID")
#     parser.add_argument(
#         "device_id", type=int, metavar="<device id>", help="OpenCL device ID")
#     parser.add_argument("--clsmith", action="store_true",
#                         help="Only reduce CLSmith results")
#     parser.add_argument("--clgen", action="store_true",
#                         help="Only reduce CLgen results")
#     args = parser.parse_args()

#     db.init(args.hostname)  # initialize db engine

#     assert fs.isexe(CREDUCE)
#     assert fs.isexe(INTERESTING_TEST)

#     platform_id, device_id = args.platform_id, args.device_id
#     platform_name = get_platform_name(platform_id)
#     device_name = get_device_name(platform_id, device_id)

#     devname = util.device_str(device_name)

#     if args.clgen and args.clsmith:
#         tablesets = [CLSMITH_TABLES, CLGEN_TABLES]
#     elif args.clsmith:
#         tablesets = [CLSMITH_TABLES]
#     elif args.clgen:
#         tablesets = [CLGEN_TABLES]
#     else:
#         tablesets = [CLSMITH_TABLES, CLGEN_TABLES]

#     with Session() as s:
#         testbed = get_testbed(s, platform_name, device_name)

#         for tables in tablesets:
#             print(f"Reducing {tables.name} w-classified results for {devname} ...")

#             # progress bar
#             num_ran, total = get_num_results_to_reduce(s, tables, testbed)
#             bar = progressbar.ProgressBar(init_value=num_ran, max_value=total)

#             # main execution loop:
#             while True:
#                 # get the next result to reduce
#                 done = s.query(tables.reductions.id)
#                 result = s.query(tables.results)\
#                     .join(tables.classifications)\
#                     .filter(tables.results.testbed_id == testbed.id,
#                             tables.classifications.classification == CLASSIFICATIONS_TO_INT["w"],
#                             ~tables.results.id.in_(done))\
#                     .order_by(tables.results.date)\
#                     .first()

#                 if not result:
#                     break

#                 reduction = run_reduction(s, tables, result)

#                 s.add(reduction)
#                 s.commit()

#                 # update progress bar
#                 num_ran, total = get_num_results_to_reduce(s, tables, testbed)
#                 bar.max_value = total
#                 bar.update(min(num_ran, total))
#     print("done.")
