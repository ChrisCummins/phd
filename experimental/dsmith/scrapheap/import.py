#!/usr/bin/env python
from dsmith.db import *

# def import_result_protobufs(s: session_t, path: Path, generator: Generators.type) -> None:
#     print(f"importing results from {path}")

#     STDOUTS = dict((k, v) for k, v in s.query(Stdout.sha1, Stdout.id))
#     print(len(STDOUTS), "stdouts")

#     STDERRS = dict((k, v) for k, v in s.query(Stderr.sha1, Stderr.id))
#     print(len(STDERRS), "stderrs")

#     PROGRAMS = dict((k, v) for k, v in s.query(Program.sha1, Program.id)\
#                                         .filter(Program.generator == generator))
#     print(len(PROGRAMS), "programs")

#     PLATFORMS = dict(((platform, device), id) for platform, device, id
#                      in s.query(Platform.platform, Platform.device, Platform.id))
#     print(len(PLATFORMS), "platforms")

#     TESTBEDS = dict(((platform_id, optimizations), id)
#                     for platform_id, optimizations, id
#                     in s.query(Testbed.platform_id, Testbed.optimizations, Testbed.id))
#     print(len(TESTBEDS), "testbeds")

#     THREADS = dict((k, v) for k, v in s.query(Threads.gsize_x, Threads.id))
#     print(len(THREADS), "threads")

#     TESTCASES = dict(((program_id, threads_id), id)
#                      for program_id, threads_id, id
#                      in s.query(Testcase.program_id, Testcase.threads_id, Testcase.id)\
#                             .join(Program)\
#                             .filter(Program.generator == generator))
#     print(len(TESTCASES), "testcases")

#     files = list(path.iterdir())
#     bar = ProgressBar(max_value=len(files))
#     outbox = deque()

#     for i, path in enumerate(bar(files)):
#         with open(path, "rb") as f:
#             buf = pb.Result.FromString(f.read())

#             platform_id = PLATFORMS[(buf.testbed.platform.cl_platform, buf.testbed.platform.cl_device)]
#             testbed_id = TESTBEDS[(platform_id, buf.testbed.cl_opt)]
#             stdout_id = STDOUTS[crypto.sha1_str(buf.stdout)]
#             stderr_id = STDERRS[crypto.sha1_str(buf.stderr)]
#             threads_id = THREADS[buf.testcase.params.gsize_x]
#             program_id = PROGRAMS[crypto.sha1_str(buf.testcase.program.src)]
#             testcase_id = TESTCASES[(program_id, threads_id)]

#             result = Result(
#                 testbed_id=testbed_id, testcase_id=testcase_id,
#                 date=datetime.datetime.fromtimestamp(buf.date),
#                 returncode=buf.returncode, outcome=-1, runtime=buf.runtime,
#                 stdout_id=stdout_id, stderr_id=stderr_id)
#             outbox.append(result)

#         if i and not i % 1000:
#             s.bulk_save_objects(outbox)
#             s.commit()
#             outbox = deque()


# def import_clang_protobufs(s: session_t, path: Path, generator: Generators.type) -> None:
#     for clang in ["3.6.2", "3.7.1", "3.8.1", "3.9.1", "4.0.1", "5.0.0", "6.0.0"]:
#         p = get_or_create(s, Platform, platform="clang", device="",
#                           driver=clang, opencl="", devtype="Compiler",
#                           host="Ubuntu 16.04 64bit")
#         s.flush()
#         t = get_or_create(s, Testbed, platform=p, optimizations=1)
#         s.add(t)
#     s.commit()

#     print(f"importing results from clang")

#     STDOUTS = dict((k, v) for k, v in s.query(Stdout.sha1, Stdout.id))
#     print(len(STDOUTS), "stdouts")

#     STDERRS = dict((k, v) for k, v in s.query(Stderr.sha1, Stderr.id))
#     print(len(STDERRS), "stderrs")

#     # threads_id = s.query(Threads.id).filter(Threads.gsize_x == 1, Threads.lsize_x == 1).scalar()

#     PLATFORMS = dict(((platform, device), id) for platform, device, id
#                      in s.query(Platform.platform, Platform.device, Platform.id))
#     print(len(PLATFORMS), "platforms")

#     TESTBEDS = dict((driver, id) for driver, id
#                     in s.query(Platform.driver, Testbed.id)\
#                         .join(Testbed)\
#                         .filter(Platform.platform == "clang"))
#     print(len(TESTBEDS), "testbeds")

#     TESTCASES = dict((program_id, id)
#                      for program_id, id
#                      in s.query(Program.sha1, Testcase.id)\
#                             .join(Testcase)\
#                             .filter(Testcase.harness == Harnesses.COMPILE_ONLY,
#                                     Testcase.threads_id == 0))
#     print(len(TESTCASES), "testcases")

#     files = list(path.iterdir())
#     bar = ProgressBar(max_value=len(files))
#     outbox = deque()

#     for i, path in enumerate(bar(files)):
#         with open(path, "rb") as f:
#             buf = pb.Result.FromString(f.read())

#             testbed_id = TESTBEDS[buf.testbed.platform.cl_driver]
#             testcase_id = TESTCASES[crypto.sha1_str(buf.testcase.program.src)]

#             stdout_id = STDOUTS[crypto.sha1_str(buf.stdout)]
#             try:
#                 sha1 = crypto.sha1_str(buf.stderr)
#                 stderr_id = STDERRS[sha1]
#             except KeyError:
#                 stderr = Stderr(sha1=sha1, stderr=buf.stderr)
#                 s.add(stderr)
#                 s.flush()
#                 stderr_id = stderr.id
#                 STDERRS[sha1] = stderr_id

#             result = Result(
#                 testbed_id=testbed_id, testcase_id=testcase_id,
#                 date=datetime.datetime.fromtimestamp(buf.date),
#                 returncode=buf.returncode, runtime=buf.runtime,
#                 outcome=Outcomes.BC if buf.returncode == -6 else Outcomes.PASS,
#                 stdout_id=stdout_id, stderr_id=stderr_id)
#             outbox.append(result)

#         if i and not i % 1000:
#             s.bulk_save_objects(outbox)
#             s.commit()
#             outbox = deque()


# def import_protobufs(s: session_t, path: Path) -> None:
#     # import_result_protobufs(s, path / "clsmith", 0)
#     # import_result_protobufs(s, path / "dsmith", 1)
#     import_clang_protobufs(s, path / "clang", 1)


# if __name__ == "__main__":
#     parser = ArgumentParser(description="Collect difftest results for a device")
#     parser.add_argument("-H", "--hostname", type=str, default="cc1",
#                         help="MySQL database hostname")
#     parser.add_argument("dir", metavar="<dir>", help="directory to import from")
#     args = parser.parse_args()

#     # Connect to database
#     db_hostname = args.hostname
#     print("connected to", db.init(db_hostname))

#     with Session(commit=False) as s:
#         import_protobufs(s, Path(args.dir))
#     print("done")
