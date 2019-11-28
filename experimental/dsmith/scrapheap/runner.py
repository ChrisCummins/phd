"""
Usage:
    ./runner --ls [--verbose]
    # List available testbeds

    ./runer [--verbose] [--only <testbed_ids>] [--exclude <testbed_ids>] [--batch-size <int>] [--host <hostname>]
    # Run testcases
"""
from dsmith.db import *

# result_t = Union[ClsmithResult, DsmithResult]
# testcase_t = Union[ClsmithTestcase, DsmithTestcase]


# def push_testcase_results(s: session_t, outbox: List[result_t]):
#     while not outbox.empty():
#         result = outbox.popleft()
#         result.push(s)


# def fetch_testcase_batch(s: session_t, testbed_ids: List[Testbed.id_t],
#                          tablesets: List[Tableset],
#                          batch_size: int) -> Deque[testcase_t]:
#     inbox = deque()

#     for tables in tablesets:
#         done = s.query(tables.results.testcase_id)\
#             .filter(tables.results.testbed_id.in_(testbed_ids))

#         todo = session.query(tables.testcases)\
#             .filter(~tables.testcases.id.in_(done))\
#             .order_by(tables.testcases.id)

#         for testcase in todo:
#             inbox.append(testcase.toProtobuf())

#     return inbox


# def runner(tested_ids: List[Testbed.id_t], tablesets: List[Tableset],
#            batch_size: int=1000):
#     outbox = deque()

#     while True:
#         with Session() as s:
#             push_testcase_results(s, outbox)
#             assert not len(outbox)
#             inbox = fetch_testcase_batch(s, testbed_ids, tablesets, batch_size)

#         if not len(inbox):
#             return

#         for testcase in inbox:
#             result = testcase.run()
#             outbox.append(result)

# if __name__ == "__main__":
#     parser = ArgumentParser()
#     parser.add_argument("-H", "--hostname", type=str, default=dsmith.MYSQL_HOSTNAME,
#                         help="MySQL database hostname")
#     parser.add_argument("--verbose", action="store_true", help="Verbose execution")
#     args = parser.parse_args()

#     # Initialize database and logging engines:
#     db.init(args.hostname)
#     # TODO: init logging enging

#     testbed_ids = [] # TODO
#     tablesets = [] # TODO

#     runner(testbed_ids, tablesets, args.batch_size)
#     print("done")
