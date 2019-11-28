#!/usr/bin/env python3
# from dsmith.db import CLgenProgram, Session
# if __name__ == "__main__":
#     parser = ArgumentParser()
#     parser.add_argument("-H", "--hostname", type=str, default="cc1",
#                         help="MySQL database hostname")
#     args = parser.parse_args()
#     db.init(args.hostname)
#     with Session(commit=True) as s:
#         q = s.query(CLgenProgram).filter(CLgenProgram.runtime == None)
#         for program in ProgressBar(max_value=q.count())(q):
#             program.runtime = len(program.src) / clgen_fetch.CLGEN_INFERENCE_CPS
#     print("done.")
