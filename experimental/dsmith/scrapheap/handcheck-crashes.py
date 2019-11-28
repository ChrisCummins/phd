#!/usr/bin/env python
import random
import sys
from argparse import ArgumentParser

from dsmith import db
from dsmith.db import *


def yes_no_or_skip(question, default="skip"):
  """Ask a yes/no/skip question via input() and return their answer.

  "question" is a string that is presented to the user.
  "default" is the presumed answer if the user just hits <Enter>.
      It must be "yes" (the default), "no" or None (meaning
      an answer is required of the user).

  The "answer" return value is True for "yes" or False for "no".
  """
  valid = {
    "yes": "yes",
    "y": "yes",
    "ye": "yes",
    "no": "no",
    "n": "no",
    "skip": "skip",
    "ski": "skip",
    "sk": "skip",
    "s": "skip",
  }
  if default is None:
    prompt = "[y/n/s]"
  elif default == "yes":
    prompt = "[Y/n/s]"
  elif default == "no":
    prompt = "[y/N/s]"
  elif default == "skip":
    prompt = "[y/n/S]"
  else:
    raise ValueError("invalid default answer: '%s'" % default)

  while True:
    sys.stdout.write(f"{question} {prompt} ")
    choice = input().lower()
    if default is not None and choice == "":
      return valid[default]
    elif choice in valid:
      return valid[choice]
    else:
      sys.stdout.write(f"Invalid input, select form {prompt}.\n")


def handcheck(recheck=False, include_all=False):
  program = None
  with Session() as session:
    q = (
      session.query(CLgenProgram)
      .distinct()
      .join(
        cl_launcherCLgenResult,
        cl_launcherCLgenResult.program_id == CLgenProgram.id,
      )
      .filter(CLgenProgram.gpuverified == 1)
    )

    if not include_all:
      q = q.filter(
        cl_launcherCLgenResult.status == 0,
        cl_launcherCLgenResult.classification == "Wrong code",
      )

    if not recheck:
      q = q.filter(CLgenProgram.handchecked == None)

    num_todo = q.count()
    if num_todo:
      program = q.limit(1).offset(random.randint(0, num_todo - 1)).first()

      print()
      print(f"{num_todo} kernels to check")
      print("=====================================")
      print(program.src)
      print()
      answer = yes_no_or_skip("Is this a valid kernel?")
      if answer == "skip":
        print("skip")
      else:
        valid = answer == "yes"
        print(valid)
        print()
        program.handchecked = 1 if valid else 0

  # next check
  if program:
    handcheck(recheck=recheck, include_all=include_all)


def main():
  parser = ArgumentParser(description="Collect difftest results for a device")
  parser.add_argument(
    "-H", "--hostname", type=str, default="cc1", help="MySQL database hostname"
  )
  parser.add_argument(
    "-r",
    "--recheck",
    action="store_true",
    help="include previously checked kernels",
  )
  parser.add_argument(
    "-a",
    "--all",
    dest="include_all",
    action="store_true",
    help="include all kernels, not just wrong-code",
  )
  args = parser.parse_args()

  # get testbed information
  db_hostname = args.hostname
  db_url = db.init(db_hostname)

  try:
    handcheck(recheck=args.recheck, include_all=args.include_all)
    print("done.")
  except KeyboardInterrupt:
    print("\nthanks for playing")


if __name__ == "__main__":
  main()
