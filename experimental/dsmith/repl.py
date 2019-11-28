#
# Copyright 2017, 2018 Chris Cummins <chrisc.101@gmail.com>.
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
Parser for dsmith

Attributes:
    __help__ (str): REPL help string.
    __available_commands__ (str): Help string for available commands.
"""
import datetime
import math
import os
import random
import re
import sys
import traceback

from experimental import dsmith
from experimental.dsmith import Colors
from experimental.dsmith.langs import Generator
from experimental.dsmith.langs import Language
from experimental.dsmith.langs import mklang
from labm8.py import fs
from labm8.py import humanize

_lang_str = f"{Colors.RED}<lang>{Colors.END}{Colors.BOLD}"
_generator_str = f"{Colors.GREEN}<generator>{Colors.END}{Colors.BOLD}"
_harness_str = f"{Colors.YELLOW}<harness>{Colors.END}{Colors.BOLD}"
_testbed_str = f"{Colors.PURPLE}<testbed>{Colors.END}{Colors.BOLD}"
_num_str = f"{Colors.BLUE}<number>{Colors.END}{Colors.BOLD}"

__available_commands__ = f"""\
  {Colors.BOLD}describe [all]{Colors.END}
    Provide an overview of stored data.

  {Colors.BOLD}describe {_lang_str} {{{Colors.GREEN}generators{Colors.END}{Colors.BOLD},
  programs}}{Colors.END}
    Provide details about generators, or generated programs.

  {Colors.BOLD}describe [available] {_lang_str} [{_harness_str}] {Colors.PURPLE}testbeds{Colors.END}
    Provide details about the available testbeds, or all the testbeds in
    the database.

  {Colors.BOLD}describe {_lang_str} [{_generator_str}] {{testcases,results}}{Colors.END}
    Provide details about testcases, or results.

  {Colors.BOLD}make [[up to] {_num_str}] {_lang_str} programs [using {_generator_str}]{Colors.END}
    Generate the specified number of programs. If no generator is specified,
    default to dsmith.

  {Colors.BOLD}import {_generator_str} {_lang_str} programs from {Colors.BOLD}{Colors.BLUE}<dir>{
  Colors.END}
    Import programs from a directory.

  {Colors.BOLD}make {_lang_str} [[{_harness_str}]:[{_generator_str}]] testcases{Colors.END}
    Prepare testcases from programs.

  {Colors.BOLD}run {_lang_str} [[{_harness_str}]:[{_generator_str}]] testcases [on {
  _testbed_str}]{Colors.END}
    Run testcases. If no generator is specified, run testcases from all
    generators. If no testbed is specified, use all available testbeds.

  {Colors.BOLD}difftest {_lang_str} [[{_harness_str}]:[{_generator_str}]] results{Colors.END}
    Compare results across devices.

  {Colors.BOLD}test{Colors.END}
    Run the self-test suite.

  {Colors.BOLD}version{Colors.END}
    Print version information.

  {Colors.BOLD}exit{Colors.END}
    End the session.\
"""

__help__ = f"""\
This is the DeepSmith interactive session. The following commands are available:

{__available_commands__}
"""


class UnrecognizedInput(ValueError):
  pass


def _hello(file=sys.stdout):
  print("Hi there!", file=file)


def _help(file=sys.stdout):
  print(__help__, file=file)


def _version(file=sys.stdout):
  print(dsmith.__version_str__, file=file)


def _test(file=sys.stdout):
  import dsmith.test

  dsmith.test.testsuite()


def _exit(*args, **kwargs):
  file = kwargs.pop("file", sys.stdout)

  farewell = random.choice(
    [
      "Have a nice day!",
      "Over and out.",
      "God speed.",
      "See ya!",
      "See you later alligator.",
    ]
  )
  print(f"{Colors.END}{farewell}", file=file)
  sys.exit()


def _describe_generators(lang: Language, file=sys.stdout):
  gen = ", ".join(
    f"{Colors.BOLD}{generator}{Colors.END}" for generator in lang.generators
  )
  print(f"The following {lang} generators are available: {gen}.", file=file)


def _describe_programs(lang: Language, file=sys.stdout):
  for generator in lang.generators:
    num = humanize.Commas(generator.num_programs())
    sloc = humanize.Commas(generator.sloc_total())
    print(
      f"You have {Colors.BOLD}{num} {generator}{Colors.END} "
      f"programs, total {Colors.BOLD}{sloc}{Colors.END} SLOC.",
      file=file,
    )


def _describe_testcases(lang: Language, generator: Generator, file=sys.stdout):
  for harness in generator.harnesses:
    num = humanize.Commas(generator.num_testcases())
    print(
      f"There are {Colors.BOLD}{num} {generator}:{harness} " "testcases.",
      file=file,
    )


def _make_programs(
  lang: Language,
  generator: Generator,
  n: int,
  up_to: bool = False,
  file=sys.stdout,
):
  up_to_val = n if up_to else math.inf
  n = math.inf if up_to else n
  generator.generate(n=n, up_to=up_to_val)


def _execute(statement: str, file=sys.stdout) -> None:
  if not isinstance(statement, str):
    raise TypeError

  # parsing is case insensitive
  statement = re.sub("\s+", " ", statement.strip().lower())
  components = statement.split(" ")

  if not statement:
    return

  # Parse command modifiers:
  if components[0] == "debug":
    statement = re.sub(r"^debug ", "", statement)
    with dsmith.debug_scope():
      return _execute(statement, file=file)
  elif components[0] == "verbose":
    components = components[1:]
    statement = re.sub(r"^verbose ", "", statement)
    with dsmith.verbose_scope():
      return _execute(statement, file=file)

  csv = ", ".join(f"'{x}'" for x in components)
  app.Log(2, f"parsing input [{csv}]")

  # Full command parser:
  if len(components) == 1 and re.match(r"(hi|hello|hey)", components[0]):
    return _hello(file=file)

  if len(components) == 1 and re.match(r"(exit|quit)", components[0]):
    return _exit(file=file)

  if len(components) == 1 and components[0] == "help":
    return _help(file=file)

  if len(components) == 1 and components[0] == "version":
    return _version(file=file)

  if len(components) == 1 and components[0] == "test":
    return _test(file=file)

  if components[0] == "describe":
    generators_match = re.match(
      r"describe (?P<lang>\w+) generators$", statement
    )
    testbeds_match = re.match(
      r"describe (?P<available>available )?(?P<lang>\w+) testbeds$", statement
    )
    programs_match = re.match(r"describe (?P<lang>\w+) programs$", statement)
    testcases_match = re.match(
      r"describe (?P<lang>\w+) ((?P<generator>\w+) )?testcases$", statement
    )
    results_match = re.match(r"describe (?P<lang>\w+) results$", statement)

    if generators_match:
      lang = mklang(generators_match.group("lang"))
      return _describe_generators(lang=lang, file=file)
    elif testbeds_match:
      lang = mklang(testbeds_match.group("lang"))
      available_only = True if testbeds_match.group("available") else False
      return lang.describe_testbeds(available_only=available_only, file=file)
    elif programs_match:
      lang = mklang(programs_match.group("lang"))
      return _describe_programs(lang=lang, file=file)
    elif testcases_match:
      lang = mklang(testcases_match.group("lang"))
      gen = testcases_match.group("generator")
      if gen:
        generator = lang.mkgenerator(gen)
        return _describe_testcases(lang=lang, generator=generator, file=file)
      else:
        for generator in lang.generators:
          _describe_testcases(lang=lang, generator=generator, file=file)
        return
    elif results_match:
      lang = mklang(results_match.group("lang"))
      return lang.describe_results(file=file)
    else:
      raise UnrecognizedInput

  if components[0] == "make":
    programs_match = re.match(
      r"make ((?P<up_to>up to )?(?P<number>\d+) )?(?P<lang>\w+) program(s)?( using ("
      r"?P<generator>\w+))?$",
      statement,
    )
    testcases_match = re.match(
      r"make (?P<lang>\w+) ((?P<harness>\w+):(?P<generator>\w+)? )?testcases$",
      statement,
    )

    if programs_match:
      number = int(programs_match.group("number") or 0) or math.inf
      lang = mklang(programs_match.group("lang"))
      generator = lang.mkgenerator(programs_match.group("generator"))

      return _make_programs(
        lang=lang,
        generator=generator,
        n=number,
        up_to=True if programs_match.group("up_to") else False,
        file=file,
      )

    elif testcases_match:
      lang = mklang(testcases_match.group("lang"))
      if testcases_match.group("harness"):
        harness = lang.mkharness(testcases_match.group("harness"))
        if testcases_match.group("generator"):
          generators = [lang.mkgenerator(testcases_match.group("generator"))]
        else:
          # No generator specified, use all:
          generators = list(harness.generators)

        for generator in generators:
          harness.make_testcases(generator)
      else:
        # No harness specified, use all:
        for harness in lang.harnesses:
          for generator in harness.generators:
            harness.make_testcases(generator)
      return
    else:
      raise UnrecognizedInput

  if components[0] == "import":
    match = re.match(
      r"import (?P<generator>\w+) (?P<lang>\w+) program(s)? from (?P<path>.+)$",
      statement,
    )

    if match:
      lang = mklang(match.group("lang"))
      generator = lang.mkgenerator(match.group("generator"))
      path = fs.abspath(match.group("path"))
      if not fs.isdir(path):
        raise ValueError(f"'{path}' is not a directory")

      return generator.import_from_dir(path)
    else:
      raise UnrecognizedInput

  if components[0] == "run":
    match = re.match(
      r"run (?P<lang>\w+) ((?P<harness>\w+):(?P<generator>\w+)? )?testcases( on (?P<testbed>["
      r"\w+-±]+))?$",
      statement,
    )
    if match:
      lang = mklang(match.group("lang"))

      if match.group("harness"):
        harness = lang.mkharness(match.group("harness"))
        if match.group("generator"):
          generators = [lang.mkgenerator(match.group("generator"))]
        else:
          # No generator specified, use all:
          generators = list(harness.generators)

        pairs = [(harness, generator) for generator in generators]
      else:
        pairs = []
        # No harness specified, use all:
        for harness in lang.harnesses:
          pairs += [(harness, generator) for generator in harness.generators]

      for harness, generator in pairs:
        if match.group("testbed"):
          testbeds = lang.mktestbeds(match.group("testbed"))
        else:
          testbeds = harness.available_testbeds()

        for testbed in testbeds:
          testbed.run_testcases(harness, generator)
      return
    else:
      raise UnrecognizedInput

  if components[0] == "difftest":
    match = re.match(r"difftest (?P<lang>\w+) results$", statement)
    lang = mklang(match.group("lang"))

    return lang.difftest()

  raise UnrecognizedInput


def _user_message_with_stacktrace(exception):
  # get limited stack trace
  def _msg(i, x):
    n = i + 1

    filename = fs.basename(x[0])
    lineno = x[1]
    fnname = x[2]

    loc = "{filename}:{lineno}".format(**vars())
    return "      #{n}  {loc: <18} {fnname}()".format(**vars())

  _, _, tb = sys.exc_info()
  NUM_ROWS = 5  # number of rows in traceback

  trace = reversed(traceback.extract_tb(tb, limit=NUM_ROWS + 1)[1:])
  stack_trace = "\n".join(_msg(*r) for r in enumerate(trace))
  typename = type(exception).__name__

  print(
    f"""
======================================================================
💩  Fatal error!
{exception} ({typename})

  stacktrace:
{stack_trace}

Please report bugs at <https://github.com/ChrisCummins/dsmith/issues>\
""",
    file=sys.stderr,
  )


def run_command(command: str, file=sys.stdout) -> None:
  """
  Pseudo-natural language command parsing.
  """
  try:
    _execute(command, file=file)
  except UnrecognizedInput as e:
    print(
      "😕  I don't understand. " "Type 'help' for available commands.", file=file
    )
    if os.environ.get("DEBUG"):
      raise e
  except NotImplementedError as e:
    print("🤔  I don't know how to do that (yet).", file=file)
    if os.environ.get("DEBUG"):
      raise e
  except KeyboardInterrupt:
    print("", file=file)
    _exit(file=file)
  except Exception as e:
    _user_message_with_stacktrace(e)
    if os.environ.get("DEBUG"):
      raise e


def repl(file=sys.stdout) -> None:
  hour = datetime.datetime.now().hour

  greeting = "Good evening."
  if hour > 4:
    greeting = "Good morning."
  if hour > 12 and hour < 18:
    greeting = "Good afternoon."

  print(greeting, "Type 'help' for available commands.", file=file)

  while True:
    sys.stdout.write(f"{Colors.BOLD}> ")
    choice = input()
    sys.stdout.write(Colors.END)
    sys.stdout.flush()

    # Strip '#' command, and split ';' separated commands
    commands = choice.split("#")[0].split(";")

    for command in commands:
      run_command(command, file=file)
