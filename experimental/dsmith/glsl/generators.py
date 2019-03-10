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
GLSL program generators.
"""
import math
import random
import string
from time import time

from experimental.dsmith.glsl.db import *
from experimental.dsmith.langs import Generator
from labm8 import fs


class GlslGenerator(Generator):
  """
  Common baseclass for program generators.
  """

  # Abstract methods (must be implemented):
  def generate_one(self, session: session_t) -> ProgramProxy:
    """ Generate a single program. """
    raise NotImplementedError("abstract class")

  # Default methods (may be overriden):

  def __repr__(self):
    return f"{Colors.BOLD}{Colors.GREEN}{self.__name__}{Colors.END}"

  def num_programs(self, session: session_t = None) -> int:
    """ return the number of generated programs in the database """
    with ReuseSession(session) as s:
      return s.query(func.count(Program.id)) \
        .filter(Program.generator == self.id) \
        .scalar()

  def sloc_total(self, session: session_t = None) -> int:
    """ return the total linecount of generated programs """
    with ReuseSession(session) as s:
      return s.query(func.sum(Program.linecount)) \
        .filter(Program.generator == self.id) \
        .scalar()

  def generation_time(self, session: session_t = None) -> float:
    """ return the total generation time of all programs """
    with ReuseSession(session) as s:
      return s.query(func.sum(Program.generation_time)) \
               .filter(Program.generator == self.id) \
               .scalar() or 0

  def num_testcases(self, session: session_t = None) -> int:
    """ return the total number of testcases """
    with ReuseSession(session) as s:
      return s.query(func.count(Testcase.id)) \
        .join(Program) \
        .filter(Program.generator == self.id) \
        .scalar()

  def generate(self, n: int = math.inf, up_to: int = math.inf) -> None:
    """ generate 'n' new programs 'up_to' this many exist in db """
    with Session() as s:
      num_progs = self.num_programs(s)

      # Determine the termination criteria:
      if n == math.inf and up_to == math.inf:
        max_value = math.inf
        bar_max = progressbar.UnknownLength
      elif n == math.inf:
        max_value = up_to
        bar_max = max_value
      else:
        max_value = num_progs + n
        bar_max = max_value

      # Exit early if possible:
      if num_progs >= max_value:
        print(f"There are already {Colors.BOLD}{num_progs}{Colors.END} "
              "programs in the database. Nothing to be done.")
        return

      # Print a preamble message:
      num_to_generate = max_value - num_progs
      if num_to_generate < math.inf:
        estimated_time = (
            self.generation_time(s) / max(num_progs, 1)) * num_to_generate
        eta = humanize.Duration(estimated_time)
        print(f"{Colors.BOLD}{num_to_generate}{Colors.END} programs are "
              "to be generated. Estimated generation time is " +
              f"{Colors.BOLD}{eta}{Colors.END}.")
      else:
        print(f"Generating programs {Colors.BOLD}forever{Colors.END} ...")

      bar = progressbar.ProgressBar(
          initial_value=num_progs, max_value=bar_max, redirect_stdout=True)

      # The actual generation loop:
      buf = []
      while num_progs < max_value:
        buf.append(self.generate_one(s))

        # Update progress bar
        num_progs += 1
        bar.update(num_progs)

        if len(buf) >= dsmith.DB_BUF_SIZE:
          save_proxies_uniq_on(s, buf, "sha1")
          num_progs = self.num_programs(s)
          buf = []
      save_proxies_uniq_on(s, buf, "sha1")
    print(f"All done! You now have {Colors.BOLD}{num_progs}{Colors.END} "
          f"{self} programs in the database")

  def import_from_dir(self, indir: Path) -> None:
    """ import program sources from a directory """
    with Session() as s:
      start_num_progs = self.num_programs(s)

      def _save(proxies):
        # Create records from proxies:
        programs = [proxy.to_record(s) for proxy in proxies]

        logging.warning(getattr(type(programs[0]), "sha1"))

        import sys
        sys.exit(0)

        # Filter duplicates in the set of new records:
        programs = dict(
            (program.sha1, program) for program in programs).values()

        # Fetch a list of dupe keys already in the database:
        sha1s = [program.sha1 for program in programs]
        dupes = set(
            x[0] for x in s.query(Program.sha1).filter(Program.sha1.in_(sha1s)))

        # Filter the list of records to import, excluding dupes:
        uniq = [program for program in programs if program.sha1 not in dupes]

        # Import those suckas:
        s.add_all(uniq)
        s.commit()

        nprog, nuniq = len(programs), len(uniq)
        logging.info(f"imported {nuniq} of {nprog} unique programs")

      num_progs = self.num_programs(s)

      # Print a preamble message:
      paths = fs.ls(indir, abspaths=True)
      num_to_import = humanize.Commas(len(paths))
      print(f"{Colors.BOLD}{num_to_import}{Colors.END} files are "
            "to be imported.")

      bar = progressbar.ProgressBar(redirect_stdout=True)

      # The actual import loop:
      buf = []
      for i, path in enumerate(bar(paths)):
        buf.append(self.import_from_file(s, path))

        if len(buf) >= dsmith.DB_BUF_SIZE:
          save_proxies_uniq_on(s, buf, "sha1")
          buf = []
      save_proxies_uniq_on(s, buf, "sha1")

    num_imported = humanize.Commas(self.num_programs(s) - start_num_progs)
    num_progs = humanize.Commas(self.num_programs(s))
    print(f"All done! Imported {Colors.BOLD}{num_imported}{Colors.END} "
          f"new {self} programs. You now have "
          f"{Colors.BOLD}{num_progs}{Colors.END} {self} programs in the "
          "database")

  def import_from_file(self, session: session_t,
                       path: Path) -> Union[None, ProgramProxy]:
    """ Import a program from a file. """
    # logging.debug(f"importing '{path}'")
    # Simply ignore non-ASCII chars:
    src = ''.join(
        [i if ord(i) < 128 else '' for i in fs.read_file(path).strip()])
    return ProgramProxy(generator=self.id, generation_time=0, src=src)


class RandChar(GlslGenerator):
  """
  This generator produces a uniformly random sequence of ASCII characters, of
  a random length.
  """
  __name__ = "randchar"
  id = Generators.RANDCHAR

  # Arbitrary range
  charcount_range = (100, 100000)

  def generate_one(self, session: session_t) -> ProgramProxy:
    """ Generate a single program. """
    start_time = time()
    charcount = random.randint(*self.charcount_range)
    src = ''.join(random.choices(string.printable, k=charcount))
    runtime = time() - start_time

    return ProgramProxy(generator=self.id, generation_time=runtime, src=src)


class GitHub(GlslGenerator):
  """
  Programs mined from GitHub.
  """
  __name__ = "github"
  id = Generators.GITHUB


class DSmith(GlslGenerator):
  __name__ = "dsmith"
  id = Generators.DSMITH
