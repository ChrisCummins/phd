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
OpenCL program generators.
"""
import math
import random
import string
from pathlib import Path
from tempfile import NamedTemporaryFile
from time import time

from experimental.dsmith.langs import Generator
from experimental.dsmith.opencl import clsmith
from experimental.dsmith.opencl.db import *
from labm8 import fs


class OpenCLGenerator(Generator):
  """
  Common baseclass for OpenCL program generators.
  """

  # Abstract methods (must be implemented):
  def generate_one(self, session: session_t) -> ProgramProxy:
    """ Generate a single program. """
    raise NotImplementedError("abstract class")

  def import_from_file(self, session: session_t, path: Path) -> ProgramProxy:
    """ Import a program from a file. """
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
          save_proxies(s, buf)
          num_progs = self.num_programs(s)
          buf = []
      save_proxies(s, buf)
    print(f"All done! You now have {Colors.BOLD}{num_progs}{Colors.END} "
          "{self} programs in the database")

  def import_from_dir(self, indir: Path) -> None:
    """ import program sources from a directory """
    with Session() as s:
      num_progs = self.num_programs(s)

      # Print a preamble message:
      paths = fs.ls(indir, abspaths=True)
      num_to_import = len(paths)
      print(f"{Colors.BOLD}{num_to_import}{Colors.END} programs are "
            "to be imported.")
      bar_max = num_progs + num_to_import

      bar = progressbar.ProgressBar(
          initial_value=num_progs, max_value=bar_max, redirect_stdout=True)

      # The actual import loop:
      buf = []
      for i, path in enumerate(paths):
        buf.append(self.import_from_file(s, path))

        # Update progress bar
        num_progs += 1
        bar.update(num_progs)

        if len(buf) >= dsmith.DB_BUF_SIZE:
          save_proxies(s, buf)
          num_progs = self.num_programs(s)
          buf = []
      save_proxies(s, buf)
    print(f"All done! Imported {Colors.BOLD}{num_to_import}{Colors.END} "
          f"programs. You now have {Colors.BOLD}{num_progs}{Colors.END} "
          "{self} programs in the database")


class CLSmith(OpenCLGenerator):
  __name__ = "clsmith"
  id = Generators.CLSMITH

  def generate_one(self,
                   session: session_t,
                   attempt: int = 1,
                   max_attempts: int = 10) -> ProgramProxy:
    """ Generate a single CLSmith program. """
    with NamedTemporaryFile(prefix='dsmith-clsmith-', suffix='.cl') as tmp:
      runtime, status, _, stderr = clsmith.clsmith('-o', tmp.name)

      # A non-zero exit status of clsmith implies that no program was
      # generated. Try again:
      if status:
        if attempt > max_attempts:
          app.Error(stderr)
          raise OSError(f"Failed to produce {self} program after "
                        f"{max_attempts} attempts")
        else:
          return self.generate_one(
              session, attempt=attempt + 1, max_attempts=max_attempts)

      src = fs.Read(tmp.name)

    return ProgramProxy(generator=self.id, generation_time=runtime, src=src)


class DSmith(OpenCLGenerator):
  __name__ = "dsmith"
  id = Generators.DSMITH

  def generate_one(self, session: session_t) -> ProgramProxy:
    """ Generate a single program. """
    raise NotImplementedError

  def import_from_file(self, session: session_t, path: Path) -> ProgramProxy:
    """ Import a program from a file. """
    raise NotImplementedError


class RandChar(OpenCLGenerator):
  """
  This generator produces a uniformly random sequence of ASCII characters, of
  a random length.
  """
  __name__ = "randchar"
  id = Generators.RANDCHAR

  # This is the hardcoded range of kernel lengths found in the GitHub corpus
  # (after preprocessing).
  charcount_range = (33, 451563)

  def generate_one(self, session: session_t) -> ProgramProxy:
    """ Generate a single program. """
    start_time = time()
    charcount = random.randint(*self.charcount_range)
    src = ''.join(random.choices(string.printable, k=charcount))
    runtime = time() - start_time

    return ProgramProxy(generator=self.id, generation_time=runtime, src=src)


class RandTok(OpenCLGenerator):
  __name__ = "randtok"
  id = Generators.RANDTOK

  def generate_one(self, session: session_t) -> Program:
    """ Generate a single program. """
    raise NotImplementedError
