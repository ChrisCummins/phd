import re
import subprocess

from labm8.py import fs

_LINE_RE = re.compile("^(?P<count>\d+) instcount - Number of (?P<type>.+)")

DEFAULT_LLVM_PATH = fs.path("~/src/msc-thesis/skelcl/libraries/llvm/build/bin/")


class Error(Exception):
  """
  LLVM module error.
  """
  pass


class ProgramNotFoundError(Error):
  """
  Error thrown if a program is not found.
  """
  pass


class ClangError(Error):
  """
  Error thrown if clang exits with non-zero status.
  """
  pass


class OptError(Error):
  """
  Error thrown if opt exits with non-zero status.
  """
  pass


def assert_program_exists(path):
  """
  Assert that a program exists.

  If the given path does not exist and is not a file, raises
  ProgramNotFoundError.
  """
  if not fs.exists(path) or not fs.isfile(path):
    raise ProgramNotFoundError(path)


def parse_instcount(output):
  """
  Parse the output of
  """
  line_re = re.compile("^(?P<count>\d+) instcount - Number of (?P<type>.+)")
  lines = [x.strip() for x in output.split("\n")]
  out = {}

  # Build a list of counts for each type.
  for line in lines:
    match = re.search(line_re, line)
    if match:
      count = int(match.group("count"))
      key = match.group("type")
      if key in out:
        out[key].append(count)
      else:
        out[key] = [count]

  # Sum all counts.
  for key in out:
    out[key] = sum(out[key])

  return out


def bitcode(source, language="cl", path=DEFAULT_LLVM_PATH):
  assert_program_exists(str(path) + "clang")

  clang_args = [
      str(path) + "clang",
      "-Dcl_clang_storage_class_specifiers",
      "-isystem",
      "libclc/generic/include",
      "-include",
      "clc/clc.h",
      "-target",
      "nvptx64-nvidia-nvcl",
      "-x" + str(language),
      "-emit-llvm",
      "-c",
      "-",  # Read from stdin
      "-o",
      "-"  # Output to stdout
  ]

  clang = subprocess.Popen(clang_args,
                           stdin=subprocess.PIPE,
                           stdout=subprocess.PIPE,
                           stderr=subprocess.PIPE)
  bitcode, err = clang.communicate(source)
  if clang.returncode != 0:
    raise ClangError(err)

  return bitcode


def parse_instcounts(out):
  lines = [x.strip() for x in out.split("\n")]
  counts = {}

  # Build a list of counts for each type.
  for line in lines:
    match = re.search(_LINE_RE, line)
    if match:
      count = int(match.group("count"))
      key = match.group("type")
      if key in counts:
        counts[key].append(count)
      else:
        counts[key] = [count]

  # Sum all counts.
  for key in counts:
    counts[key] = sum(counts[key])

  return counts


def instcounts(bitcode, path=DEFAULT_LLVM_PATH):
  assert_program_exists(str(path) + "opt")

  opt_args = [
      str(path) + "opt",
      "-analyze",
      "-stats",
      "-instcount",
      "-"  # Read from stdin
  ]

  # LLVM pass output pritns to stderr, so we'll pipe stderr to
  # stdout.
  opt = subprocess.Popen(opt_args,
                         stdin=subprocess.PIPE,
                         stdout=subprocess.PIPE,
                         stderr=subprocess.STDOUT)
  out, _ = opt.communicate(bitcode)
  if opt.returncode != 0:
    raise OptError(out)

  return parse_instcounts(out)


def instcounts2ratios(counts):
  ratios = {}

  total_key = "instructions (of all types)"
  total = counts[total_key]
  ratios["instruction_count"] = total

  # Remove total from dict.
  del counts[total_key]

  for key in counts:
    ratio_key = "ratio " + key
    ratio = float(counts[key]) / float(total)
    ratios[ratio_key] = ratio

  return ratios
