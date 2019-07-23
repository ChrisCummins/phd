"""Jasper is a small command line utility to run long-running MySQL queries.

It combines a `git commit` style query prompt with `lmk` so that you receive
an email when the query terminates.

Usage:

  $ bazel run //util/jasper -- --host=my.server.com
"""
import os
import sys

import datetime
import pathlib
import subprocess
import tempfile
import typing

from labm8 import app
from labm8 import fs
from labm8 import text
from tools.code_style.linters import linters_lib
from util.lmk import lmk

FLAGS = app.FLAGS
app.DEFINE_string('host', 'localhost',
                  'The MySQL host to execute the command on.')


def runEditorOnFileOrDie(path: pathlib.Path) -> None:
  """Run the user's text editor on the file.

  The text editor is determined by $VISUAL or $EDITOR.

  Arg:
    path: The path to edit.
  """
  editor = os.environ.get('VISUAL', os.environ.get('EDITOR'))
  if not editor:
    print('fatal: $EDITOR not set', file=sys.stderr)
    sys.exit(1)

  cmd = f'{editor} {path}'
  try:
    subprocess.check_call(cmd, shell=True)
  except subprocess.SubprocessError:
    print('fatal: Editor `{cmd}` failed. Aborting', file=sys.stderr)
    sys.exit(1)


def getQueryFromUserOrDie(
    edit_file_callback: typing.Callable[[pathlib.Path], None]) -> str:
  """Get the query to run from the user.

  This prompts the user for a query to run, then formats it.

  Args:
    edit_file_callback: A callback function which invokes the user's editor on
      a file.

  Returns:
    The query to execute.
  """
  query = """
# Please enter the SQL query to execute. Comments beginning with 
# '//', '#', and '--' are ignored. An empty query executes nothing.
"""
  with tempfile.TemporaryDirectory(prefix='phd_jasper_') as d:
    query_file = pathlib.Path(d) / 'query.sql'
    fs.Write(query_file, query.encode('utf-8'))
    edit_file_callback(query_file)
    query = fs.Read(query_file)

    # Strip comments and trailing whitespace.
    query = text.StripSingleLineComments(query, '(#|//|--)')
    query = '\n'.join(
        [x.rstrip() for x in query.split('\n') if x.strip()]).strip()

    # Format the SQL.
    fs.Write(query_file, query.encode('utf-8'))
    linters_lib.SqlFormat.Lint(query_file)
    query = fs.Read(query_file)

  if not query:
    print("fatal: No query to execute", file=sys.stderr)
    sys.exit(1)
  return query


def execMysqlQuery(query: str, host: str) -> None:
  """Run the given MySQL query.

  Args:
    query: The query to run.

  Returns:
    The output of MySQL.

  Raises:
    OSError: If MySQL fails.
  """
  process = subprocess.Popen(['mysql', '-h', host, '--table'],
                             stdin=subprocess.PIPE,
                             stdout=subprocess.PIPE,
                             stderr=subprocess.PIPE,
                             universal_newlines=True)
  stdout, stderr = process.communicate(query)
  if process.returncode:
    raise OSError(f"MySQL terminated with returncode {process.returncode} and "
                  f"output:\n{stdout}\n{stderr}")
  return stdout


def getQueryAndExecuteAndNotify(host: str):
  """Main entrypoint: fetch a query and execute it on the given host."""
  date_started = datetime.datetime.now()
  query = getQueryFromUserOrDie(runEditorOnFileOrDie)

  message_to_print = f'{datetime.datetime.now()}\n\n{query}'
  print('\n'.join([f'-- {x}' for x in message_to_print.split('\n')]))

  try:
    output = execMysqlQuery(query, FLAGS.host)
    print(output)
    lmk.let_me_know(output,
                    command=query,
                    returncode=0,
                    date_started=date_started,
                    date_ended=datetime.datetime.now())
  except OSError as e:
    print(e, file=sys.stderr)
    lmk.let_me_know(str(e),
                    command=query,
                    returncode=1,
                    date_started=date_started,
                    date_ended=datetime.datetime.now())
    sys.exit(1)


def main():
  """Main entry point."""
  getQueryAndExecuteAndNotify(FLAGS.host)


if __name__ == '__main__':
  app.Run(main)
