"""Jasper is a small command line utility to run MySQL queries.

It combines a `git commit` style query prompt with `lmk` so that you receive
an email when the query terminates.

There are three ways for Jasper to obtain the query(s) to execute. The first
is by interactively prompting the user with their chosen $EDITOR:

  $ jasper --host=my.server.com

Alternatively, Jasper can read from standard input:

  $ echo "show processlist" < jasper --host=my.server.com

Finally, Jasper can read files provided as positional arguments:

  $ jasper --host=my.server.com query1.sql query2.sql query3.sql

In this final case, each query will be executed in turn.
"""
import datetime
import os
import pathlib
import select
import subprocess
import sys
import tempfile
import typing

from labm8 import app
from labm8 import fs
from labm8 import text
from tools.code_style.formatters import sql_formatter
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
    print(f'fatal: Editor `{cmd}` failed. Aborting', file=sys.stderr)
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

    query = sql_formatter.FormatSql(query)

  if not query:
    print('No query to execute, aborting.', file=sys.stderr)
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
    raise OSError(f'MySQL terminated with returncode {process.returncode} and '
                  f'output:\n{stdout}\n{stderr}')
  return stdout


def executeQueryAndNotify(query: str, host: str) -> bool:
  """Execute the query, send a notification, and return True on error."""
  date_started = datetime.datetime.now()
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
    return False
  except OSError as e:
    print(e, file=sys.stderr)
    lmk.let_me_know(str(e),
                    command=query,
                    returncode=1,
                    date_started=date_started,
                    date_ended=datetime.datetime.now())
    return True


def readFileOrDie(arg: str) -> str:
  """Read contents of file to string or die."""
  path = pathlib.Path(arg)
  if not path.is_file():
    print(f'fatal: file not found {arg}', file=sys.stderr)
    sys.exit(1)
  with open(path) as f:
    data = f.read()
  return data


def main(argv: typing.List[str]):
  """Main entry point."""
  error = False
  # There are three ways for jasper to get the query(s) to execute:
  #
  #  1. Positional arguments, which are interpreted as file paths. If there are
  #     multiple arguments, they are executed sequentially.
  #  2. Standard input, which, if present, will be read and executed as a
  #     query, such as "echo 'select * from foo' | jasper".
  #  3. (default) If neither standard input or positional arguments are
  #     provided, the user is prompted for a query.
  if len(argv) >= 2:
    for arg in argv[1:]:
      query = readFileOrDie(arg)
      error |= executeQueryAndNotify(query, FLAGS.host)
  elif select.select([
      sys.stdin,
  ], [], [], 0.0)[0]:
    query = sys.stdin.read()
    error = executeQueryAndNotify(query, FLAGS.host)
  else:
    query = getQueryFromUserOrDie(runEditorOnFileOrDie)
    error = executeQueryAndNotify(query, FLAGS.host)

  sys.exit(1 if error else 0)


if __name__ == '__main__':
  app.RunWithArgs(main)
