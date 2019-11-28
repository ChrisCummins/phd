"""Export a dataset for use in training and testing discriminators."""
import configparser
import pathlib
import typing

import MySQLdb

from experimental.deeplearning.fish.proto import fish_pb2
from labm8.py import app
from labm8.py import fs
from labm8.py import humanize
from labm8.py import pbutil

FLAGS = app.FLAGS

app.DEFINE_string(
  "export_path", None, "Directory to write training dataset to."
)


def _SetIf(
  out: typing.Dict[str, typing.Any],
  key: typing.Any,
  value: typing.Any,
  setvalue: typing.Any = None,
) -> typing.Dict[str, typing.Any]:
  if value:
    out[key] = setvalue or value
  return out


def GetClangAssertionStub(assertion_text):
  return ":".join(assertion_text.split(":")[3:])


def ExportOpenCLResults(cursor, start_id, proto_dir):
  batch_size = 1000
  result_id = start_id
  while True:
    app.Log(1, "Exporting batch of %s results", humanize.Commas(batch_size))
    cursor.execute(
      """
SELECT
  results.id,
  assertions.assertion,
  results.outcome,
  programs.src
FROM results
LEFT JOIN testbeds ON results.testbed_id = testbeds.id
LEFT JOIN platforms ON testbeds.platform_id = platforms.id
LEFT JOIN testcases ON results.testcase_id = testcases.id
LEFT JOIN programs ON testcases.program_id = programs.id
LEFT JOIN stderrs ON results.stderr_id = stderrs.id
LEFT JOIN assertions ON stderrs.assertion_id = assertions.id
WHERE results.id >= %s
AND programs.generator = 1
AND testbeds.id = (
  SELECT testbeds.id
    FROM testbeds
    LEFT JOIN platforms ON testbeds.platform_id=platforms.id
  WHERE platform = 'clang'
  AND driver = '3.6.2'
)
ORDER BY results.id
LIMIT %s
""",
      (result_id, batch_size),
    )
    i = 0
    for row in cursor:
      i += 1
      (result_id, assertion_text, outcome_num, program_src,) = row

      outcome = fish_pb2.CompilerCrashDiscriminatorTrainingExample.Outcome.Name(
        outcome_num
      ).lower()
      proto = fish_pb2.CompilerCrashDiscriminatorTrainingExample(
        src=program_src,
        outcome=outcome_num,
        raised_assertion=True if assertion_text else False,
        assertion_name=(
          GetClangAssertionStub(assertion_text) if assertion_text else ""
        ),
      )
      pbutil.ToFile(proto, proto_dir / outcome / (str(result_id) + ".pbtxt"))

    # If we received fewer results than the requested batch size, then we have
    # ran out of data.
    if i < batch_size:
      return


def GetMySqlCredentials():
  """Read MySQL credentials from ~/.my.cnf."""
  cfg = configparser.ConfigParser()
  cfg.read(pathlib.Path("~/.my.cnf").expanduser())
  return cfg["mysql"]["user"], cfg["mysql"]["password"]


def main(argv):
  """Main entry point."""
  if len(argv) > 1:
    raise app.UsageError("Unknown arguments: '{}'.".format(" ".join(argv[1:])))

  if not FLAGS.export_path:
    raise app.UsageError("--export_path must be a directory")
  export_path = pathlib.Path(FLAGS.export_path)
  if export_path.is_file():
    raise app.UsageError("--export_path must be a directory")

  # Make a directory for each outcome class.
  for key in fish_pb2.CompilerCrashDiscriminatorTrainingExample.Outcome.keys():
    (export_path / key.lower()).mkdir(parents=True, exist_ok=True)

  app.Log(1, "Connecting to MySQL database")
  credentials = GetMySqlCredentials()
  cnx = MySQLdb.connect(
    database="dsmith_04_opencl",
    host="cc1",
    user=credentials[0],
    password=credentials[1],
  )
  cursor = cnx.cursor()
  app.Log(1, "Determining last export ID")
  ids = sorted(
    [
      int(pathlib.Path(f).stem)
      for f in fs.lsfiles(export_path, recursive=True, abspaths=True)
    ]
  )
  last_export_id = ids[-1] if ids else 0
  app.Log(1, "Exporting results from ID %s", last_export_id)
  ExportOpenCLResults(cursor, last_export_id, export_path)
  cursor.close()
  cnx.close()
  app.Log(
    1,
    "Exported training set of %s files to %s",
    humanize.Commas(len(list(export_path.iterdir()))),
    export_path,
  )


if __name__ == "__main__":
  app.RunWithArgs(main)
