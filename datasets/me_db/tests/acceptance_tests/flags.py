"""Flags for :acceptance_tests.py.

A quirk in the combination of pytest and absl flags is that you can't define
a flag in the same file that you invoke pytest.main(). This is because the
pytest collector re-imports the file, causing absl to error because the flags
have already been defined.
"""

from labm8 import app

FLAGS = app.FLAGS

app.DEFINE_string(
    'me_db_acceptance_tests_inbox', None,
    'If set, this sets the inbox path to be used by the '
    'acceptance test suite. This overrides the default path of '
    '//datasets/me_db/tests/test_inbox.')
