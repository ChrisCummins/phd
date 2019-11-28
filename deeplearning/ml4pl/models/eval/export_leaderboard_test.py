"""This file contains TODO: one line summary.

TODO: Detailed explanation of the file.
"""
from labm8.py import app

FLAGS = app.FLAGS

requires_google_sheets_credentials_file = pytest.mark.skipif(
  not FLAGS.credentials.is_file(), reason="google sheets credentials not found",
)
