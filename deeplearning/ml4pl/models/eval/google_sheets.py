"""Module for working with google sheets."""
import pathlib

import gspread
import gspread_dataframe
import pandas as pd
from oauth2client import service_account

from labm8.py import app

app.DEFINE_input_path(
    'google_sheets_credentials',
    '/var/phd/deeplearning/ml4pl/google_sheets_credentials.json',
    'The path to a google service account credentials JSON file.')

FLAGS = app.FLAGS


class GoogleSheets():

  def __init__(self, credentials_file: pathlib.Path):
    scope = [
        'https://spreadsheets.google.com/feeds',
        'https://www.googleapis.com/auth/drive'
    ]

    credentials = (
        service_account.ServiceAccountCredentials.from_json_keyfile_name(
            str(credentials_file), scope))

    self._connection = gspread.authorize(credentials)

  def GetOrCreateSpreadsheet(self, name: str, share_with_email_address: str):
    """Return the speadsheet with the given name, creating it if necessary
    and sharing it with the given email address."""
    try:
      sheet = self._connection.open(name)
    except gspread.exceptions.SpreadsheetNotFound:
      sheet = self._connection.create(name)
      sheet.share(share_with_email_address, perm_type='user', role='writer')
    return sheet

  @staticmethod
  def GetOrCreateWorksheet(sheet, name: str):
    """Return the worksheet with the given name, creating it if necessary."""
    try:
      return sheet.worksheet(name)
    except gspread.exceptions.WorksheetNotFound:
      return sheet.add_worksheet(title=name, rows=1, cols=1)

  @staticmethod
  def ExportDataFrame(worksheet, df: pd.DataFrame, index: bool = True) -> None:
    """Export the given dataframe to a worksheet."""

    gspread_dataframe.set_with_dataframe(worksheet,
                                         df,
                                         include_index=index,
                                         resize=True)

  @classmethod
  def CreateFromFlagsOrDie(cls) -> 'GoogleSheets':
    try:
      return cls(FLAGS.google_sheets_credentials)
    except Exception as e:
      app.FatalWithoutStackTrace("%s", e)
