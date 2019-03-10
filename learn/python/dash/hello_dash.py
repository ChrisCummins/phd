"""My first dash app."""
import typing

import dash
import dash_core_components as dcc
import dash_html_components as html
import pandas as pd
from absl import app
from absl import flags

FLAGS = flags.FLAGS


def CreateDataFrame() -> pd.DataFrame:
  """Create a test dataframe."""
  return pd.DataFrame([
      ("foo", 1),
      ("bar", 2),
      ("car", 3),
  ],
                      columns=['label', 'num'])


def CreateTableHtml(df: pd.DataFrame, max_rows: int = 10):
  """Create HTML element for a table."""
  return html.Table(
      # Header
      [html.Tr([html.Th(col) for col in df.columns])] +

      # Body
      [
          html.Tr([html.Td(df.iloc[i][col])
                   for col in df.columns])
          for i in range(min(len(df), max_rows))
      ])


def CreateApp() -> dash.Dash:
  """Create dash app."""
  dash_app = dash.Dash(
      __name__,
      external_stylesheets=["https://codepen.io/chriddyp/pen/bWLwgP.css"])

  dash_app.layout = html.Div(children=[
      html.H1(children="Hello World"),
      html.Div(children=[
          dcc.Markdown(children="""
## Markdown Section.

Rendered markdown.
""")
      ]),
      CreateTableHtml(CreateDataFrame()),
      dcc.Graph(
          id="example-graph",
          figure={
              "data": [
                  {
                      "x": [1, 2, 3],
                      "y": [4, 1, 2],
                      "type": "bar",
                      "name": "SF"
                  },
                  {
                      "x": [1, 2, 3],
                      "y": [2, 4, 5],
                      "type": "bar",
                      "name": "Montréal"
                  },
              ],
              "layout": {
                  "title": "My Graph"
              }
          })
  ])
  return dash_app


def main(argv: typing.List[str]):
  """Main entry point."""
  if len(argv) > 1:
    raise app.UsageError("Unknown arguments: '{}'.".format(" ".join(argv[1:])))

  dash_app = CreateApp()
  dash_app.run_server(debug=True)


if __name__ == "__main__":
  app.run(main)
