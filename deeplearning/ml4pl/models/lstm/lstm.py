"""LSTM sequential classifier models."""
from deeplearning.ml4pl.graphs.labelled import graph_tuple_database
from deeplearning.ml4pl.models import run
from deeplearning.ml4pl.models.lstm import graph_lstm
from deeplearning.ml4pl.models.lstm import node_lstm
from labm8.py import app

FLAGS = app.FLAGS


def main():
  """Main entry point."""
  graph_db: graph_tuple_database.Database = FLAGS.graph_db()
  if graph_db.node_y_dimensionality:
    model_class = node_lstm.NodeLstm
  else:
    model_class = graph_lstm.GraphLstm

  run.Run(model_class, graph_db)


if __name__ == "__main__":
  app.Run(main)
