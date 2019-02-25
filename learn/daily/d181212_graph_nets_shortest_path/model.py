"""A model for predicting the shortest path in a graph."""
from absl import flags

FLAGS = flags.FLAGS


class ShortestPathModel(object):
  """A model for predicting shortest paths.

  Based on DeepMind's graph net example colab notebook:
  https://colab.research.google.com/github/deepmind/graph_nets/blob/master/graph_nets/demos/shortest_path.ipynb

  The model we explore includes three components:
  - An "Encoder" graph net, which independently encodes the edge, node, and
    global attributes (does not compute relations etc.).
  - A "Core" graph net, which performs N rounds of processing (message-passing)
    steps. The input to the Core is the concatenation of the Encoder's output
    and the previous output of the Core (labeled "Hidden(t)" below, where "t" is
    the processing step).
  - A "Decoder" graph net, which independently decodes the edge, node, and
    global attributes (does not compute relations etc.), on each
    message-passing step.

                      Hidden(t)   Hidden(t+1)
                         |            ^
            *---------*  |  *------*  |  *---------*
            |         |  |  |      |  |  |         |
  Input --->| Encoder |  *->| Core |--*->| Decoder |---> Output(t)
            |         |---->|      |     |         |
            *---------*     *------*     *---------*

  The model is trained by supervised learning. Input graphs are procedurally
  generated, and output graphs have the same structure with the nodes and edges
  of the shortest path labeled (using 2-element 1-hot vectors). We could have
  predicted the shortest path only by labeling either the nodes or edges, and
  that does work, but we decided to predict both to demonstrate the flexibility
  of graph nets' outputs.

  The training loss is computed on the output of each processing step. The
  reason for this is to encourage the model to try to solve the problem in as
  few steps as possible. It also helps make the output of intermediate steps
  more interpretable.

  There's no need for a separate evaluate dataset because the inputs are
  never repeated, so the training loss is the measure of performance on graphs
  from the input distribution.

  We also evaluate how well the models generalize to graphs which are up to
  twice as large as those on which it was trained. The loss is computed only
  on the final processing step.

  Variables with the suffix _tr are training parameters, and variables with the
  suffix _ge are test/generalization parameters.

  After around 2000-5000 training iterations the model reaches near-perfect
  performance on graphs with between 8-16 nodes.
  """

  def __init__(self, name: str = "shortest_path_gnn"):
    """Instantiate a model."""
    super(ShortestPathModel, self).__init__(name=name)
    raise NotImplementedError

  def _build(self):
    raise NotImplementedError
