"""Train and evaluate a model for graph classification.

Usage:
    ggnn_classifyapp_model.py [options]

Options:
    -h --help                Show this screen.
    --config-file FILE       Hyperparameter configuration file path (in JSON format).
    --config CONFIG          Hyperparameter configuration dictionary (in JSON format).
    --log_dir DIR            Log dir name.
    --data_dir DIR           Data dir name.
    --restore FILE           File to restore weights from.
    --freeze-graph-model     Freeze weights of graph model components.
    --evaluate               test evaluation mode using a restored model
"""
from collections import defaultdict, namedtuple

import numpy as np
import tensorflow as tf
from deeplearning.ml4pl.models.ggnn.utils import MLP, SMALL_NUMBER, glorot_init
from docopt import docopt
from typing import Any, Dict, Sequence, Tuple  # noqa

from deeplearning.ml4pl.models.ggnn import ggnn_base
from labm8 import app


FLAGS = app.FLAGS

GGNNWeights = namedtuple(
    "GGNNWeights",
    [
        "edge_weights",
        "edge_weights_for_embs",
        "edge_biases",
        "edge_biases_for_embs",
        "edge_type_attention_weights",
        "rnn_cells",
    ],
)


class GGNNClassifyappModel(ggnn_base.GGNNBaseModel):

  def __init__(self, args):
    super().__init__(args)

  @classmethod
  def default_params(cls):
    params = dict(super().default_params())
    params.update({
        "use_edge_bias": False,
        "use_propagation_attention": False,
        "use_edge_msg_avg_aggregation": True,
        "residual_connections":
        {  # For layer i, specify list of layers whose output is added as an input
            #"4": [0, 2]
        },
        "layer_timesteps":
        [2, 2, 2],  # number of layers & propagation steps per layer
        # "layer_timesteps": [2, 2, 1, 2, 1]
        "graph_rnn_cell": "GRU",  # GRU, CudnnCompatibleGRUCell, or RNN
        "graph_rnn_activation": "tanh",  # tanh, ReLU
        "graph_state_dropout_keep_prob": 1.0,
        "edge_weight_dropout_keep_prob": 0.8,
        # set to ignore different edgetypes 'data', 'ctrl', 'path', 'none'
        "ignore_edge_types": False,  # only have one edgetype
        "ignore_node_features": True,
        "embeddings": "inst2vec"  # "finetune", "random"
    })
    return params

  def load_data(self, file_name, is_training_data: bool):
    full_path = os.path.join(self.data_dir, file_name)
    with prof.Profile(f"loaded data `{full_path}`"):
      with open(full_path, "r") as f:
        data = json.load(f)

    restrict = self.args.get("--restrict_data")
    if restrict is not None and restrict > 0:
      data = data[:restrict]

    # Get some common data out:
    ss = time.time()
    app.Log(1, "Getting some common data from loaded json...")
    num_fwd_edge_types = 0
    for g in data:
      num_fwd_edge_types = max(num_fwd_edge_types,
                               max([e[1] for e in g["graph"]]))

    app.Log(
        1,
        f"Getting some common data from loaded json... took {time.time() - ss} seconds."
    )

    return self.process_raw_graphs(data, is_training_data)

  def prepare_specific_graph_model(self) -> None:
    h_dim = FLAGS.hidden_size

    # self.placeholders["initial_node_representation"] = tf.placeholder(
    #     tf.float32, [None, h_dim], name="node_features"
    # )
    self.placeholders["num_of_nodes_in_batch"] = tf.placeholder(
        tf.int32, [], name="num_of_nodes_in_batch")
    self.placeholders["adjacency_lists"] = [
        tf.placeholder(tf.int32, [None, 3], name="adjacency_e%s" % e)
        for e in range(self.GetNumberOfEdgeTypes())
    ]  # changed adj. list to [None, 3] to add emb_idx at the end!
    self.placeholders["num_incoming_edges_per_type"] = tf.placeholder(
        tf.float32, [None, self.GetNumberOfEdgeTypes()],
        name="num_incoming_edges_per_type")
    self.placeholders["graph_nodes_list"] = tf.placeholder(
        tf.int32, [None], name="graph_nodes_list")
    self.placeholders["graph_state_keep_prob"] = tf.placeholder(
        tf.float32, None, name="graph_state_keep_prob")
    self.placeholders["edge_weight_dropout_keep_prob"] = tf.placeholder(
        tf.float32, None, name="edge_weight_dropout_keep_prob")

    activation_name = self.params["graph_rnn_activation"].lower()
    if activation_name == "tanh":
      activation_fun = tf.nn.tanh
    elif activation_name == "relu":
      activation_fun = tf.nn.relu
    else:
      raise Exception(
          "Unknown activation function type '%s'." % activation_name)

    # Generate per-layer values for edge weights, biases and gated units:
    self.weights = {}  # Used by super-class to place generic things
    self.gnn_weights = GGNNWeights([], [], [], [], [], [])
    for layer_idx in range(len(self.params["layer_timesteps"])):
      with tf.variable_scope("gnn_layer_%i" % layer_idx):
        edge_weights = tf.Variable(
            glorot_init([self.GetNumberOfEdgeTypes() * h_dim, h_dim]),
            name="gnn_edge_weights_%i" % layer_idx,
        )
        edge_weights = tf.reshape(edge_weights,
                                  [self.GetNumberOfEdgeTypes(), h_dim, h_dim])
        edge_weights = tf.nn.dropout(
            edge_weights,
            keep_prob=self.placeholders["edge_weight_dropout_keep_prob"],
        )
        self.gnn_weights.edge_weights.append(edge_weights)

        # analogous to how edge_weights (for mult. with neighbor states) looked like.
        # this is where we designed the update func. to be: U_m = A*s_n + B*e_(n,m) for all neighbors n of m.
        edge_weights_for_emb = tf.Variable(
            glorot_init([self.GetNumberOfEdgeTypes() * h_dim, h_dim]),
            name="gnn_edge_weights_for_emb_%i" % layer_idx,
        )
        edge_weights_for_emb = tf.reshape(edge_weights_for_emb,
                                          [self.GetNumberOfEdgeTypes(), h_dim, h_dim])
        edge_weights_for_emb = tf.nn.dropout(
            edge_weights_for_emb,
            keep_prob=self.placeholders["edge_weight_dropout_keep_prob"],
        )
        self.gnn_weights.edge_weights_for_embs.append(edge_weights_for_emb)

        if self.params["use_propagation_attention"]:
          self.gnn_weights.edge_type_attention_weights.append(
              tf.Variable(
                  np.ones([self.GetNumberOfEdgeTypes()], dtype=np.float32),
                  name="edge_type_attention_weights_%i" % layer_idx,
              ))

        if self.params["use_edge_bias"]:
          self.gnn_weights.edge_biases.append(
              tf.Variable(
                  np.zeros([self.GetNumberOfEdgeTypes(), h_dim], dtype=np.float32),
                  name="gnn_edge_biases_%i" % layer_idx,
              ))
          self.gnn_weights.edge_biases_for_embs.append(
              tf.Variable(
                  np.zeros([self.GetNumberOfEdgeTypes(), h_dim], dtype=np.float32),
                  name="gnn_edge_biases_%i" % layer_idx,
              ))

        cell_type = self.params["graph_rnn_cell"].lower()
        if cell_type == "gru":
          cell = tf.nn.rnn_cell.GRUCell(h_dim, activation=activation_fun)
        elif cell_type == "cudnncompatiblegrucell":
          assert activation_name == "tanh"
          import tensorflow.contrib.cudnn_rnn as cudnn_rnn

          cell = cudnn_rnn.CudnnCompatibleGRUCell(h_dim)
        elif cell_type == "rnn":
          cell = tf.nn.rnn_cell.BasicRNNCell(h_dim, activation=activation_fun)
        else:
          raise Exception("Unknown RNN cell type '%s'." % cell_type)
        cell = tf.nn.rnn_cell.DropoutWrapper(
            cell, state_keep_prob=self.placeholders["graph_state_keep_prob"])
        self.gnn_weights.rnn_cells.append(cell)

  def compute_final_node_representations(self) -> tf.Tensor:
    # one entry per layer (final state of that layer), shape: number of nodes in batch v x D
    node_states_per_layer = []

    # number of nodes in batch
    num_nodes = self.placeholders['num_of_nodes_in_batch']

    # we drop the initial states placeholder in the first layer as they are all zero.
    node_states_per_layer.append(
        tf.zeros([num_nodes, FLAGS.hidden_size], dtype=tf.float32,))

    message_targets = []  # list of tensors of message targets of shape [E]
    message_edge_types = []  # list of tensors of edge type of shape [E]
    message_emb_idxs = (
        []
    )  # list of tensors of idxs of the edge_emb lookup table for that message; of shape [E]

    # each edge type gets a unique edge_type_idx from its own adjacency list.
    # I will have to only carry one adj. list (one edge type, maybe could go to 2 for data and flow) and instead
    # figure out how to carry the emb as additional information, cf. "prep. spec. graphmodel: placeholder def.".
    for edge_type_idx, adjacency_list_for_edge_type in enumerate(
        self.placeholders["adjacency_lists"]):
      edge_targets = adjacency_list_for_edge_type[:, 1]
      message_targets.append(edge_targets)
      message_edge_types.append(
          tf.ones_like(edge_targets, dtype=tf.int32) * edge_type_idx)
      edge_emb_idxs = adjacency_list_for_edge_type[:, 2]
      message_emb_idxs.append(edge_emb_idxs)

    message_targets = tf.concat(message_targets, axis=0)  # Shape [M]
    message_edge_types = tf.concat(message_edge_types, axis=0)  # Shape [M]
    message_emb_idxs = tf.concat(message_emb_idxs, axis=0)  # Shape [M]

    for (layer_idx, num_timesteps) in enumerate(self.params["layer_timesteps"]):
      with tf.variable_scope("gnn_layer_%i" % layer_idx):
        # Used shape abbreviations:
        #   V ~ number of nodes
        #   D ~ state dimension
        #   E ~ number of edges of current type
        #   M ~ number of messages (sum of all E)

        # Extract residual messages, if any:
        layer_residual_connections = self.params["residual_connections"].get(
            str(layer_idx))
        if layer_residual_connections is None:
          layer_residual_states = []
        else:
          layer_residual_states = [
              node_states_per_layer[residual_layer_idx]
              for residual_layer_idx in layer_residual_connections
          ]

        if self.params["use_propagation_attention"]:
          message_edge_type_factors = tf.nn.embedding_lookup(
              params=self.gnn_weights.edge_type_attention_weights[layer_idx],
              ids=message_edge_types,
          )  # Shape [M]

        # Record new states for this layer. Initialised to last state, but will be updated below:
        node_states_per_layer.append(node_states_per_layer[-1])
        for step in range(num_timesteps):
          with tf.variable_scope("timestep_%i" % step):
            # list of tensors of messages of shape [E, D]
            messages = []
            # list of tensors of edge source states of shape [E, D]
            message_source_states = []

            # Collect incoming messages per edge type
            for edge_type_idx, adjacency_list_for_edge_type in enumerate(
                self.placeholders["adjacency_lists"]):
              edge_sources = adjacency_list_for_edge_type[:, 0]
              edge_emb_idxs = adjacency_list_for_edge_type[:, 2]
              edge_embs = tf.nn.embedding_lookup(
                  params=self.constants["embedding_table"],
                  ids=edge_emb_idxs,
              )
              edge_source_states = tf.nn.embedding_lookup(
                  params=node_states_per_layer[-1],
                  ids=edge_sources)  # Shape [E, D]

              # This seems to be the critical line, where the message propagation function is implemented.
              # Here the new "single edgetype but edge-emb as extra information"-model needs to hook.
              # Update: We are not single edge type, but can discern 'path', 'data', 'ctrl' and 'none' type.
              all_messages_for_edge_type = tf.matmul(
                  edge_source_states,
                  self.gnn_weights.edge_weights[layer_idx][edge_type_idx],
              ) + tf.matmul(
                  edge_embs,
                  self.gnn_weights.edge_weights[layer_idx][edge_type_idx],
              )  # Shape [E, D]
              messages.append(all_messages_for_edge_type)
              message_source_states.append(edge_source_states)

            messages = tf.concat(messages, axis=0)  # Shape [M, D]

            # TODO: not well understood
            if self.params["use_propagation_attention"]:
              message_source_states = tf.concat(message_source_states,
                                                axis=0)  # Shape [M, D]
              message_target_states = tf.nn.embedding_lookup(
                  params=node_states_per_layer[-1],
                  ids=message_targets)  # Shape [M, D]
              message_attention_scores = tf.einsum(
                  "mi,mi->m", message_source_states,
                  message_target_states)  # Shape [M]
              message_attention_scores = (message_attention_scores *
                                          message_edge_type_factors)

              # The following is softmax-ing over the incoming messages per node.
              # As the number of incoming varies, we can't just use tf.softmax. Reimplement with logsumexp trick:
              # Step (1): Obtain shift constant as max of messages going into a node
              message_attention_score_max_per_target = tf.unsorted_segment_max(
                  data=message_attention_scores,
                  segment_ids=message_targets,
                  num_segments=num_nodes,
              )  # Shape [V]
              # Step (2): Distribute max out to the corresponding messages again, and shift scores:
              message_attention_score_max_per_message = tf.gather(
                  params=message_attention_score_max_per_target,
                  indices=message_targets,
              )  # Shape [M]
              message_attention_scores -= (
                  message_attention_score_max_per_message)
              # Step (3): Exp, sum up per target, compute exp(score) / exp(sum) as attention prob:
              message_attention_scores_exped = tf.exp(
                  message_attention_scores)  # Shape [M]
              message_attention_score_sum_per_target = tf.unsorted_segment_sum(
                  data=message_attention_scores_exped,
                  segment_ids=message_targets,
                  num_segments=num_nodes,
              )  # Shape [V]
              message_attention_normalisation_sum_per_message = tf.gather(
                  params=message_attention_score_sum_per_target,
                  indices=message_targets,
              )  # Shape [M]
              message_attention = message_attention_scores_exped / (
                  message_attention_normalisation_sum_per_message + SMALL_NUMBER
              )  # Shape [M]
              # Step (4): Weigh messages using the attention prob:
              messages = messages * tf.expand_dims(message_attention, -1)

            incoming_messages = tf.unsorted_segment_sum(
                data=messages,
                segment_ids=message_targets,
                num_segments=num_nodes,
            )  # Shape [V, D]

            if self.params["use_edge_bias"]:
              incoming_messages += tf.matmul(
                  self.placeholders["num_incoming_edges_per_type"],
                  self.gnn_weights.edge_biases[layer_idx],
              )  # Shape [V, D]

            if self.params["use_edge_msg_avg_aggregation"]:
              num_incoming_edges = tf.reduce_sum(
                  self.placeholders["num_incoming_edges_per_type"],
                  keep_dims=True,
                  axis=-1,
              )  # Shape [V, 1]
              incoming_messages /= num_incoming_edges + SMALL_NUMBER

            incoming_information = tf.concat(
                layer_residual_states + [incoming_messages],
                axis=-1)  # Shape [V, D*(1 + num of residual connections)]

            # pass updated vertex features into RNN cell
            node_states_per_layer[-1] = self.gnn_weights.rnn_cells[layer_idx](
                incoming_information,
                node_states_per_layer[-1])[1]  # Shape [V, D]

    return node_states_per_layer[-1]

  def gated_regression(self, last_h, regression_gate, regression_transform):
    # last_h: [v x h]

    if self.params['ignore_node_features']:
      num_nodes = self.placeholders['num_of_nodes_in_batch']
      initial_node_rep = tf.zeros([num_nodes, FLAGS.hidden_size])
    else:
      initial_node_rep = self.placeholders["initial_node_representation"]

    gate_input = tf.concat([last_h, initial_node_rep], axis=-1)  # [v x 2h]
    gated_outputs = (tf.nn.sigmoid(regression_gate(gate_input)) *
                     regression_transform(last_h))  # [v x 1]

    # Sum up all nodes per-graph
    self.output = tf.unsorted_segment_sum(
        data=gated_outputs,
        segment_ids=self.placeholders["graph_nodes_list"],
        num_segments=self.placeholders["num_graphs"],
    )  # [g, c]
    return self.output

  # ----- Data preprocessing and chunking into minibatches:
  def process_raw_graphs(self, raw_data: Sequence[Any],
                         is_training_data: bool) -> Any:
    processed_graphs = []
    for d in raw_data:
      (
          adjacency_lists,
          num_incoming_edge_per_type,
      ) = self.__graph_to_adjacency_lists(d["graph"])

      graph_dic = {
          "adjacency_lists": adjacency_lists,
          "num_incoming_edge_per_type": num_incoming_edge_per_type,
          "label": d["target"],
          "number_of_nodes": d["number_of_nodes"],
      }
      if not self.params["ignore_node_features"] and d["node_features"]:
        graph_dic["init"] = d["node_features"]
      elif not self.params["ignore_node_features"]:
        raise NotImplementedError(
            "The current dataset doesn't implement node_features, but ignore_node_features == False!"
        )

      processed_graphs.append(graph_dic)

    if is_training_data:
      np.random.shuffle(processed_graphs)

    return processed_graphs

  def __graph_to_adjacency_lists(
      self, graph) -> Tuple[Dict[int, np.ndarray], Dict[int, Dict[int, int]]]:
    adj_lists = defaultdict(list)
    num_incoming_edges_dicts_per_type = defaultdict(lambda: defaultdict(lambda:
                                                                        0))
    for src, fwd_edge_type, dest, emb_idx in graph:
      if self.params["ignore_edge_types"]:
        fwd_edge_type = 0  # Edge type optionally always 0

      adj_lists[fwd_edge_type].append(
          (src, dest, emb_idx))  # attach emb_idx in adj. list!
      num_incoming_edges_dicts_per_type[fwd_edge_type][dest] += 1

      if FLAGS.tie_fwd_bkwd:
        adj_lists[fwd_edge_type].append((dest, src, emb_idx))  # here too
        num_incoming_edges_dicts_per_type[fwd_edge_type][src] += 1

    final_adj_lists = {
        e: np.array(sorted(lm), dtype=np.int32) for e, lm in adj_lists.items()
    }  # turn in normal dict.

    # Add backward edges as an additional edge type that goes backwards:
    if not FLAGS.tie_fwd_bkwd:
      for (edge_type, edges) in adj_lists.items():
        bwd_edge_type = self.GetNumberOfEdgeTypes() + edge_type
        final_adj_lists[bwd_edge_type] = np.array(sorted(
            (y, x, emb_idx) for (x, y, emb_idx) in edges),
                                                  dtype=np.int32)  # like this?
        for (x, y, emb_idx) in edges:
          num_incoming_edges_dicts_per_type[bwd_edge_type][y] += 1

    return final_adj_lists, num_incoming_edges_dicts_per_type

  def load_datasets(self) -> None:
    if args.get("--evaluate"):
      train_data = None
      app.Log(1, "Skipped loading train data...")
    else:
      train_data = self.load_data(params["train_file"], is_training_data=True)

    valid_data = self.load_data(params["valid_file"], is_training_data=False)

    if params["test_file"]:
      test_data = self.load_data(params["test_file"], is_training_data=False)
    else:
      test_data = None
      app.Log(1, "Skipped loading test data...")

    self.data = {
        "train": train_data,
        "valid": valid_data,
        "test": test_data,
    }

  def make_target_values_placeholder(self) -> tf.Tensor:
    return tf.placeholder(tf.int32, [None], name="target_values")

  def make_loss_and_accuracy_ops(self) -> Tuple[tf.Tensor, tf.Tensor]:
    with tf.variable_scope("regression_gate"):
      self.weights["regression_gate"] = MLP(
          # Concatenation of initial and final node states
          in_size=2 * FLAGS.hidden_size,
          out_size=self.GetNumberOfClasses(),
          hid_sizes=[],
          dropout_keep_prob=self.placeholders["out_layer_dropout_keep_prob"],
      )
    with tf.variable_scope("regression"):
      self.weights["regression_transform"] = MLP(
          FLAGS.hidden_size,
          self.GetNumberOfClasses(),
          [],
          self.placeholders["out_layer_dropout_keep_prob"],
      )

    # this is all Eq. 7 in the GGNN paper here... (i.e. Eq. 4 in NMP for QC)
    computed_values = self.gated_regression(
        self.ops["final_node_representations"],
        self.weights[
            "regression_gate"],  # these "weights" actually pass the mlp function.
        self.weights["regression_transform"],
    )
    accuracy = tf.reduce_mean(
        tf.cast(
            tf.equal(
                self.placeholders["target_values"],
                tf.argmax(computed_values, axis=1, output_type=tf.int32),
            ),
            tf.float32,
        ))
    # cross-entropy loss
    loss = tf.losses.sparse_softmax_cross_entropy(
        self.placeholders["target_values"], computed_values)

    return loss, accuracy

  def make_minibatch_iterator(self, epoch_type: str):
    """Create minibatches by flattening adjacency matrices into a single adjacency matrix with
    multiple disconnected components."""
    data = self.data[epoch_type]
    is_training = epoch_type == "train"

    if is_training:
      np.random.shuffle(data)
    # Pack until we cannot fit more graphs in the batch
    state_dropout_keep_prob = (self.params["graph_state_dropout_keep_prob"]
                               if is_training else 1.0)
    edge_weights_dropout_keep_prob = (
        self.params["edge_weight_dropout_keep_prob"] if is_training else 1.0)
    num_graphs = 0
    while num_graphs < len(data):
      num_graphs_in_batch = 0
      # batch_node_features = []
      batch_target_values = []
      batch_adjacency_lists = [[] for _ in range(self.GetNumberOfEdgeTypes())]
      batch_num_incoming_edges_per_type = []
      batch_graph_nodes_list = []
      node_offset = 0

      while (num_graphs < len(data) and
             node_offset + data[num_graphs]["number_of_nodes"] <
             FLAGS.batch_size):
        cur_graph = data[num_graphs]
        num_nodes_in_graph = cur_graph["number_of_nodes"]
        # padded_features = np.pad(
        #     cur_graph["init"],
        #     ((0, 0), (0, FLAGS.hidden_size - self.GetNodeFeaturesDimensionality())),
        #     "constant",
        # )
        # batch_node_features.extend(padded_features)
        batch_graph_nodes_list.append(
            np.full(
                shape=[num_nodes_in_graph],
                fill_value=num_graphs_in_batch,
                dtype=np.int32,
            ))
        for i in range(self.GetNumberOfEdgeTypes()):
          if i in cur_graph["adjacency_lists"]:
            # change: don't offset the edge emb idx.
            batch_adjacency_lists[i].append(
                cur_graph["adjacency_lists"][i] +
                np.array((node_offset, node_offset, 0), dtype=np.int32))

        # Turn counters for incoming edges into np array:
        num_incoming_edges_per_type = np.zeros(
            (num_nodes_in_graph, self.GetNumberOfEdgeTypes()))
        for (e_type, num_incoming_edges_per_type_dict
            ) in cur_graph["num_incoming_edge_per_type"].items():
          for (
              node_id,
              edge_count,
          ) in num_incoming_edges_per_type_dict.items():
            num_incoming_edges_per_type[node_id, e_type] = edge_count
        batch_num_incoming_edges_per_type.append(num_incoming_edges_per_type)

        target_value = (
            cur_graph["label"] - 1)  # because classes start counting at 1.
        # sanity checks
        assert (type(target_value) == int
               ), f"target_value type should be int but is {type(target_value)}"
        assert (target_value <= 103 and
                target_value >= 0), f"target_value range wrong: {target_value}"

        batch_target_values.append(target_value)
        num_graphs += 1
        num_graphs_in_batch += 1
        node_offset += num_nodes_in_graph

      batch_feed_dict = {
          # self.placeholders["initial_node_representation"]: np.array(
          #     batch_node_features
          # ),  # list[np.array], len ~ 100.000 (batch_size)
          self.placeholders["num_incoming_edges_per_type"]:
          np.concatenate(
              batch_num_incoming_edges_per_type,
              axis=0),  # list[np.array], len ~ 681 (num of graphs in batch)
          self.placeholders["graph_nodes_list"]:
          np.concatenate(batch_graph_nodes_list
                        ),  # list[np.array], len ~ 681 (num of graphs in batch)
          self.placeholders["target_values"]:
          np.array(batch_target_values),  # list(int) , len 681,
          self.placeholders["num_graphs"]:
          num_graphs_in_batch,
          self.placeholders["num_of_nodes_in_batch"]:
          node_offset,
          self.placeholders["graph_state_keep_prob"]:
          state_dropout_keep_prob,
          self.placeholders["edge_weight_dropout_keep_prob"]:
          edge_weights_dropout_keep_prob,
      }

      # Merge adjacency lists and information about incoming nodes:
      for i in range(self.GetNumberOfEdgeTypes()):
        if len(batch_adjacency_lists[i]) > 0:
          adj_list = np.concatenate(batch_adjacency_lists[i])
        else:
          adj_list = np.zeros((0, 3),
                              dtype=np.int32)  # shape (0, 3) should be right.
        batch_feed_dict[self.placeholders["adjacency_lists"][i]] = adj_list

      yield batch_feed_dict


def main():
  args = docopt(__doc__)
  # try:
  model = GGNNClassifyappModel(args)
  if args["--evaluate"]:
    assert args["--restore"], "You meant to provide a restore file."
    model.Test()
  else:
    model.Train()
  # except:
  #     typ, value, tb = sys.exc_info()
  #     traceback.print_exc()
  #     pdb.post_mortem(tb)


if __name__ == "__main__":
  main()
