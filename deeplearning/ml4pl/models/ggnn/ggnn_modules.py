"""Modules that make up the pytorch GGNN model."""
import torch
import torch.nn.functional as F
from torch import nn
from torch import optim

from labm8.py import app

FLAGS = app.FLAGS
SMALL_NUMBER = 1e-7


# optimizer Adam
# FLAGS.learning_rate * self.placeholders["learning_rate_multiple"]
#
# clip gradients by norm
# (tf.clip_by_norm(grad, FLAGS.clamp_gradient_norm), var)

###########################
# Main Model
###########################


class GGNNModel(nn.Module):
  def __init__(self, config, pretrained_embeddings=None, test_only=False):
    super().__init__()
    self.config = config
    self.node_embeddings = NodeEmbeddings(config, pretrained_embeddings)
    self.ggnn = GGNNProper(config)
    self.nodewise_readout = NodewiseReadout(config)

    self.graphlevel_readout = None
    if config.has_graph_labels:
      self.graphlevel_readout = AuxiliaryReadout(config)

    # eval and training
    self.loss = Loss(config)
    self.metrics = Metrics()

    # not instantiating the optimizer should save ~67% of GPU memory, bc Adam carries two momentum params
    # per trainable model parameter!
    if test_only:
      self.opt = None
      self.eval()
    else:
      self.opt = self.get_optimizer(config)

  def get_optimizer(self, config):
    return optim.AdamW(self.parameters(), lr=config.lr)  # NB: AdamW

  def forward(
    self,
    vocab_ids,
    selector_ids,
    labels,
    edge_lists,
    num_graphs=None,
    graph_nodes_list=None,
    aux_in=None,
  ):
    raw_in = self.node_embeddings(vocab_ids, selector_ids)
    raw_out, raw_in = self.ggnn(
      edge_lists, raw_in
    )  # OBS! self.ggnn might change raw_in inplace, so use the two outputs instead!
    prediction = self.nodewise_readout(raw_in, raw_out)

    if self.graphlevel_readout:
      prediction, graph_features = self.graphlevel_readout(
        prediction, num_graphs, graph_nodes_list, aux_in
      )
    else:
      graph_features = None

    # accuracy, pred_targets, correct, targets
    metrics_tuple = self.metrics(prediction, labels)

    outputs = (prediction,) + metrics_tuple + (graph_features,)

    return outputs


#############################
# Loss Accuracy Prediction
#############################


class Loss(nn.Module):
  """[Binary] Cross Entropy loss with weighted intermediate loss"""

  def __init__(self, config):
    super().__init__()
    self.config = config
    if config.num_classes == 1:
      self.loss = nn.BCELoss()  # in: (N, *), target: (N, *)
    else:
      # TODO(github.com/ChrisCummins/ProGraML/issues/27): class labels '-1' don't contribute to the gradient! I was under the impression that we wanted to exploit this fact somewhere. I.e. not predicting labels on nodes that don't constitute branching statements. Let's discuss!
      self.loss = nn.CrossEntropyLoss(ignore_index=-1)

  def forward(self, inputs, targets):
    """inputs: (predictions) or (predictions, intermediate_predictions)"""
    loss = self.loss(inputs[0], targets)
    if self.config.has_graph_labels:
      loss += self.config.intermediate_loss_weight * self.loss(
        inputs[1], targets
      )
    return loss


class Metrics(nn.Module):
  """Common metrics and info for inspection of results.
  Args:
  logits, labels
  Returns:
  (accuracy, pred_targets, correct_preds, targets)"""

  def __init__(self):
    super().__init__()

  def forward(self, logits, labels):
    targets = labels.argmax(dim=1)
    pred_targets = logits.argmax(dim=1)
    correct_preds = targets.eq(pred_targets).float()
    accuracy = torch.mean(correct_preds)
    return accuracy, logits, correct_preds, targets


###########################
# GGNNInput: Embeddings
###########################


class NodeEmbeddings(nn.Module):
  """Construct node embeddings (content embeddings + selector embeddings)
  Args:
  pretrained_embeddings (Tensor, optional) – FloatTensor containing weights for the Embedding. First dimension is being passed to Embedding as num_embeddings, second as embedding_dim.

  Forward
  Args:
  vocab_ids: <N, 1>
  selector_ids: <N, 1>
  Returns:
  node_states: <N, config.hidden_size>
  """

  # TODO(github.com/ChrisCummins/ProGraML/issues/27):: Maybe LayerNorm and Dropout on node_embeddings?
  # TODO(github.com/ChrisCummins/ProGraML/issues/27):: Make selector embs trainable?

  # TODO(github.com/ChrisCummins/ml4pl/issues/12): In the future we may want
  # to be more flexible in supporting multiple types of embeddings tables, but
  # for now I have hardcoded this to always return a tuple
  # <inst2vec_embeddings, selector_embeddings>, where inst2vec_embeddings
  # is the augmented table of pre-trained statement embeddings (the
  # augmentation adds !MAGIC, !IMMEDIATE, and !IDENTIFIER vocabulary
  # elements). selector_embeddings is a 2x2 1-hot embedding table:
  # [[1, 0], [0, 1]. The selector_embeddings table is always constant, the
  # inst2vec_embeddings table can be made trainable or re-initialized with
  # random values using the --inst2vec_embeddings flag.

  def __init__(self, config, pretrained_embeddings=None):
    super().__init__()

    if config.inst2vec_embeddings == "constant":
      app.Log(
        1, "Using pre-trained inst2vec embeddings without further training"
      )
      assert pretrained_embeddings is not None
      self.node_embs = nn.Embedding.from_pretrained(
        pretrained_embeddings, freeze=True
      )

    elif config.inst2vec_embeddings == "constant_zero":
      init = torch.zeros(config.vocab_size, config.emb_size)
      self.node_embs = nn.Embedding.from_pretrained(init, freeze=True)

    elif config.inst2vec_embeddings == "constant_random":
      init = torch.rand(config.vocab_size, config.emb_size)
      self.node_embs = nn.Embedding.from_pretrained(init, freeze=True)

    elif config.inst2vec_embeddings == "finetune":
      app.Log(1, "Fine-tuning inst2vec embeddings")
      assert pretrained_embeddings is not None
      self.node_embs = nn.Embedding.from_pretrained(
        pretrained_embeddings, freeze=False
      )

    elif config.inst2vec_embeddings == "random":
      app.Log(1, "Initializing with random embeddings")
      self.node_embs = nn.Embedding(config.vocab_size, config.emb_size)

    else:
      raise app.UsageError(
        f"--inst2vec_embeddings=`{FLAGS.inst2vec_embeddings}` "
        "unrecognized. Must be one of "
        "{constant,constant_zero,finetune,random}"
      )

    if config.use_selector_embeddings:
      selector_init = torch.tensor(
        # TODO(github.com/ChrisCummins/ProGraML/issues/27): x50 is maybe a problem for unrolling (for selector_embs)?
        [[0, 50.0], [50.0, 0]],
        dtype=torch.get_default_dtype(),
      )
      self.selector_embs = nn.Embedding.from_pretrained(
        selector_init, freeze=True
      )
    else:
      self.selector_embs = None

  def forward(self, vocab_ids, selector_ids):
    embs = self.node_embs(vocab_ids)
    if self.selector_embs:
      selector_embs = self.selector_embs(selector_ids)
      embs = torch.cat((embs, selector_embs), dim=1)
    return embs


################################################
# GGNNProper: Weights, TransformAndUpdate
################################################


class GGNNProper(nn.Module):
  def __init__(self, config):
    super().__init__()
    self.backward_edges = config.backward_edges
    self.layer_timesteps = config.layer_timesteps

    self.message = []
    for i in range(len(self.layer_timesteps)):
      self.message.append(MessagingLayer(config))

    self.update = []
    for i in range(len(self.layer_timesteps)):
      self.update.append(GGNNLayer(config))

  def forward(self, edge_lists, node_states, position_embeddings=None):

    # TODO(github.com/ChrisCummins/ProGraML/issues/27): This modifies the
    # arguments in-place.

    old_node_states = torch.tensor(node_states, requires_grad=True)
    # TODO(github.com/ChrisCummins/ProGraML/issues/30): position embeddings
    assert position_embeddings is None, "Position Embs not implemented"

    if self.backward_edges:
      back_edge_lists = [x.flip([1]) for x in edge_lists]
      edge_lists.extend(back_edge_lists)

    for (layer_idx, num_timesteps) in enumerate(self.layer_timesteps):
      for t in range(num_timesteps):
        messages = self.message[layer_idx](edge_lists, node_states)
        node_states = self.update[layer_idx](messages, node_states)
    return node_states, old_node_states


class MessagingLayer(nn.Module):
  """takes an edge_list (for a single edge type) and node_states <N, D+S> and
  returns incoming messages per node of shape <N, D+S>"""

  def __init__(self, config):
    super().__init__()
    self.forward_and_backward_edge_type_count = (
      config.edge_type_count * 2
      if config.backward_edges
      else config.edge_type_count
    )
    self.msg_mean_aggregation = config.msg_mean_aggregation
    self.dim = config.hidden_size

    # TODO(github.com/ChrisCummins/ProGraML/issues/27): why do edges carry no
    # bias? Seems restrictive. Now they can, maybe default corr. FLAG to true?
    self.transform = LinearNet(
      self.dim,
      self.dim * self.forward_and_backward_edge_type_count,
      bias=config.use_edge_bias,
      dropout=config.edge_weight_dropout,
    )

  def forward(self, edge_lists, node_states):
    """edge_lists: [<M_i, 2>, ...]"""
    # all edge types are handled in one matrix, but we
    # let propagated_states[i] be equal to the case with only edge_type i
    propagated_states = (
      self.transform(node_states)
      .transpose(0, 1)
      .view(self.forward_and_backward_edge_type_count, self.dim, -1)
    )

    messages_by_targets = torch.zeros_like(node_states)
    if self.msg_mean_aggregation:
      bincount = torch.zeros(node_states.size()[0], dtype=torch.long)

    for i, edge_list in enumerate(edge_lists):
      edge_targets = edge_list[:, 1]
      edge_sources = edge_list[:, 0]
      # TODO(github.com/ChrisCummins/ProGraML/issues/27): transform all
      # node_states? maybe wasteful, maybe MUCH better than propagating them
      # after the embedding lookup (except if graph is super sparse
      # (per edge_type)).
      # TODO(github.com/ChrisCummins/ProGraML/issues/30): with edge positions,
      # we can do better by distribution rule: A (h + p) = Ah + Ap, so the
      # position table can be multiplied before addition as well.
      # TODO(github.com/ChrisCummins/ProGraML/issues/30): with "fancy" mode,
      # anyway it's just another edge type
      states_by_source = F.embedding(
        edge_sources, propagated_states[i].transpose(0, 1)
      )
      messages_by_targets.index_add_(0, edge_targets, states_by_source)
      if self.msg_mean_aggregation:
        bincount += edge_targets.bincount(minlength=node_states.size()[0])

    if self.msg_mean_aggregation:
      divisor = bincount.float()
      divisor[bincount == 0] = 1.0  # avoid div by zero for lonely nodes
      messages_by_targets /= divisor.unsqueeze_(1)
    return messages_by_targets


class GGNNLayer(nn.Module):
  def __init__(self, config):
    super().__init__()
    self.dropout = config.graph_state_dropout
    # TODO(github.com/ChrisCummins/ProGraML/issues/27): Maybe decouple hidden GRU size: make hidden GRU size larger and EdgeTrafo size non-square instead? Or implement stacking gru layers between message  passing steps.
    self.gru = nn.GRUCell(
      input_size=config.hidden_size, hidden_size=config.hidden_size
    )

  def forward(self, messages, node_states):
    if self.dropout > 0.0:
      F.dropout_(messages, p=self.dropout, training=self.training, inplace=True)

    output = self.gru(messages, node_states)

    if self.dropout > 0.0:
      F.dropout_(output, p=self.dropout, training=self.training, inplace=True)
    return output


# position propagation matrices are treated like another edge type
#                if FLAGS.position_embeddings == "fancy":
#                    type_count_with_fancy = 1 + self.stats.edge_type_count
#                else:
#                    type_count_with_fancy = self.stats.edge_type_count

#    def _GetPositionEmbeddingsAsTensorflowVariable(self) -> tf.Tensor:
#        """It's probably a good memory/compute trade-off to have this additional embedding table instead of computing it on the fly."""
#        embeddings = base_utils.pos_emb(
#            positions=range(self.stats.max_edge_positions), demb=FLAGS.hidden_size - 2
#        )  # hard coded
#        pos_emb = tf.Variable(
#            initial_value=embeddings, trainable=False, dtype=tf.float32
#        )
#        return pos_emb
class PositionEmbeddings(nn.Module):
  def __init__(self):
    super().__init__()

  def forward(self, *args, **kwargs):
    return args, kwargs


########################################
# Output Layer
########################################


class NodewiseReadout(nn.Module):
  """aka GatedRegression"""

  def __init__(self, config):
    super().__init__()
    self.regression_gate = LinearNet(
      2 * config.hidden_size, config.num_classes, dropout=config.output_dropout,
    )
    self.regression_transform = LinearNet(
      config.hidden_size, config.num_classes, dropout=config.output_dropout,
    )

  def forward(self, raw_node_in, raw_node_out):
    gate_input = torch.cat((raw_node_in, raw_node_out), dim=-1)
    gating = F.sigmoid(self.regression_gate(gate_input))
    return gating * self.regression_transform(raw_node_out)


class LinearNet(nn.Module):
  """Single Linear layer with WeightDropout, ReLU and Xavier Uniform initialization.
  Applies a linear transformation to the incoming data: :math:`y = xA^T + b`

  Args:
  in_features: size of each input sample
  out_features: size of each output sample
  bias: If set to ``False``, the layer will not learn an additive bias.
  Default: ``True``

  Shape:
  - Input: :math:`(N, *, H_{in})` where :math:`*` means any number of
  additional dimensions and :math:`H_{in} = \text{in\_features}`
  - Output: :math:`(N, *, H_{out})` where all but the last dimension
  are the same shape as the input and :math:`H_{out} = \text{out\_features}`.
  """

  def __init__(self, in_features, out_features, bias=True, dropout=0.0):
    super().__init__()
    self.dropout = dropout
    self.in_features = in_features
    self.out_features = out_features
    self.weight = nn.parameter.Parameter(
      torch.Tensor(out_features, in_features)
    )
    if bias:
      self.bias = nn.parameter.Parameter(torch.Tensor(out_features))
    else:
      self.register_parameter("bias", None)
    self.reset_parameters()

  def reset_parameters(self):
    nn.init.xavier_uniform_(self.weight)
    # TODO(github.com/ChrisCummins/ProGraML/issues/27): why use xavier_uniform, not kaiming init? Seems old-school
    if self.bias is not None:
      #    fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
      #    bound = 1 / math.sqrt(fan_in)
      #    nn.init.uniform_(self.bias, -bound, bound)
      nn.init.zeros_(self.bias)

  def forward(self, input):
    if self.dropout > 0.0:
      w = F.dropout(self.weight, p=self.dropout, training=self.training)
    else:
      w = self.weight
    return F.linear(input, w, self.bias)

  def extra_repr(self):
    return "in_features={}, out_features={}, bias={}, dropout={}".format(
      self.in_features, self.out_features, self.bias is not None, self.dropout
    )


#######################################
# Adding Graph Level Features to Model


class AuxiliaryReadout(nn.Module):
  """Produces per-graph predictions using the per-node predictions and auxiliary features"""

  # TODO(github.com/ChrisCummins/ProGraML/issues/27): I don't like that we only introduce the global features AFTER the per node predictions have been made and not while we do those! This is limiting the expressivity of the model.
  def __init__(self, config):
    super().__init__()
    self.config = config

    self.feed_forward = None
    if config.has_graph_labels:
      self.feed_forward = nn.Sequential(
        nn.BatchNorm1d(config.num_classes + config.aux_in_len),
        nn.Linear(
          config.num_classes + config.aux_in_len, config.aux_in_layer_size,
        ),
        nn.ReLU(),
        nn.Dropout(1 - config.output_dropout),
        nn.Linear(config.aux_in_layer_size, config.num_classes),
      )

  def forward(
    self, raw_node_out, num_graphs, graph_nodes_list, auxiliary_features
  ):
    graph_features = torch.zeros(num_graphs, self.config.num_classes)
    graph_features.index_add_(
      dim=0, index=graph_nodes_list, source=raw_node_out
    )

    aggregate_features = torch.cat((graph_features, auxiliary_features), dim=1)
    return self.feed_forward(aggregate_features), graph_features
