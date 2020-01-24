# Copyright 2019 the ProGraML authors.
#
# Contact Chris Cummins <chrisc.101@gmail.com>.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Modules that make up the pytorch GGNN model."""
import torch
import torch.nn.functional as F
from torch import nn
from torch import optim

from labm8.py import app
from labm8.py import gpu_scheduler

FLAGS = app.FLAGS
SMALL_NUMBER = 1e-8

# optimizer Adam
# FLAGS.learning_rate * self.placeholders["learning_rate_multiple"]

###########################
# Main Model
###########################


def assert_no_nan(tensor_list):
  for i, t in enumerate(tensor_list):
    assert not torch.isnan(t).any(), f"{i}: {tensor_list}"


class GGNNModel(nn.Module):
  def __init__(self, config, pretrained_embeddings=None, test_only=False):
    super().__init__()
    self.config = config

    # input layer
    self.node_embeddings = NodeEmbeddings(config, pretrained_embeddings)

    self.ggnn = GGNNProper(config)

    # maybe tack on the aux readout
    self.has_aux_input = getattr(self.config, "has_aux_input", False)

    if self.has_aux_input:
      self.aux_readout = AuxiliaryReadout(config)

    # make readout available to label_convergence tests in GGNN Proper (at runtime)
    if self.config.unroll_strategy == "label_convergence":
      assert (
        not self.config.has_aux_input
      ), "aux_input is not supported with label_convergence"

    # eval and training
    self.loss = Loss(config)
    self.metrics = Metrics()

    # not instantiating the optimizer should save 2 x #model_params of GPU memory, bc. Adam
    # carries two momentum params per trainable model parameter.

    # move model to device before making optimizer!
    self.dev = (
      torch.device("cuda")
      if gpu_scheduler.LockExclusiveProcessGpuAccess()
      else torch.device("cpu")
    )

    self.to(self.dev)
    print(f"Moved model to {self.dev}")

    if test_only:
      self.opt = None
      self.eval()
    else:
      self.opt = self.get_optimizer(config)

  def get_optimizer(self, config):
    return optim.AdamW(self.parameters(), lr=config.lr)

  def forward(
    self,
    vocab_ids,
    labels,
    edge_lists,
    selector_ids=None,
    pos_lists=None,
    num_graphs=None,
    graph_nodes_list=None,
    node_types=None,
    aux_in=None,
    test_time_steps=None,
  ):
    raw_in = self.node_embeddings(vocab_ids, selector_ids)
    raw_out, raw_in, *unroll_stats = self.ggnn(
      edge_lists, raw_in, pos_lists, node_types, test_time_steps
    )  # OBS! self.ggnn might change raw_in inplace, so use the two outputs
    # instead!

    if self.config.has_graph_labels:
      assert (
        graph_nodes_list is not None and num_graphs is not None
      ), "has_graph_labels requires graph_nodes_list and num_graphs tensors."

    nodewise_readout, graphwise_readout = self.ggnn.readout(
      raw_in, raw_out, graph_nodes_list=graph_nodes_list, num_graphs=num_graphs
    )

    logits = (
      graphwise_readout if self.config.has_graph_labels else nodewise_readout
    )

    if self.has_aux_input:
      logits, graphwise_readout = self.aux_readout(logits, aux_in)

    # accuracy, pred_targets, correct, targets
    metrics_tuple = self.metrics(logits, labels)

    outputs = (
      (logits,) + metrics_tuple + (graphwise_readout,) + tuple(unroll_stats)
    )

    return outputs

  def num_parameters(self) -> int:
    """Compute the number of trainable parameters in this nn.Module and its children."""
    return sum(
      param.numel()
      for param in self.parameters(recurse=True)
      if param.requires_grad
    )


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
      # TODO(github.com/ChrisCummins/ProGraML/issues/27): class labels '-1'
      # don't contribute to the gradient! I was under the impression that we
      # wanted to exploit this fact somewhere. I.e. not predicting labels on
      # nodes that don't constitute branching statements. Let's discuss!
      self.loss = nn.CrossEntropyLoss(ignore_index=-1)

  def forward(self, inputs, targets):
    """inputs: (predictions) or (predictions, intermediate_predictions)"""
    loss = self.loss(inputs[0], targets)
    if getattr(self.config, "has_aux_input", False):
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
    # be flexible with 1hot labels vs indices
    if len(labels.size()) == 2:
      targets = labels.argmax(dim=1)
    elif len(labels.size()) == 1:
      targets = labels
    else:
      raise ValueError(
        f"labels={labels.size()} tensor is is neither 1 nor 2-dimensional. :/"
      )

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
  pretrained_embeddings (Tensor, optional) – FloatTensor containing weights for
  the Embedding. First dimension is being passed to Embedding as
  num_embeddings, second as embedding_dim.

  Forward
  Args:
  vocab_ids: <N, 1>
  selector_ids: <N, 1>
  Returns:
  node_states: <N, config.hidden_size>
  """

  # TODO(github.com/ChrisCummins/ProGraML/issues/27):: Maybe LayerNorm and
  # Dropout on node_embeddings?
  # TODO(github.com/ChrisCummins/ProGraML/issues/27):: Make selector embs
  # trainable?

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
    self.inst2vec_embeddings = config.inst2vec_embeddings
    self.emb_size = config.emb_size

    if config.inst2vec_embeddings == "constant":
      app.Log(1, "Using pre-trained inst2vec embeddings frozen.")
      assert pretrained_embeddings is not None
      self.node_embs = nn.Embedding.from_pretrained(
        pretrained_embeddings, freeze=True
      )
    elif config.inst2vec_embeddings == "zero":
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
    elif config.inst2vec_embeddings == "none":
      app.Log(
        1, "Initializing with a embedding for statements and identifiers each."
      )
      self.node_embs = nn.Embedding(2, config.emb_size)
    else:
      raise NotImplementedError(config.inst2vec_embeddings)

    if (
      hasattr(config, "use_selector_embeddings")
      and config.use_selector_embeddings
    ):
      selector_init = torch.tensor(
        # TODO(github.com/ChrisCummins/ProGraML/issues/27): x50 is maybe a
        # problem for unrolling (for selector_embs)?
        [[0, 50.0], [50.0, 0]],
        dtype=torch.get_default_dtype(),
      )
      self.selector_embs = nn.Embedding.from_pretrained(
        selector_init, freeze=True
      )
    else:
      self.selector_embs = None

  def forward(self, vocab_ids, selector_ids=None):
    if self.inst2vec_embeddings == "none":
      # map IDs to 1 and everything else to 0
      ids = (vocab_ids == 8565).to(torch.long)  # !IDENTIFIER token id
      embs = self.node_embs(ids)
    else:  # normal embeddings
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
    self.readout = Readout(config)
    self.backward_edges = config.backward_edges
    self.layer_timesteps = config.layer_timesteps
    self.position_embeddings = config.position_embeddings

    # optional eval time unrolling parameter
    self.test_layer_timesteps = (
      config.test_layer_timesteps
      if hasattr(config, "test_layer_timesteps")
      else 0
    )
    self.unroll_strategy = (
      config.unroll_strategy if hasattr(config, "unroll_strategy") else "none"
    )
    self.max_timesteps = (
      config.max_timesteps if hasattr(config, "max_timesteps") else 1000
    )
    self.label_conv_threshold = (
      config.label_conv_threshold
      if hasattr(config, "label_conv_threshold")
      else 0.995
    )
    self.label_conv_stable_steps = (
      config.label_conv_stable_steps
      if hasattr(config, "label_conv_stable_steps")
      else 1
    )

    # Message and update layers
    self.message = nn.ModuleList()
    for i in range(len(self.layer_timesteps)):
      self.message.append(MessagingLayer(config))

    self.update = nn.ModuleList()
    for i in range(len(self.layer_timesteps)):
      self.update.append(GGNNLayer(config))

  def forward(
    self,
    edge_lists,
    node_states,
    pos_lists=None,
    node_types=None,
    test_time_steps=None,
  ):
    old_node_states = node_states.clone()

    if self.backward_edges:
      back_edge_lists = [x.flip([1]) for x in edge_lists]
      edge_lists.extend(back_edge_lists)

      # For backward edges we keep the positions of the forward edge!
      if self.position_embeddings:
        pos_lists.extend(pos_lists)

    # we allow for some fancy unrolling strategies.
    # Currently only at eval time, but there is really no good reason for this.
    if self.training or self.unroll_strategy == "none":
      layer_timesteps = self.layer_timesteps
    elif self.unroll_strategy == "constant":
      layer_timesteps = self.test_layer_timesteps
    elif self.unroll_strategy == "edge_count":
      assert (
        test_time_steps is not None
      ), f"You need to pass test_time_steps or not use unroll_strategy '{self.unroll_strategy}''"
      layer_timesteps = [min(test_time_steps, self.max_timesteps)]
    elif self.unroll_strategy == "data_flow_max_steps":
      assert (
        test_time_steps is not None
      ), f"You need to pass test_time_steps or not use unroll_strategy '{self.unroll_strategy}''"
      layer_timesteps = [min(test_time_steps, self.max_timesteps)]
    elif self.unroll_strategy == "label_convergence":
      node_states, unroll_steps, converged = self.label_convergence_forward(
        edge_lists,
        node_states,
        pos_lists,
        node_types,
        initial_node_states=old_node_states,
      )
      return node_states, old_node_states, unroll_steps, converged

    for (layer_idx, num_timesteps) in enumerate(layer_timesteps):
      for t in range(num_timesteps):
        messages = self.message[layer_idx](edge_lists, node_states, pos_lists)
        node_states = self.update[layer_idx](messages, node_states, node_types)
    return node_states, old_node_states

  def label_convergence_forward(
    self, edge_lists, node_states, pos_lists, node_types, initial_node_states
  ):
    assert (
      len(self.layer_timesteps) == 1
    ), f"Label convergence only supports one-layer GGNNs, but {len(self.layer_timesteps)} are configured in layer_timesteps: {self.layer_timesteps}"

    stable_steps, i = 0, 0
    old_tentative_labels = self.tentative_labels(
      initial_node_states, node_states
    )

    while True:
      messages = self.message[0](edge_lists, node_states, pos_lists)
      node_states = self.update[0](messages, node_states, node_types)
      new_tentative_labels = self.tentative_labels(
        initial_node_states, node_states
      )
      i += 1

      # return the new node states if their predictions match the old node states' predictions.
      # It doesn't matter during testing since the predictions are the same anyway.
      stability = (
        (new_tentative_labels == old_tentative_labels)
        .to(dtype=torch.get_default_dtype())
        .mean()
      )
      if stability >= self.label_conv_threshold:
        stable_steps += 1

      if stable_steps >= self.label_conv_stable_steps:
        return node_states, i, True

      if i >= self.max_timesteps:  # maybe escape
        return node_states, i, False

      old_tentative_labels = new_tentative_labels

    raise ValueError("Serious Design Error: Unreachable code!")

  def tentative_labels(self, initial_node_states, node_states):
    logits, _ = self.readout(initial_node_states, node_states)
    preds = F.softmax(logits, dim=1)
    predicted_labels = torch.argmax(preds, dim=1)
    return predicted_labels


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

    self.transform = LinearNet(
      self.dim,
      self.dim * self.forward_and_backward_edge_type_count,
      bias=config.use_edge_bias,
      dropout=config.edge_weight_dropout,
    )

    self.pos_transform = None
    if config.position_embeddings:
      self.register_buffer(
        "position_embs",
        PositionEmbeddings()(
          torch.arange(512, dtype=torch.get_default_dtype()),
          config.emb_size,
          dpad=config.selector_size,
        ),
      )
      self.pos_transform = LinearNet(
        self.dim,
        self.dim,
        bias=config.use_edge_bias,
        dropout=config.edge_weight_dropout,
      )

  def forward(self, edge_lists, node_states, pos_lists=None):
    """edge_lists: [<M_i, 2>, ...]"""

    if self.pos_transform:
      pos_gating = 2 * torch.sigmoid(self.pos_transform(self.position_embs))

    # all edge types are handled in one matrix, but we
    # let propagated_states[i] be equal to the case with only edge_type i
    propagated_states = (
      self.transform(node_states)
      .transpose(0, 1)
      .view(self.forward_and_backward_edge_type_count, self.dim, -1)
    )

    messages_by_targets = torch.zeros_like(node_states)
    if self.msg_mean_aggregation:
      device = node_states.device
      bincount = torch.zeros(
        node_states.size()[0], dtype=torch.long, device=device
      )

    for i, edge_list in enumerate(edge_lists):
      edge_targets = edge_list[:, 1]
      edge_sources = edge_list[:, 0]

      messages_by_source = F.embedding(
        edge_sources, propagated_states[i].transpose(0, 1)
      )

      if self.pos_transform:
        pos_list = pos_lists[i]
        pos_by_source = F.embedding(pos_list, pos_gating)
        messages_by_source.mul_(pos_by_source)

      messages_by_targets.index_add_(0, edge_targets, messages_by_source)

      if self.msg_mean_aggregation:
        bins = edge_targets.bincount(minlength=node_states.size()[0])
        bincount += bins

    if self.msg_mean_aggregation:
      divisor = bincount.float()
      divisor[bincount == 0] = 1.0  # avoid div by zero for lonely nodes
      messages_by_targets /= divisor.unsqueeze_(1) + SMALL_NUMBER
    return messages_by_targets


class GGNNLayer(nn.Module):
  def __init__(self, config):
    super().__init__()
    self.dropout = config.graph_state_dropout
    # TODO(github.com/ChrisCummins/ProGraML/issues/27): Maybe decouple hidden
    # GRU size: make hidden GRU size larger and EdgeTrafo size non-square
    # instead? Or implement stacking gru layers between message passing steps.

    self.gru = nn.GRUCell(
      input_size=config.hidden_size, hidden_size=config.hidden_size
    )

    # currently only admits node types 0 and 1 for statements and identifiers.
    self.use_node_types = (
      config.use_node_types if hasattr(config, "use_node_types") else False
    )
    if self.use_node_types:
      self.id_gru = nn.GRUCell(
        input_size=config.hidden_size, hidden_size=config.hidden_size
      )

  def forward(self, messages, node_states, node_types=None):
    if self.use_node_types:
      assert (
        node_types is not None
      ), "Need to provide node_types <N> if config.use_node_types!"
      output = torch.zeros_like(node_states, device=node_states.device)
      stmt_mask = node_types == 0
      output[stmt_mask] = self.gru(messages[stmt_mask], node_states[stmt_mask])
      id_mask = node_types == 1
      output[id_mask] = self.id_gru(messages[id_mask], node_states[id_mask])
    else:
      output = self.gru(messages, node_states)

    if self.dropout > 0.0:
      F.dropout(output, p=self.dropout, training=self.training, inplace=True)
    return output


class PositionEmbeddings(nn.Module):
  def __init__(self):
    super().__init__()

  def forward(self, positions, demb, dpad: int = 0):
    """Transformer-like sinusoidal positional embeddings.
        Args:
        position: 1d long Tensor of positions,
        demb: int    size of embedding vector
      """
    inv_freq = 1 / (10000 ** (torch.arange(0.0, demb, 2.0) / demb))

    sinusoid_inp = torch.ger(positions, inv_freq)
    pos_emb = torch.cat(
      (torch.sin(sinusoid_inp), torch.cos(sinusoid_inp)), dim=1
    )

    if dpad > 0:
      in_length = positions.size()[0]
      pad = torch.zeros((in_length, dpad))
      pos_emb = torch.cat([pos_emb, pad], dim=1)
      assert torch.all(
        pos_emb[:, -1] == torch.zeros(in_length)
      ), f"test failed. pos_emb: \n{pos_emb}"

    return pos_emb

  # def forward(self, positions, dim, out):
  #     assert dim > 0, f'dim of position embs has to be > 0'
  #     power = 2 * (positions / 2) / dim
  #     position_enc = np.array(
  #         [[pos / np.power(10000, 2 * (j // 2) / dim) for j in range(dim)]
  #          for pos in range(n_pos)])
  #     out[:, 0::2] = torch.FloatTensor(np.sin(position_enc[:, 0::2]))
  #     out[:, 1::2] = torch.FloatTensor(np.cos(position_enc[:, 1::2]))
  #     out.detach_()
  #     out.requires_grad = False


########################################
# Output Layer
########################################


class Readout(nn.Module):
  """aka GatedRegression. See Eq. 4 in Gilmer et al. 2017 MPNN."""

  def __init__(self, config):
    super().__init__()
    self.has_graph_labels = config.has_graph_labels
    self.num_classes = config.num_classes

    self.regression_gate = LinearNet(
      2 * config.hidden_size, self.num_classes, dropout=config.output_dropout,
    )
    self.regression_transform = LinearNet(
      config.hidden_size, self.num_classes, dropout=config.output_dropout,
    )

  def forward(
    self, raw_node_in, raw_node_out, graph_nodes_list=None, num_graphs=None
  ):
    gate_input = torch.cat((raw_node_in, raw_node_out), dim=-1)
    gating = torch.sigmoid(self.regression_gate(gate_input))
    nodewise_readout = gating * self.regression_transform(raw_node_out)

    graph_readout = None
    if self.has_graph_labels:
      assert (
        graph_nodes_list is not None and num_graphs is not None
      ), "has_graph_labels requires graph_nodes_list and num_graphs tensors."
      # aggregate via sums over graphs
      device = raw_node_out.device
      graph_readout = torch.zeros(num_graphs, self.num_classes, device=device)
      graph_readout.index_add_(
        dim=0, index=graph_nodes_list, source=nodewise_readout
      )
    return nodewise_readout, graph_readout


class LinearNet(nn.Module):
  """Single Linear layer with WeightDropout, ReLU and Xavier Uniform
  initialization. Applies a linear transformation to the incoming data:
  :math:`y = xA^T + b`

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
    self.test = nn.Parameter(torch.Tensor(out_features, in_features))
    if bias:
      self.bias = nn.Parameter(torch.Tensor(out_features))
    else:
      self.register_parameter("bias", None)
    self.reset_parameters()

  def reset_parameters(self):
    nn.init.xavier_uniform_(self.test)
    # TODO(github.com/ChrisCummins/ProGraML/issues/27): why use xavier_uniform,
    # not kaiming init? Seems old-school
    if self.bias is not None:
      #    fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
      #    bound = 1 / math.sqrt(fan_in)
      #    nn.init.uniform_(self.bias, -bound, bound)
      nn.init.zeros_(self.bias)

  def forward(self, input):
    if self.dropout > 0.0:
      w = F.dropout(self.test, p=self.dropout, training=self.training)
    else:
      w = self.test
    return F.linear(input, w, self.bias)

  def extra_repr(self):
    return "in_features={}, out_features={}, bias={}, dropout={}".format(
      self.in_features, self.out_features, self.bias is not None, self.dropout,
    )


#######################################
# Adding Graph Level Features to Model


class AuxiliaryReadout(nn.Module):
  """Produces per-graph predictions by combining
    the per-graph predictions with auxiliary features"""

  # TODO(github.com/ChrisCummins/ProGraML/issues/27): I don't like that we only
  # introduce the global features AFTER the per node predictions have been made
  # and not while we do those! This is limiting the expressivity of the model.
  def __init__(self, config):
    super().__init__()
    self.num_classes = config.num_classes
    self.log1p_graph_x = getattr(config, "log1p_graph_x", False)
    assert (
      config.has_graph_labels
    ), "We expect aux readout in combination with graph labels, not node labels"
    self.feed_forward = None

    self.batch_norm = nn.BatchNorm1d(config.num_classes + config.aux_in_len)
    self.feed_forward = nn.Sequential(
      nn.Linear(
        config.num_classes + config.aux_in_len, config.aux_in_layer_size,
      ),
      nn.ReLU(),
      nn.Dropout(1 - config.output_dropout),
      nn.Linear(config.aux_in_layer_size, config.num_classes),
    )

  def forward(self, graph_features, auxiliary_features):
    assert (
      graph_features.size()[0] == auxiliary_features.size()[0]
    ), "every graph needs aux_features. Dimension mismatch."
    if self.log1p_graph_x:
      auxiliary_features.log1p_()

    aggregate_features = torch.cat((graph_features, auxiliary_features), dim=1)

    normed_features = self.batch_norm(aggregate_features)
    out = self.feed_forward(normed_features)
    return out, graph_features
