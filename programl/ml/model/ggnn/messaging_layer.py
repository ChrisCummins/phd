# Copyright 2019-2020 the ProGraML authors.
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
"""The GGNN messaging layer."""
import torch
import torch.nn.functional as F
from torch import nn

from programl.ml.model.ggnn.linear_net import LinearNet
from programl.ml.model.ggnn.position_embeddings import PositionEmbeddings

SMALL_NUMBER = 1e-8


class MessagingLayer(nn.Module):
  """takes an edge_list (for a single edge type) and node_states <N, D+S> and
  returns incoming messages per node of shape <N, D+S>"""

  def __init__(
    self,
    text_embedding_dimensionality: int,
    selector_embedding_dimensionality: int,
    forward_edge_type_count: int,
    use_backward_edges: bool,
    use_position_embeddings: bool,
    use_edge_bias: bool,
    msg_mean_aggregation: bool,
    edge_weight_dropout: float,
  ):
    super().__init__()
    self.forward_and_backward_edge_type_count = (
      forward_edge_type_count * 2
      if use_backward_edges
      else forward_edge_type_count
    )
    self.msg_mean_aggregation = msg_mean_aggregation
    self.dimensionality = (
      text_embedding_dimensionality + selector_embedding_dimensionality
    )

    self.transform = LinearNet(
      self.dimensionality,
      self.dimensionality * self.forward_and_backward_edge_type_count,
      bias=use_edge_bias,
      dropout=edge_weight_dropout,
    )

    self.pos_transform = None
    if use_position_embeddings:
      self.register_buffer(
        "position_embs",
        PositionEmbeddings()(
          torch.arange(512, dtype=torch.get_default_dtype()),
          demb=text_embedding_dimensionality,
          # Padding to exclude selector embeddings from positional offset.
          dpad=selector_embedding_dimensionality,
        ),
      )
      self.pos_transform = LinearNet(
        self.dimensionality,
        self.dimensionality,
        bias=use_edge_bias,
        dropout=edge_weight_dropout,
      )

  def forward(self, edge_lists, node_states, pos_lists=None):
    """edge_lists: [<M_i, 2>, ...]"""
    # Conditionally set variables:
    bincount = None
    pos_gating = None

    if self.pos_transform:
      pos_gating = 2 * torch.sigmoid(self.pos_transform(self.position_embs))

    # all edge types are handled in one matrix, but we
    # let propagated_states[i] be equal to the case with only edge_type i
    propagated_states = (
      self.transform(node_states)
      .transpose(0, 1)
      .view(self.forward_and_backward_edge_type_count, self.dimensionality, -1)
    )

    messages_by_targets = torch.zeros_like(node_states)
    if self.msg_mean_aggregation:
      device = node_states.device
      bincount = torch.zeros(
        node_states.size()[0], dtype=torch.long, device=device
      )

    for i, edge_list in enumerate(edge_lists):
      edge_sources = edge_list[:, 0]
      edge_targets = edge_list[:, 1]

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
