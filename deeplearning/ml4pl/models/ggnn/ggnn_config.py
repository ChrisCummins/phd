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
"""Configuration for GGNN models."""
from typing import List

from labm8.py import app

FLAGS = app.FLAGS


class GGNNConfig(object):
  def __init__(
    self,
    num_classes: int,
    has_graph_labels: bool,
    edge_type_count: int = 3,
    has_aux_input: bool = False,
  ):
    # not implemented here, because not relevant:
    # train_subset, random_seed,
    ###############

    self.lr: float = FLAGS.learning_rate
    self.clip_grad_norm: bool = FLAGS.clamp_gradient_norm  # use 6.0 as default when clipping! Set to 0.0 for no clipping.

    self.vocab_size: int = 8568
    self.inst2vec_embeddings = FLAGS.inst2vec_embeddings
    self.emb_size: int = 200

    # TODO This should be turned off on devmap!
    self.use_selector_embeddings: bool = True
    self.selector_size: int = 2 if self.use_selector_embeddings else 0
    # TODO(github.com/ChrisCummins/ProGraML/issues/27):: Maybe refactor non-rectangular edge passing matrices for independent hidden size.
    # hidden size of the whole model
    self.hidden_size: int = self.emb_size + self.selector_size
    self.position_embeddings: bool = FLAGS.position_embeddings
    ###############

    self.edge_type_count: int = edge_type_count
    self.layer_timesteps: List[int] = [int(x) for x in FLAGS.layer_timesteps]
    self.use_edge_bias: bool = FLAGS.use_edge_bias
    # NB: This is currently unused as the only way of differentiating the type
    # of node is by looking at the encoded 'x' value, but may be added in the
    # future.
    self.use_node_types: bool = False
    self.msg_mean_aggregation: bool = FLAGS.msg_mean_aggregation
    self.backward_edges: bool = True
    ###############

    self.num_classes: int = num_classes
    self.aux_in_len: int = 2
    self.aux_in_layer_size: int = FLAGS.aux_in_layer_size
    self.output_dropout: float = FLAGS.output_layer_dropout  # dropout prob = 1-keep_prob
    self.edge_weight_dropout: float = FLAGS.edge_weight_dropout
    self.graph_state_dropout: float = FLAGS.graph_state_dropout
    ###############

    self.has_graph_labels: bool = has_graph_labels
    self.has_aux_input: bool = has_aux_input
    self.log1p_graph_x = FLAGS.log1p_graph_x

    self.intermediate_loss_weight: float = FLAGS.intermediate_loss_weight
    #########
    self.unroll_strategy = FLAGS.unroll_strategy
    self.test_layer_timesteps = FLAGS.test_layer_timesteps
    self.max_timesteps = 1000
    self.label_conv_threshold: float = FLAGS.label_conv_threshold
    self.label_conv_stable_steps: int = FLAGS.label_conv_stable_steps
