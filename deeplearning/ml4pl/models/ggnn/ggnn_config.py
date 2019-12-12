import enum
from typing import List

from labm8.py import app

FLAGS = app.FLAGS


class NodeTextEmbeddingType(enum.Enum):
    """Type of node text embedding to use."""

    INST2VEC_CONSTANT = 1
    ZERO_CONSTANT = 2
    RANDOM_CONSTANT = 3
    INST2VEC_FINETUNE = 4
    RANDOM = 5


class GGNNConfig(object):
    def __init__(
        self, num_classes: int, has_graph_labels: bool, edge_type_count: int = 3
    ):
        self.inst2vec_embeddings: NodeTextEmbeddingType = FLAGS.inst2vec_embeddings()
        self.lr: float = FLAGS.learning_rate
        self.clip_grad_norm: bool = FLAGS.clamp_gradient_norm  # use 6.0 as default! Set to 0.0 for no clipping.
        self.vocab_size: int = 8568  # embeddings = list(self.graph_db.embeddings_tables)
        self.emb_size: int = 200
        self.use_selector_embeddings: bool = True
        self.selector_size: int = 2 if self.use_selector_embeddings else 0
        # TODO(github.com/ChrisCummins/ProGraML/issues/27):: Maybe refactor non-rectangular edge passing matrices for independent hidden size.
        # hidden size of the whole model
        self.hidden_size: int = self.emb_size + self.selector_size
        ###############

        self.edge_type_count: int = edge_type_count
        self.layer_timesteps: List[int] = [
            int(x) for x in FLAGS.layer_timesteps
        ]
        self.use_edge_bias: bool = FLAGS.use_edge_bias
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
        self.log1p_graph_x = FLAGS.log1p_graph_x

        self.intermediate_loss_weight: float = FLAGS.intermediate_loss_weight
