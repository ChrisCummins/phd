from labm8.py import app

FLAGS = app.FLAGS


class GGNNConfig(object):
  def __init__(self, y_dimensionality: int, has_graph_labels: bool):
    ##############
    self.vocab_size = 8868  # embeddings = list(self.graph_db.embeddings_tables)
    self.hidden_size = 202  # 200~
    self.inst2vec_embeddings = FLAGS.inst2vec_embeddings
    self.emb_size = 200
    self.use_selector_embeddings = True
    self.selector_size = 2
    ###############
    self.num_classes = 2  # binary classification!
    self.aux_in_len = 2
    self.auxiliary_inputs_dense_layer_size = (
      FLAGS.auxiliary_inputs_dense_layer_size
    )
    self.output_dropout = (
      1.0 - FLAGS.output_layer_dropout_keep_prob
    )  # dropout prob = 1-keep_prob. (self.placeholders["output_layer_dropout_keep_prob"])
    self.labels_dimensionality = y_dimensionality
    self.has_graph_labels = has_graph_labels
    self.graph_loss_weight = FLAGS.intermediate_loss_discount_factor
    self.lr = FLAGS.learning_rate
