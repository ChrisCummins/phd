bazel run //deeplearning/ml4pl/models/eval:export_leaderboard -- \
--log_db='sqlite:////users/zfisches/logs_node_lstm_series.db' \
--worksheet=node_lstm_series \
--extra_model_flags=inst2vec_embeddings,output_layer_dropout_keep_prob \
--extra_flags=batch_size,\
max_train_per_epoch,\
max_val_per_epoch,\
use_lr_schedule,\
learning_rate,\
layer_timesteps,\
edge_weight_dropout_keep_prob,\
manual_tag,\
bytecode_encoder \
