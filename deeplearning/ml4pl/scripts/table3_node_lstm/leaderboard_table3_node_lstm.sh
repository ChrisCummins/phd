bazel run //deeplearning/ml4pl/models/eval:export_leaderboard -- \
--google_sheets_credentials='/users/zfisches/google_api.json' \
--log_db='sqlite:////users/zfisches/logs_node_lstm_series.db' \
--worksheet=lstm_table5 \
--extra_model_flags=inst2vec_embeddings,output_layer_dropout_keep_prob \
--extra_flags=batch_size,\
max_train_per_epoch,\
max_val_per_epoch,\
use_lr_schedule,\
learning_rate,\
layer_timesteps,\
edge_weight_dropout_keep_prob,\
manual_tag \
bytecode_encoder \
