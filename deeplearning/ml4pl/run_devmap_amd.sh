for i in {0..9}
do
	bazel run //deeplearning/ml4pl/models/ggnn -- \
		--log_db='sqlite:////home/zacharias/db/logs/ggnn_devmap_amd_20191109_25epochs.db' \
		--working_dir='/tmp/devmap' \
		--num_epochs=25 \
		--alsologtostderr \
		--graph_db='sqlite:////home/zacharias/db/devmap_amd_20191107.db' \
		--test_group=$i \
		--val_group=$(((i+1) % 10)) \
		--position_embeddings=true \
		--inst2vec_embeddings='constant' \
		--output_layer_dropout_keep_prob=0.5
done
