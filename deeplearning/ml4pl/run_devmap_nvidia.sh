for i in {0..9}
do
	bazel run //deeplearning/ml4pl/models/ggnn -- \
		--log_db='sqlite:////home/zacharias/db/logs/ggnn_devmap_nvidia_20191109.db' \
		--working_dir='/tmp/devmap' \
		--num_epochs=200 \
		--alsologtostderr \
		--graph_db='sqlite:////home/zacharias/db/devmap_nvidia_20191107.db' \
		--test_group=$i \
		--val_group=$(((i+1) % 10)) \
		--position_embeddings=true \
		--inst2vec_embeddings='constant' \
		--output_layer_dropout_keep_prob=0.5
done
