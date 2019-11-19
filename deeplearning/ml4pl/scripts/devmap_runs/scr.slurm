#!/bin/bash
#SBATCH --job-name=devmap
#SBATCH --time=00:30:00
#SBATCH --partition=total
#SBATCH --gres=gpu:1
#SBATCH --mail-type=ALL
#SBATCH --mail-user=zacharias.vf@gmail.com
#SBATCH --exclude=ault07,ault08

source /users/zfisches/.bash_profile;
cd /users/zfisches/phd;

# ./deeplearning/ml4pl/scripts/devmap_runs/debug_run.sh


srun bash bazel run //deeplearning/ml4pl/models/ggnn:ggnn -- \
    --graph_db='sqlite:////users/zfisches/db/devmap_amd_20191113.db' \
    --log_db='sqlite:////users/zfisches/logs_node_lstm_series.db' \
    --working_dir='/users/zfisches/logs_ggnn_devmap_20191119'\
    --num_epochs=150 \
    --alsologtostderr \
    --position_embeddings=off \
    --layer_timesteps=2,2,2 \
    --inst2vec_embeddings=random \
    --output_layer_dropout_keep_prob=0.5 \
    --graph_state_dropout_keep_prob=1.0 \
    --edge_weight_dropout_keep_prob=0.8 \
    --batch_size=18000 \
    --kfold \
    --manual_tag=debug-sbatch
