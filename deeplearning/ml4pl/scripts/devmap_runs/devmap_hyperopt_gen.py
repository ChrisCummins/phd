import hashlib
import os
from pathlib import Path
import itertools

def stamp(stuff):
    hash_object = hashlib.sha1(str(stuff).encode('utf-8'))
    hex_dig = hash_object.hexdigest()
    return hex_dig[:7]

def ggnn_devmap_hyperopt(start_step=0, gpus=[0,1,2,3], how_many=None, test_groups=[0,1,2,3,4,5,6,7,8,9]):
    # GGNN DEVMAP HYPER OPT SERIES
    # fix
    log_db = 'ggnn_devmap_hyperopt.db'
    #devices = [0,1,2,3]
    # flexible
    state_drops = ['1.0', '0.95', '0.9']
    timestep_choices = ['2,2,2', '3,3', '30']
    datasets = ["amd", "nvidia"]
    batch_sizes = ['18000','40000', '9000']
    out_drops = ['0.5', '0.8', '1.0']
    edge_drops = ['0.8', '1.0', '0.9']
    embs = ['constant', 'random']
    pos_choices = ['off', 'fancy']
    
    # order is important!
    template_keys = ['state_drop','timesteps','dataset','batch_size','out_drop','edge_drop','emb','pos']
    opt_space = [[state_drops[0]], [timestep_choices[0]], datasets, [batch_sizes[0]], out_drops, edge_drops, embs, pos_choices]
    configs = list(itertools.product(*opt_space))
    length = len(configs)
    print(f"Generating {length} configs per test group.\n")
    
    # None -> all.
    if not how_many:
        how_many = length - start_step
    
    # cd phd; export CUDA_VISIBLE_DEVICES={device}; \
    
    template = """#!/bin/bash\n \
bazel run //deeplearning/ml4pl/models/ggnn:ggnn -- \
--graph_db='sqlite:////users/zfisches/db/devmap_{dataset}_20191113.db' \
--log_db='sqlite:////users/zfisches/{log_db}' \
--working_dir='/users/zfisches/logs_ggnn_devmap_20191117' \
--num_epochs=150 \
--alsologtostderr \
--position_embeddings={pos} \
--layer_timesteps={timesteps} \
--inst2vec_embeddings={emb} \
--output_layer_dropout_keep_prob={out_drop} \
--graph_state_dropout_keep_prob={state_drop} \
--edge_weight_dropout_keep_prob={edge_drop}
--batch_size={batch_size} \
--manual_tag=HyperOpt-{i:03d}-{stamp} \
    """

    if test_groups == 'kfold':
        template += template + '--kfold'
        test_groups=['kfold']
    else:
        template += "--test_group={test_group} --val_group={val_group}"
    for g in test_groups:
        print(f'############### TEST GROUP {g} ##############\n')
        for i in range(start_step, start_step + how_many):
            config = dict(zip(template_keys, configs[i]))
            print(f'############## HYPEROPT {i} ###################\n')
            print(config)
            print('')
            config.update({
                'stamp': stamp(config),
                'log_db': log_db,
                'i': i,
                # 'device': gpus[i % len(gpus)]
            })
            if not g == 'kfold':
                config.update({'test_group': g,
                              'val_group': (g + 1) % 9})
            print(template.format(**config))
            print('\n\n\n\n')
            
            (base_path / str(g)).mkdir(parents=True, exist_ok=True)
            with open(base_path / str(g) / f'run_{g}_{i:03d}.sh', 'w') as f:
                f.write(template.format(**config))


if __name__ == '__main__':
    import sys
    tg = sys.argv[1]
    tgs = [int(tg)]
    base_path = Path('/home/zacharias/ml4pl/deeplearning/ml4pl/scripts/devmap_runs/')
    ggnn_devmap_hyperopt(test_groups=tgs)
