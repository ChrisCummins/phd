# Experiments in Compiler Reachability Analysis

**To generate a control flow graph:**

```sh
$ bazel run //experimental/compilers/reachability -- \
    --reachability_scaling_param=.57 \
    --reachability_num_nodes=5 \
    --reachability_seed=1
```

The output is a list of node names and successors.

**To train a model:**

```sh
$ bazel run //experimental/compilers/reachability:train_model -- \
    --reachability_num_training_graphs=10000 \
    --reachability_num_testing_graphs=1000 \
    --reachability_scaling_param=.57 \
    --reachability_num_nodes=5
```
