# Experiments in Compiler Reachability Analysis

**To generate a control flow graph:**

```sh
$ bazel run //experimental/compilers/reachability -- \
    --reachability_scaling_param=.57 \
    --reachability_num_nodes=5 \
    --reachability_seed=1
```

The output is a list of node names and successors.

**To generate training data:**

```sh
$ bazel run //experimental/compilers/reachability:make_training_data -- \
    --reachability_num_training_graphs=10000 \
    --reachability_scaling_param=.57 \
    --reachability_num_nodes=5
```

The output is a training data proto.
