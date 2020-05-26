# Dataflow Dataset

The data flow dataset contains LLVM IRs taken from a wide range of projects and source programming
languages, and includes labels for several compiler data flows.

## Directory Layout

The uncompressed dataset uses the following layout:

* `cdfg/`
    * Directory containing [CDFG](/programl/graph/format/cdfg.h) representations of programs.
    * `cdfg/<source>.<id>.<lang>.ProgramGraph.pb`
        * A [ProgramGraph](/programl/proto/program_graph.proto) protocol buffer of a program in the CDFG representation.
    * `cdfg/<source>.<id>.<lang>.NodeIndexList.pb`
        * A [NodeIndexList](/programl/proto/node.proto) protocol buffer containing a list of node indices used to translate from a full ProGraML graph to a CDFG graph.
* `labels/`
    * Directory containing machine learning features and labels for programs for compiler data flow analyses.
    * `labels/<analysis>/<source>.<id>.<lang>.ProgramFeaturesList.pb`
        * A [ProgramFeaturesList](/programl/proto/program_graph_features.proto) protocol buffer containing a list of features resulting from running a data flow analysis on a program.
* `graphs/`
    * Directory containing ProGraML representations of LLVM IRs.
    * `graphs/<source>.<id>.<lang>.ProgramGraph.pb`
        * A [ProgramGraph](/programl/proto/program_graph.proto) protocol buffer of an LLVM IR in the ProGraML representation.
* `ir/`
    * Directory containing LLVM IRs, stored as a pair of files: an `.Ir.pb` protocol buffer and a `.ll` text file.
    * `ir/<source>.<id>.<lang>.Ir.pb`
        * An [Ir](/programl/proto/ir.proto) protocol buffer containing an LLVM IR.
    * `ir/<source>.<id>.<lang>.ll`
        * An LLVM IR in text format, as produced by `clang -emit-llvm -S`.
* `test/`
    * A directory containing symlinks to graphs in the `graphs/` directory, indicating which graphs should be used as part of the test set.
* `train/`
    * A directory containing symlinks to graphs in the `graphs/` directory, indicating which graphs should be used as part of the training set.
* `val/`
    * A directory containing symlinks to graphs in the `graphs/` directory, indicating which graphs should be used as part of the validation set.
* `vocal/`
    * Directory containing vocabulary files.
    * `vocab/<type>.csv`
      * A vocabulary file, which lists unique node texts, their frequency in the dataset, and the cumulative proportion of total unique node texts that is covered.


After running the dataflow experiments, log files are produced in this directory with the structure:

* `logs/<model>/<analysis>/<run_id>`
    * Logs recording the performance of a model running on a specific analysis. `<run_id>` is a timestamp used to distinguish multiple runs of the same model on the same analysis.
    * `logs/<model>/<analysis>/<run_id>/flags.txt`
        * A text file listing the command-line flags that were set during this run, either explicitly, or through default values.
    * `logs/<model>/<analysis>/<run_id>/build_info.json`
        * A JSON file describing the state of the code base that was used. This, along with the `flags.txt`, fully describes the experimental setup.
    * `logs/<model>/<analysis>/<run_id>/checkpoints/`
        * Directory containing model checkpoints, which are restorable snapshots of model state.
        * `logs/<model>/<analysis>/<run_id>/checkpoints/<epoch>.Checkpoint.pb`
            * A [Checkpoint](/programl/proto/checkpoint.proto) protocol buffer recording a snapshot of the model state at the end of the given epoch.
    * `logs/<model>/<analysis>/<run_id>/epochs/`
        * `logs/<model>/<analysis>/<run_id>/epochs/<epoch>.EpochList.pbtxt`
            * An [EpochList](/programl/proto/epoch.proto) protocol buffer recording the performance of the model during this epoch.
    * `logs/<model>/<analysis>/<run_id>/graph_loader/`
        * A directory containing graph loader log files.
        * `logs/<model>/<analysis>/<run_id>/graph_loader/<epoch>.txt`
            * A text file containing a list of the graphs that were loaded in a given epoch, in the order they were read.


### File Types

To save disk space, most of the protocol buffers are stored in binary wire format, indicated by the
`.pb` file extension. The [pbq](/programl/cmd/pbq.cc) program can used to decode binary
protocol buffers into a human-readable text format. To do so, you must specify the type of the
message, indicated using a `.<type>.pb` suffix on the filename. For example, to decode the
ProgramGraph protocol buffer `graphs/foo.c.ProgramGraph.pb`, run:

```sh
$ pbq ProgramGraph --stdin_fmt=pb < graphs/foo.c.ProgramGraph.pb
```
