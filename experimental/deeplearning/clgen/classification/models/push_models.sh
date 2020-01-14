#!/usr/bin/env bash

set -eux

model_file=models-$(hostname).tar.bz2

rm -f $model_file
tar cjvf $model_file *

set +e
test $(hostname) = cc1 || scp $model_file cc1:phd/experimental/clgen/classification/models
test $(hostname) = cc2 || scp $model_file cc2:phd/experimental/clgen/classification/models
test $(hostname) = cc3 || scp $model_file cc3:phd/experimental/clgen/classification/models
true
