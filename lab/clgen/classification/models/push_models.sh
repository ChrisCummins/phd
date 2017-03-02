#!/usr/bin/env bash
set -eux

model_file=models-$(hostname).tar.bz2

rm -f $model_file
tar cjvf $model_file *
scp $model_file cc3:phd/lab/clgen/classification/models
