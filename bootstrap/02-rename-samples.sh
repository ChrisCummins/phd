#!/usr/bin/env bash

set -eux

samples_db="$(clgen --sampler-dir model.json sampler.json)/kernels.db"

mkdir -pv ../data/bootstrap/
cp -v "$samples_db" ../data/bootstrap/samples.db

clgen db dump --dir ../data/bootstrap/samples.db ../data/bootstrap/samples

for sample in $(ls ../data/bootstrap/samples/*); do
    sed -i 's/__kernel void A/__kernel void entry/' "$sample"
done
