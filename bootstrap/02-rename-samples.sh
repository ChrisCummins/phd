#!/usr/bin/env bash

set -eux

samples_db="$(clgen --sampler-dir model.json sampler.json)/kernels.db"

mkdir -pv ../data/bootstrap/
cp -v "$samples_db" ../data/bootstrap/samples.db

rm -rf ../data/bootstrap/samples
clgen db dump --dir ../data/bootstrap/samples.db ../data/bootstrap/samples
clgen db dump --file-sep ../data/bootstrap/samples.db ../data/bootstrap/samples.cl

for sample in $(ls ../data/bootstrap/samples/*); do
    sed -i 's/__kernel void A/__kernel void entry/' "$sample"
done
