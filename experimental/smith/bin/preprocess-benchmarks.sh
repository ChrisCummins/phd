#!/usr/bin/env bash

set -eu

cd benchmarks
smith-preprocess -f -i $(ls | grep -v 'synthetic.cl')
