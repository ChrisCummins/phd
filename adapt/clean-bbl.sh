#!/usr/bin/env bash

set -eu

bbl=$1

sed -i '/\\field{annotation}/d' $bbl
sed -i '/\\field{abstract}/d' $bbl
