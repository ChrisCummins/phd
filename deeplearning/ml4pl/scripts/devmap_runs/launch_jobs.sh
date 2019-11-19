#!/bin/bash

for file in *.sh; do
    export APP=$file
    #echo $file
    sbatch -n 1 < echo.sh
done