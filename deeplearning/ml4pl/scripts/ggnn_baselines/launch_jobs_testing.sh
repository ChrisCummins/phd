#!/bin/bash
cd testing;
for file in *.sh; do
    export APP=$file
    read -n1 -p $"$file -- Submit this job [y,n]:" doit
    case $doit in
      y|Y) echo -e "\033[2K"
       sbatch -n 1 < $file;
        echo ----- Submitted $file ----- ;;
      n|N) echo -e "\033[2K"
        echo skipping $file... ;;
      *) echo Unknown choice. skipping $file... ;;
    esac
    #sleep 0.1
done