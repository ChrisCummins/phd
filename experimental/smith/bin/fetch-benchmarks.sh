#!/usr/bin/env bash
set -eu

mkdir -pv benchmarks

extract () {
        local remove_archive
        local success
        local file_name
        local extract_dir
        if (( $# == 0 ))
        then
                echo "Usage: extract [-option] [file ...]"
                echo
                echo Options:
                echo "    -r, --remove    Remove archive."
                echo
                echo "Report bugs to <sorin.ionescu@gmail.com>."
        fi
        remove_archive=1
        if [[ "$1" = "-r" ]] || [[ "$1" = "--remove" ]]
        then
                remove_archive=0
                shift
        fi
        while (( $# > 0 ))
        do
                if [[ ! -f "$1" ]]
                then
                        echo "extract: '$1' is not a valid file" >&2
                        shift
                        continue
                fi
                success=0
                file_name="$( basename "$1" )"
                extract_dir="$( echo "$file_name" | sed "s/\.${1##*.}//g" )"
                case "$1" in
                        (*.tar.gz|*.tgz) [ -z $commands[pigz] ] && tar zxvf "$1" || pigz -dc "$1" | tar xv ;;
                        (*.tar.bz2|*.tbz|*.tbz2) tar xvjf "$1" ;;
                        (*.tar.xz|*.txz) tar --xz --help &> /dev/null && tar --xz -xvf "$1" || xzcat "$1" | tar xvf - ;;
                        (*.tar.zma|*.tlz) tar --lzma --help &> /dev/null && tar --lzma -xvf "$1" || lzcat "$1" | tar xvf - ;;
                        (*.tar) tar xvf "$1" ;;
                        (*.gz) [ -z $commands[pigz] ] && gunzip "$1" || pigz -d "$1" ;;
                        (*.bz2) bunzip2 "$1" ;;
                        (*.xz) unxz "$1" ;;
                        (*.lzma) unlzma "$1" ;;
                        (*.Z) uncompress "$1" ;;
                        (*.zip|*.war|*.jar|*.sublime-package) unzip "$1" -d $extract_dir ;;
                        (*.rar) unrar x -ad "$1" ;;
                        (*.7z) 7za x "$1" ;;
                        (*.deb) mkdir -p "$extract_dir/control"
                                mkdir -p "$extract_dir/data"
                                cd "$extract_dir"
                                sudo $apt_pref remove vx "../${1}" > /dev/null
                                cd control
                                tar xzvf ../control.tar.gz
                                cd ../data
                                tar xzvf ../data.tar.gz
                                cd ..
                                rm *.tar.gz debian-binary
                                cd .. ;;
                        (*) echo "extract: '$1' cannot be extracted" >&2
                                success=1  ;;
                esac
                (( success = $success > 0 ? $success : $? ))
                (( $success == 0 )) && (( $remove_archive == 0 )) && rm "$1"
                shift
        done
}

####################
# Benchmark Suites #
####################

# parboil

# rodinia 3.1
wget http://www.cs.virginia.edu/~kw5na/lava/Rodinia/Packages/Current/rodinia_3.1.tar.bz2
extract rodinia_3.1.tar.gz2
mv -v rodinia_3.1 rodinia-3.1

# polybench gpu 1.0
wget http://www.cse.ohio-state.edu/~pouchet/software/polybench/download/polybench-gpu-1.0.tar.gz
extract polybench-gpu-1.0.tar.gz

# shoc 1.1.5
git clone https://github.com/vetter/shoc.git shoc-1.1.5
cd shoc-1.1.5
git checkout 3bd13c7982cd8ac85bc4dfa17cbd64bb94ccf0ab
cd ..

# amd

#nvidia
