# Push and pull results to server.

push() {
    local dir=$1

    cd $dir

    local archive=$(date +"$HOSTNAME-%y-%m-%d.%H-%M.tar.bz2")

    if [[ $HOSTNAME == "monza" ]]; then
        ~/phd/experimental/smith/driver/mkerros intel $dir
        ~/phd/experimental/smith/driver/mkerros amd $dir
        tar -jcvf $archive intel.csv intel-errors.csv amd.csv amd-errors.csv
    elif [[ $HOSTNAME == "diana" ]]; then
        ~/phd/experimental/smith/driver/mkerros nvidia $dir
        tar -jcvf $archive nvidia.csv nvidia-errors.csv
    else
        echo "Unkown hostname '$HOSTNAME'" >&2
        exit 1
    fi

    echo "Prepared tarball $archive"
    scp ./$archive brendel.inf.ed.ac.uk:
}


pull() {
    local dir=$1
    local archive=$2

    cd $dir

    scp brendel.inf.ed.ac.uk:$archive .
    tar -xvf $archive
}
