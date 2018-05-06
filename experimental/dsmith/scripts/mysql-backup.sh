set -eux

main() {
    local db=$1
    local dstdir=$2

    set -eux

    mysqldump $db > /tmp/$db.mysql
    tar cjvf $dstdir/$db-$(date '+%Y-%m-%d').mysql.tar.bz2 -C /tmp $db.mysql
    rm -v /tmp/$db.mysql
}

main $@
