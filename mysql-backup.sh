host=cc1
user=cec
table=clsmith
dstdir=~/src/project_b/data

set -eux

ssh $host "mysqldump -u $user -p $table > /tmp/$table.mysql"
ssh $host "tar cjvf $dstdir/$table-$(date '+%Y-%m-%d').mysql.tar.bz2 -C /tmp $table.mysql"
ssh $host "rm -v /tmp/$table.mysql"
