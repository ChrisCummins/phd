host=cc1
db=DeepSmith_1
dstdir=~/src/dsmith/data

set -eux

ssh $host "mysqldump $db > /tmp/$db.mysql"
ssh $host "tar cjvf $dstdir/$db-$(date '+%Y-%m-%d').mysql.tar.bz2 -C /tmp $db.mysql"
ssh $host "rm -v /tmp/$db.mysql"
