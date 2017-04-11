host=cc1
user=cec
db=project_b
dstdir=~/src/project_b/data

set -eux

ssh $host "mysqldump -u $user -p $db > /tmp/$db.mysql"
ssh $host "tar cjvf $dstdir/$db-$(date '+%Y-%m-%d').mysql.tar.bz2 -C /tmp $db.mysql"
ssh $host "rm -v /tmp/$db.mysql"
