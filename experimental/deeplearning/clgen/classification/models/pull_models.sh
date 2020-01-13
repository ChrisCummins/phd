set -ux

test -f models-cc1.tar.bz2 && { test $(hostname) = cc1 || tar xjvf models-cc1.tar.bz2; }
test -f models-cc2.tar.bz2 && { test $(hostname) = cc2 || tar xjvf models-cc2.tar.bz2; }
test -f models-cc3.tar.bz2 && { test $(hostname) = cc3 || tar xjvf models-cc3.tar.bz2; }
true
