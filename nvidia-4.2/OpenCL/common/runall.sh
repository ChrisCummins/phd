cd $1
find . -name "ocl*" -executable > samples.list
$SHELL samples.list
