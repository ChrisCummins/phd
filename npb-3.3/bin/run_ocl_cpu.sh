#!/bin/bash

case $1 in
  "bt" ) PROGS=(bt);;
  "cg" ) PROGS=(cg);;
  "dc" ) PROGS=(dc);;
  "ep" ) PROGS=(ep);;
  "ft" ) PROGS=(ft);;
  "is" ) PROGS=(is);;
  "lu" ) PROGS=(lu);;
  "mg" ) PROGS=(mg);;
  "sp" ) PROGS=(sp);;
  "ua" ) PROGS=(ua);;
  *    ) PROGS=(bt ep ft mg sp cg is lu);;
esac

case $2 in
  [sS] ) CLASSES=(S);;
  [wW] ) CLASSES=(W);;
  [aA] ) CLASSES=(A);;
  [bB] ) CLASSES=(B);;
  [cC] ) CLASSES=(C);;
  [dD] ) CLASSES=(D);;
  [eE] ) CLASSES=(E);;
  *    ) CLASSES=(S W A B C);;
esac

DEV=cpu
COUNT=1

for PROG in ${PROGS[@]}; do
  PROGI="`echo $PROG | tr '[:lower:]' '[:upper:]'`"
  for CLASS in ${CLASSES[@]}; do
    LOG_FILE="log/$PROGI.$CLASS.$DEV.out"
    rm -f $LOG_FILE
    touch $LOG_FILE

    for i in `seq 1 $COUNT`; do
      echo "#$i: OPENCL_DEVICE_TYPE=$DEV ./$PROG.$CLASS.x > $LOG_FILE"
      bash -c "time OPENCL_DEVICE_TYPE=$DEV ./$PROG.$CLASS.x ../$PROGI/ >> $LOG_FILE" 2>> $LOG_FILE
      cat $LOG_FILE
    done
  done
done
