#!/bin/bash

export JULIA_NUM_THREADS=1

dirr=data/new/data4x4_N14_dtau0.02/
comments=()

# Print
for ((i = 1; i <= 10; i++))
do
    workdir=$dirr/$i
    mkdir -p $workdir
    cmd="julia --sysimage ~/.julia/sysimages/sys_itensors.so ImagMPS.jl $workdir >> $workdir/out$i &"
    comments+=("$cmd")   # append to array
    echo $cmd 
done

echo "Do you want to continue? (y/n)"
read answer

# Check
if [[ "$answer" != "y" && "$answer" != "Y" ]]; then
    echo "Aborting."
    exit 1
fi

# Run
for cmd in "${comments[@]}"
do
  eval "$cmd"
done
