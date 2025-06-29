#!/bin/bash

export JULIA_NUM_THREADS=1

workdir="test"
para="para"
comments=()

# Print
for ((i = 1; i <= 3; i++))
do
    cmd="julia --sysimage ~/.julia/sysimages/sys_itensors.so ImagMPSComb.jl $para --dir=$workdir --suffix=_$i >> $workdir/out$i &"
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
mkdir -p $workdir

for cmd in "${comments[@]}"
do
  eval "$cmd"
done
