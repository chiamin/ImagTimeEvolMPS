#!/bin/bash

export JULIA_NUM_THREADS=1

workdir="/home/nsysu601/ImagMPS/data_MixedEstimator/data4x4_N14/dtau0.02/"
comments=()

# Print
for i in 20 40 60 80 100 120 140 160 180 200
do
    cmd="julia --sysimage ~/.julia/sysimages/sys_itensors.so ImagMPS_MixedEstimator.jl $i >> $workdir/out$i &"
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
