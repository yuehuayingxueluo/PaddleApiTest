#!/bin/bash
set -ex

for((i=1;i<=16;i++));  
do
    for dtype in _FP32 _FP16 _BFP16;
    do
        cmd="python test_adamw_develop.py TestAdamWDevelopCase"$i$dtype
        $cmd
        echo $cmd
    done
done 