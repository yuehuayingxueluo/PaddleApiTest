#!/bin/bash
set -ex

card_num=$1
version=$2

export NVIDIA_TF32_OVERRIDE=0

case $card_num in 
    2 ) export CUDA_VISIBLE_DEVICES=0,1 ;;
    4 ) export CUDA_VISIBLE_DEVICES=0,1,2,3 ;;
    8 ) export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 ;;
    * )  
        echo '请输入正确的卡数'  
        exit 
    ;;
esac

#please run "bash test_vocab_parallel_embedding.sh * develop" first

if [ "$version" == 'develop' ]; then
    rm -rf *.npz
    rm -rf log/
    python get_torch_result.py
    python -m paddle.distributed.launch get_and_test_paddle_result.py
elif [ "$version" == 'incubate' ]; then
    python -m paddle.distributed.launch test_paddle_incubate.py
else
   echo "请输入develop或者incubate"
fi