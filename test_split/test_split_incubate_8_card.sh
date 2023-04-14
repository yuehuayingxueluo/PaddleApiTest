#run test_mp_all_reduce_paddle_8_card.sh first
export NVIDIA_TF32_OVERRIDE=0
python -m paddle.distributed.launch --devices "0,1,2,3,4,5,6,7" test_paddle_incubate.py

