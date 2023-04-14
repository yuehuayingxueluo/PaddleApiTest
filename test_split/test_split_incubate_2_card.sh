#run test_mp_all_reduce_paddle_2_card.sh first
export NVIDIA_TF32_OVERRIDE=0
python -m paddle.distributed.launch --devices "0,1" test_paddle_incubate.py