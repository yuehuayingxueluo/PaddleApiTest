#run test_mp_all_reduce_paddle_4_card.sh first
export NVIDIA_TF32_OVERRIDE=0
python -m paddle.distributed.launch --devices "0,1,2,3" test_paddle_incubate.py

