#run first
export NVIDIA_TF32_OVERRIDE=0
rm -rf *.npz
python get_torch_result.py
python -m paddle.distributed.launch --devices "0,1" get_and_test_paddle_result.py

