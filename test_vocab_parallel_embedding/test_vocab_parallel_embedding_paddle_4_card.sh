#run first
rm -rf *.npz
python get_torch_result.py
python -m paddle.distributed.launch --devices "0,1,2,3" get_and_test_paddle_result.py