#run first
rm -rf *.npz
python prepare_data.py
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m torch.distributed.launch --nproc_per_node=8 get_torch_result.py
python -m paddle.distributed.launch --devices "0,1,2,3,4,5,6,7" get_and_test_paddle_result.py

