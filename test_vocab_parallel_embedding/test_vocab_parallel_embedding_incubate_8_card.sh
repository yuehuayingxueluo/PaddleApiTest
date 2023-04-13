#run test_vocab_parallel_embedding_paddle_8_card.sh first
python -m paddle.distributed.launch --devices "0,1,2,3,4,5,6,7" test_paddle_incubate.py

