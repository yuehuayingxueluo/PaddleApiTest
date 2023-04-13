#run test_vocab_parallel_embedding_paddle_2_card.sh first
python -m paddle.distributed.launch --devices "0,1" test_paddle_incubate.py