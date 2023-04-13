#run test_vocab_parallel_embedding_paddle_4_card.sh first
python -m paddle.distributed.launch --devices "0,1,2,3" test_paddle_incubate.py

