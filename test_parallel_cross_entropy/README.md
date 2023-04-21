## 运行方法
```
# develop（这里card_num指的是运行的卡数，只能是2或4或8, develop必须在incubate之前运行）
$ bash test_parallel_cross_entropy.sh card_num develop
# incubate（这里card_num指的是对比的develop运行的卡数，只能是2或4或8）
# 首先修改test_paddle_incubate.py中的word_size为card_num数(2或4或8)
$ bash test_parallel_cross_entropy.sh card_num incubate
```