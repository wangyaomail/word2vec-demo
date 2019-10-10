# Word2Vec demo

参考这两个项目完成：
- https://github.com/tmikolov/word2vec/blob/master/word2vec.c
- https://github.com/NLPchina/Word2VEC_java

主要是以上的实现中，有一些工程上的东西，不太便于理解，所以将一些工程上作用比较大但算法作用有限的内容省略，并按照最没有工程化的方法对代码进行实现。

## +glove
增加glove的实现，参考官方实现完成：https://github.com/stanfordnlp/GloVe

删掉工程性的部分，只保留算法性的代码

## +rnn
参考这个代码实现：https://github.com/garstka/char-rnn-java
删掉窗口机制，保留单层和多层的训练方法
增加封装BasicRNN，其测试代码在Seq2SeqDemo2.java

## +lstm
参考这个代码实现：https://github.com/lipiji/JRNN


## +seq2seq
参考这两个项目：
- keras的seq2seq实现：https://github.com/farizrahman4u/seq2seq
- TensorFlow的seq2seq实现：https://github.com/google/seq2seq
当前版本代码中Seq2SeqDemo.java还有问题没有解决，Seq2SeqDemo2.java可以直接运行

