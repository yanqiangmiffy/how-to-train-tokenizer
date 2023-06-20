#!/usr/bin/env python
# -*- coding:utf-8 _*-
"""
@author:quincy qiang
@license: Apache Licence
@file: train_tokenzier.py
@time: 2023/05/19
@contact: yanqiangmiffy@gamil.com
@software: PyCharm
@description: coding..
"""

import time

import sentencepiece as spm

"""
sentencepiece 参数
trainer_spec {
  input: data/corpus.txt
  input_format: #
  model_prefix: open_llama # 模型输出路径
  model_type: BPE # 模型类型 bpe、char、word、unigram(gram)
  vocab_size: 50000 # 词汇表大小，数量越大训练越慢，太小（<4000）可能训练不了
  self_test_sample_size: 0
  character_coverage: 0.9995 # 模型中覆盖的字符数
  input_sentence_size: 0
  shuffle_input_sentence: 0
  seed_sentencepiece_size: 1000000 # 
  shrinking_factor: 0.75
  max_sentence_length: 16384 # 最大句子长度，默认是4192，长度按照字节计算，一个中文代表长度为2
  num_threads: 16 # 进程个数
  num_sub_iterations: 2
  max_sentencepiece_length: 16
  split_by_unicode_script: 1
  split_by_number: 1
  split_by_whitespace: 1
  split_digits: 1
  pretokenization_delimiter: 
  treat_whitespace_as_suffix: 0
  allow_whitespace_only_pieces: 1
  required_chars: 
  byte_fallback: 1
  vocabulary_output_piece_score: 1
  train_extremely_large_corpus: 1
  hard_vocab_limit: 1
  use_all_vocab: 0 # 使用
  unk_id: 0
  bos_id: 1
  eos_id: 2
  pad_id: 3
  unk_piece: <unk>
  bos_piece: <s>
  eos_piece: </s>
  pad_piece: <pad>
  unk_surface:  ⁇ 
  enable_differential_privacy: 0
  differential_privacy_noise_level: 0
  differential_privacy_clipping_threshold: 0
}
normalizer_spec {
  name: nfkc
  add_dummy_prefix: 1
  remove_extra_whitespaces: 0
  escape_whitespaces: 1
  normalization_rule_tsv: 
}
"""
start_time = time.time()
spm.SentencePieceTrainer.train(
    input='data/corpus.txt',  # 输入文件
    model_prefix='gogpt',  # 模型前缀
    shuffle_input_sentence=False,  # 是否打乱句子
    train_extremely_large_corpus=True,
    # hyperparameters of tokenizer
    max_sentence_length=16384,  # 句子最大长度
    pad_id=3,
    model_type="BPE",
    vocab_size=60000,
    split_digits=True,
    split_by_unicode_script=True,
    byte_fallback=True,
    allow_whitespace_only_pieces=True,
    remove_extra_whitespaces=False,
    normalization_rule_name="nfkc",
)

end_time = time.time()
print(end_time - start_time)
