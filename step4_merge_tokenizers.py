#!/usr/bin/env python
# -*- coding:utf-8 _*-
"""
@author:quincy qiang
@license: Apache Licence
@file: step4_merge_tokenizers.py
@time: 2023/05/19
@contact: yanqiangmiffy@gamil.com
@software: PyCharm
@description: coding..
"""
from sentencepiece import sentencepiece_model_pb2 as model

''' Merge tokenizer '''
orig_model_path = '/path/to/llama/tokenizer.model'
belle_model_path = '/path/to/belle/belle.model'
orig_m = model.ModelProto()
belle_m = model.ModelProto()
orig_m.ParseFromString(open(orig_model_path, "rb").read())
belle_m.ParseFromString(open(belle_model_path, "rb").read())
print(len(orig_m.pieces), len(belle_m.pieces))
orig_pieces = []
for piece in orig_m.pieces:
    orig_pieces.append(piece.piece)
for piece in belle_m.pieces:
    if piece.piece not in orig_pieces:
        orig_m.pieces.append(piece)
        orig_pieces.append(piece.piece)

print(len(orig_m.pieces))
save_vocab_path = '/path/to/merge_tokenizer/tokenizer.model'
with open(save_vocab_path, 'wb') as f:
    f.write(orig_m.SerializeToString())
