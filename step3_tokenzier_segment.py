#!/usr/bin/env python
# -*- coding:utf-8 _*-
"""
@author:quincy qiang
@license: Apache Licence
@file: step3_tokenzier_segment.py
@time: 2023/05/19
@contact: yanqiangmiffy@gamil.com
@software: PyCharm
@description: coding..
"""
import sentencepiece as spm

# makes segmenter instance and loads the model file (m.model)
sp = spm.SentencePieceProcessor()
sp.load('open_llama.model')

text = """
垃圾分类，一般是指按一定规定或标准将垃圾分类储存、投放和搬运，从而转变成公共资源的一系列活动的总称。
"""
# encode: text => id
print(sp.encode_as_pieces(text))
print(sp.encode_as_ids(text))

# decode: id => text
print(sp.decode_pieces([['▁', '<0x0A>', '垃圾', '分类', ',', '一般', '是指', '按', '一定', '规定', '或', '标准', '将', '垃圾', '分类', '储存', '、', '投放', '和', '搬运', ',', '从而', '转变成', '公共', '资源', '的一系列', '活动', '的总称', '。', '<0x0A>']]))
print(sp.decode_ids([[43478, 14, 6470, 1066, 43475, 544, 1267, 44573, 2333, 1211, 43737, 717, 43661, 6470, 1066, 5485, 43483, 25066, 43501, 25269, 43475, 2038, 21565, 926, 1417, 10007, 419, 32283, 43477, 14]]))
