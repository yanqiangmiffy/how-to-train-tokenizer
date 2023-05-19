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
If you don’t have write permission to the global site-packages directory or don’t want to install into it, please try:
"""
# encode: text => id
print(sp.encode_as_pieces(text))
print(sp.encode_as_ids(text))

# decode: id => text
# print(sp.decode_pieces(['▁This', '▁is', '▁a', '▁t', 'est', 'ly']))
# print(sp.decode_ids([209, 31, 9, 375, 586, 34]))
