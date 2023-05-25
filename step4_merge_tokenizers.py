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
https://github.com/ymcui/Chinese-LLaMA-Alpaca/blob/main/scripts/merge_tokenizers.py
"""
from sentencepiece import sentencepiece_model_pb2 as model
import os
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"]="python"
from transformers import LlamaTokenizer
from sentencepiece import sentencepiece_model_pb2 as sp_pb2_model
import sentencepiece as spm
import argparse

# ''' Merge tokenizer '''
# https://github.com/LianjiaTech/BELLE/blob/main/train/scripts/merge_tokenizers.py
# orig_model_path = 'llama/tokenizer.model' # llama的分词器：https://github.com/facebookresearch/llama
# belle_model_path = 'open_llama.model'
# orig_m = model.ModelProto()
# belle_m = model.ModelProto()
# orig_m.ParseFromString(open(orig_model_path, "rb").read())
# belle_m.ParseFromString(open(belle_model_path, "rb").read())
# print(len(orig_m.pieces), len(belle_m.pieces))
# orig_pieces = []
# for piece in orig_m.pieces:
#     orig_pieces.append(piece.piece)
# for piece in belle_m.pieces:
#     if piece.piece not in orig_pieces:
#         orig_m.pieces.append(piece)
#         orig_pieces.append(piece.piece)
#
# print(len(orig_m.pieces))
# save_vocab_path = 'merge_tokenizer/tokenizer.model'
# with open(save_vocab_path, 'wb') as f:
#     f.write(orig_m.SerializeToString())

"""
32000 50000
77526
"""



# https://github.com/ymcui/Chinese-LLaMA-Alpaca/blob/main/scripts/merge_tokenizers.py
parser = argparse.ArgumentParser()
parser.add_argument('--llama_tokenizer_dir', default="./llama", type=str, required=False)
parser.add_argument('--chinese_sp_model_file', default='./open_llama.model', type=str,required=False)
args = parser.parse_args()

llama_tokenizer_dir = args.llama_tokenizer_dir
chinese_sp_model_file = args.chinese_sp_model_file

# load
llama_tokenizer = LlamaTokenizer.from_pretrained(llama_tokenizer_dir)
chinese_sp_model = spm.SentencePieceProcessor()
chinese_sp_model.Load(chinese_sp_model_file)

llama_spm = sp_pb2_model.ModelProto()
llama_spm.ParseFromString(llama_tokenizer.sp_model.serialized_model_proto())
chinese_spm = sp_pb2_model.ModelProto()
chinese_spm.ParseFromString(chinese_sp_model.serialized_model_proto())

# print number of tokens
print(len(llama_tokenizer),len(chinese_sp_model))
print(llama_tokenizer.all_special_tokens)
print(llama_tokenizer.all_special_ids)
print(llama_tokenizer.special_tokens_map)

## Add Chinese tokens to LLaMA tokenizer
llama_spm_tokens_set=set(p.piece for p in llama_spm.pieces)
print(len(llama_spm_tokens_set))
print(f"Before:{len(llama_spm_tokens_set)}")
for p in chinese_spm.pieces:
    piece = p.piece
    if piece not in llama_spm_tokens_set:
        new_p = sp_pb2_model.ModelProto().SentencePiece()
        new_p.piece = piece
        new_p.score = 0
        llama_spm.pieces.append(new_p)
print(f"New model pieces: {len(llama_spm.pieces)}")

## Save
output_sp_dir = 'merged_tokenizer_sp'
output_hf_dir = 'merged_tokenizer_hf' # the path to save Chinese-LLaMA tokenizer
os.makedirs(output_sp_dir,exist_ok=True)
with open(output_sp_dir+'/open_llama.model', 'wb') as f:
    f.write(llama_spm.SerializeToString())
tokenizer = LlamaTokenizer(vocab_file=output_sp_dir+'/open_llama.model')

tokenizer.save_pretrained(output_hf_dir)
print(f"Chinese-LLaMA tokenizer has been saved to {output_hf_dir}")


# Test
llama_tokenizer = LlamaTokenizer.from_pretrained(llama_tokenizer_dir)
chinese_llama_tokenizer = LlamaTokenizer.from_pretrained(output_hf_dir)
print(tokenizer.all_special_tokens)
print(tokenizer.all_special_ids)
print(tokenizer.special_tokens_map)
text='''白日依山尽，黄河入海流。欲穷千里目，更上一层楼。
The primary use of LLaMA is research on large language models, including'''
print("Test text:\n",text)
print
print(f"Tokenized by LLaMA tokenizer:{llama_tokenizer.tokenize(text)}")
print(f"Tokenized by Chinese-LLaMA tokenizer:{chinese_llama_tokenizer.tokenize(text)}")