#!/usr/bin/env python
# -*- coding:utf-8 _*-
"""
@author:quincy qiang
@license: Apache Licence
@file: step5_test_tokenizer_efficiency.py
@time: 2023/07/12
@contact: yanqiangmiffy@gamil.com
@software: PyCharm
@description: coding..
"""
import pandas as pd
from tqdm import tqdm
from transformers import LlamaTokenizer

tokenizer = LlamaTokenizer.from_pretrained('merged_tokenizer_hf_40k')
llama_tokenizer = LlamaTokenizer.from_pretrained('llama')

print(tokenizer)

num_tokens = []
num_ids = []
num_ids_llama = []
with open('data/corpus_csl.txt', 'r', encoding='utf-8') as f:
    for line in tqdm(f.readlines()):
        line = line.strip()
        encode = tokenizer(line)
        encode_llama = llama_tokenizer(line)
        # print(encode)
        # print(len(line),len(encode['input_ids']))
        num_tokens.append(len(line))
        num_ids.append(len(encode['input_ids']))
        num_ids_llama.append(len(encode_llama['input_ids']))

df = pd.DataFrame({'num_tokens': num_tokens, 'num_ids': num_ids,'num_ids_llama':num_ids_llama})
df = df.sort_values(by=["num_tokens"], ascending=True)
df.to_csv('merged_tokenizer_hf_40k_efficiency.csv', index=False)
