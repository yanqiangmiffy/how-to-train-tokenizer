#!/usr/bin/env python
# -*- coding:utf-8 _*-
"""
@author:quincy qiang
@license: Apache Licence
@file: step0_process_text.py
@time: 2023/06/19
@contact: yanqiangmiffy@gamil.com
@software: PyCharm
@description: 搜集多个数据集 合并数据集 todo
"""
import glob

from tqdm import tqdm
import json
import json
import os

from tqdm import tqdm
from zhconv import convert

# =====================================================
# CLUECorpusSmall数据集
# =====================================================
def process_clue(dataset_name, input_path, output_file):
    corpus_comments2019 = open(output_file, 'w', encoding='utf-8')
    cnt = 0
    for file in tqdm(glob.glob(input_path)):
        with open(file, 'r', encoding='utf-8') as f:
            for line in f.readlines():
                if len(line.strip()) > 50:
                    # print(line.strip())
                    corpus_comments2019.write(line.strip() + '\n')
                    cnt += 1
    print(f"{dataset_name}数据集大小为{cnt}")

    corpus_comments2019.close()


# corpus_comments2019

# process_clue(
#     dataset_name='comments2019',
#     input_path='/data/searchgpt/data/corpus/CLUECorpusSmall/comments2019/*.txt',
#     output_file='data/corpus_comments2019.txt'
# )
# # news2016zh
# process_clue(
#     dataset_name='news2016zh',
#     input_path='/data/searchgpt/data/corpus/CLUECorpusSmall/news2016zh_corpus/*.txt',
#     output_file='data/corpus_news2016zh.txt'
# )
#
# # news2016zh
# process_clue(
#     dataset_name='webText2019zh',
#     input_path='/data/searchgpt/data/corpus/CLUECorpusSmall/webText2019zh_corpus2/*.txt',
#     output_file='data/corpus_webText2019zh.txt'
# )


# CSL数据集
def process_cls():
    corpus_cls = open('data/corpus_csl.txt', 'w', encoding='utf-8')
    cnt = 0
    with open('/data/searchgpt/data/corpus/CSL/csl_camera_readly.tsv', 'r', encoding='utf-8') as f:
        for line in tqdm(f.readlines()):
            data = ' '.join(line.strip().split('\t'))
            corpus_cls.write(data + '\n')
            cnt += 1
    print(f"cls数据集大小为{cnt}")
    # cls数据集大小为396209
    corpus_cls.close()

# process_cls()
# LCSTS_new数据集

def process_lcsts():
    corpus_lcsts = open('data/corpus_lcsts.txt', 'w', encoding='utf-8')
    cnt=0
    with open('/data/searchgpt/data/corpus/LCSTS_new/train.json','r',encoding='utf-8') as f:
        for line in f.readlines():
            data=json.loads(line.strip())
            corpus_lcsts.write(data['summary']+' '+data['content']+'\n')
            cnt += 1

    with open('/data/searchgpt/data/corpus/LCSTS_new/dev.json','r',encoding='utf-8') as f:
        for line in f.readlines():
            data=json.loads(line.strip())
            corpus_lcsts.write(data['summary']+' '+data['content']+'\n')
            cnt += 1
    print(f"lcsts数据集大小为{cnt}")
    corpus_lcsts.close()
process_lcsts()
# moviedata-10m数据集

# other数据集



# 中文wiki下载地址
# https://dumps.wikimedia.org/zhwiki/
# wikiextractor -o ./zhwiki-20230401-b 100M--json--processes 4 ./zhwiki-20230401-pages-articles.xml.bz2

basedir = 'cache/zh_wikipedia/zhwiki-20230401/AA'
corpus_file = open('cache/zh_wikipedia/corpus.txt', 'w', encoding='utf-8')
cnt = 0
for wiki_doc in tqdm(os.listdir(basedir)):
    with open(os.path.join(basedir, wiki_doc), 'r', encoding='utf-8') as f:
        for line in tqdm(f, leave=False, desc=""):
            # print(line)
            data = json.loads(line.strip())
            data['title'] = convert(data['title'], 'zh-cn')
            data['text'] = convert(data['text'], 'zh-cn')
            # print(data)
            text = data['title'] + ' ' + data['text']
            corpus_file.write(''.join(text.split('\n')) + '\n')
            cnt += 1
print("文档个数：{}".format(cnt))
# 文档个数：2521667
corpus_file.close()
