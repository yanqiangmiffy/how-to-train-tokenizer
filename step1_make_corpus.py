#!/usr/bin/env python
# -*- coding:utf-8 _*-
"""
@author:quincy qiang
@license: Apache Licence
@file: step1_make_corpus.py
@time: 2023/05/19
@contact: yanqiangmiffy@gamil.com
@software: PyCharm
@description: 构建训练语料

下面是将每个wiki本体的标题和描述形成一个文本，然后合并到一个语料文件corpus.txt
"""

import json
import os

from tqdm import tqdm
from zhconv import convert

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
