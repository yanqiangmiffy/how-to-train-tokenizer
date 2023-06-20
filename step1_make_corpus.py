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


import glob
from tqdm import  tqdm
corpus=open('data/corpus.txt','w',encoding='utf-8')
cnt=0
for file in glob.glob('data/*.txt'):
    with open(file,'r',encoding='utf-8') as f:
        for line in tqdm(f.readlines()):
            # print(line.strip())
            if len(line.strip())>100:
                corpus.write(line.strip()+'\n')
                cnt+=1
print(cnt)
# 9853042
corpus.close()
