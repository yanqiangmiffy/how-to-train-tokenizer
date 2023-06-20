# how-to-train-tokenizer

>怎么从零到一训练一个LLM分词器

## SentencePiece简介

- SentencePiece 首先将所有输入转换为 unicode 字符。这意味着它不必担心不同的语言、字符或符号，可以以相同的方式处理所有输入；
- 空白也被当作普通符号来处理。Sentencepiece显式地将空白作为基本标记来处理，用一个元符号 “▁”（ U+2581 ）转义空白，这样就可以实现简单地decoding；
- Sentencepiece 可以直接从 raw text 进行训练； 
- 支持 BPE 和 UniLM 训练方法。

## 代码说明
```text
├── data
│     └── corpus.txt 训练语料
├── llama
│     ├── tokenizer_checklist.chk
│     └── tokenizer.model
├── merged_tokenizer_hf 合并结果 hf格式
│     ├── special_tokens_map.json
│     ├── tokenizer_config.json
│     └── tokenizer.model
├── merged_tokenizer_sp
│     └── open_llama.model # 
├── merge_tokenizer
│     └── tokenizer.model
├── open_llama.model 训练的sp模型
├── open_llama.vocab 训练的sp词汇表
├── README.md
├── step0_step0_process_text.py 基于多分数据集准备训练语料
├── step1_make_corpus.py 基于中文Wikipedia数据准备训练语料
├── step2_train_tokenzier.py  训练分词器
├── step3_tokenzier_segment.py 测试训练后的模型，包括编码和解码测试样例
└── step4_merge_tokenizers.py 与原版llama的分词器进行合并，得到hf格式的tokenizer

```
![img.png](data/img.png)
> 中文Wikipedia数据中一共有2521667条数据



## 训练语料统计
- comments2019数据集大小为3730782
- news2016zh数据集大小为18032857
- webText2019zh数据集大小为5705070
- cls数据集大小为396209
- lcsts数据集大小为1481435

> 经过step1_make_corpus.py合并之后，共有9853042数据
## 测试效果
```text
32000 50000
['<s>', '</s>', '<unk>']
[1, 2, 0]
{'bos_token': '<s>', 'eos_token': '</s>', 'unk_token': '<unk>'}
32000
Before:32000
New model pieces: 77526
Chinese-LLaMA tokenizer has been saved to merged_tokenizer_hf
['<s>', '</s>', '<unk>']
[1, 2, 0]
{'bos_token': '<s>', 'eos_token': '</s>', 'unk_token': '<unk>'}
Test text:

```
-例子1
```text
 白日依山尽，黄河入海流。欲穷千里目，更上一层楼。
Tokenized by LLaMA tokenizer:['▁', '白', '日', '<0xE4>', '<0xBE>', '<0x9D>', '山', '<0xE5>', '<0xB0>', '<0xBD>', '，', '黄', '河', '入', '海', '流', '。', '<0xE6>', '<0xAC>', '<0xB2>', '<0xE7>', '<0xA9>', '<0xB7>', '千', '里', '目', '，', '更', '上', '一', '<0xE5>', '<0xB1>', '<0x82>', '<0xE6>', '<0xA5>', '<0xBC>', '。']
Tokenized by Chinese-LLaMA tokenizer:['▁白', '日', '依', '山', '尽', '，', '黄河', '入海', '流', '。', '欲', '穷', '千里', '目', '，', '更', '上一', '层楼', '。']
```
- 例子2
```text
 大模型是指具有非常大的参数数量的人工神经网络模型。 在深度学习领域，大模型通常是指具有数亿到数万亿参数的模型。
Tokenized by LLaMA tokenizer:['▁', '大', '模', '型', '是', '指', '<0xE5>', '<0x85>', '<0xB7>', '有', '非', '常', '大', '的', '参', '数', '数', '量', '的', '人', '工', '神', '经', '网', '<0xE7>', '<0xBB>', '<0x9C>', '模', '型', '。', '▁', '在', '深', '度', '学', '<0xE4>', '<0xB9>', '<0xA0>', '<0xE9>', '<0xA2>', '<0x86>', '<0xE5>', '<0x9F>', '<0x9F>', '，', '大', '模', '型', '通', '常', '是', '指', '<0xE5>', '<0x85>', '<0xB7>', '有', '数', '<0xE4>', '<0xBA>', '<0xBF>', '到', '数', '万', '<0xE4>', '<0xBA>', '<0xBF>', '参', '数', '的', '模', '型', '。']
Tokenized by Chinese-LLaMA tokenizer:['▁大', '模型', '是指', '具有', '非常', '大的', '参数', '数量的', '人工', '神经网络', '模型', '。', '▁在', '深度', '学习', '领域', '，', '大', '模型', '通常是', '指', '具有', '数', '亿', '到', '数万', '亿', '参数', '的模型', '。']
```

## 为什么需要词表扩充
- [问题4：为什么要扩充词表？直接在原版LLaMA上用中文预训练不行吗？](https://github.com/ymcui/Chinese-LLaMA-Alpaca/wiki/%E5%B8%B8%E8%A7%81%E9%97%AE%E9%A2%98#%E9%97%AE%E9%A2%984%E4%B8%BA%E4%BB%80%E4%B9%88%E8%A6%81%E6%89%A9%E5%85%85%E8%AF%8D%E8%A1%A8%E7%9B%B4%E6%8E%A5%E5%9C%A8%E5%8E%9F%E7%89%88llama%E4%B8%8A%E7%94%A8%E4%B8%AD%E6%96%87%E9%A2%84%E8%AE%AD%E7%BB%83%E4%B8%8D%E8%A1%8C%E5%90%97)
> 原版LLaMA模型的词表大小是32K，其主要针对英语进行训练（具体详见LLaMA论文），对多语种支持不是特别理想（可以对比一下多语言经典模型XLM-R的词表大小为250K）。通过初步统计发现，LLaMA词表中仅包含很少的中文字符，所以在切词时会把中文切地更碎，需要多个byte token才能拼成一个完整的汉字，进而导致信息密度降低。比如，在扩展词表后的模型中，单个汉字倾向于被切成1个token，而在原版LLaMA中可能就需要2-3个才能组合成一个汉字，显著降低编解码的效率

- [baichuan-7B 分词](https://github.com/baichuan-inc/baichuan-7B#%E5%88%86%E8%AF%8D)
> 参考百川分词部分介绍，由于目前公开LLaMA模型对中文语料存在解码效率较低的问题，可以提升训练和推理效率
