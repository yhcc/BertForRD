"""
这个文件主要包含加载数据的文件，需要支持以下的几种setting

(1) 从cn folder
    desc.json
    dev.json
    question.json
    seen_test.json
    target_words.txt
    train.json
    unseen_test.json
(2) 从en folder
    desc.json
    dev.json
    seen_test.json
    target_words.txt
    train.json
    unseen_test.json
(3) 从mix中读取数据
    (1) 有单语言的, lg为en, es, fr
        {lg}_dev.csv
        {lg}_test500.csv
        {lg}_train500_10.csv
    (2) 有多语言的
        {lg1}_{lg2}_dev.csv
        {lg1}_{lg2}_test500.csv
        {lg1}_{lg2}_train500_10.csv

paper里面一共有如下的setting
    mono-lingual
        (1) cn
        (2) en
    multi-lingual
        (1) bi-lingual  en_es, en_fr, fr_en, es_en
        (2) tri-lingual 所有的一起训练en_es, en_fr, fr_en, es_en
    unaligned
        (1) bi-lingual  en, es -> en_es, es_en
                        en, fr -> en_fr, fr_en
        (2) tri-lingual en, es ,fr -> en_es, es_en, en_fr, fr_en

另外还需要的分析实验有
    (1) 在unaligned的setting下，见过的definition在reverse dictionary中的performance
        逐渐将测试中遇到的词从train中删除，然后看performance。
    (2) 需要做的case study为
        (2.1) 单语言状态下，对多义词的performance对比
            找到测试中所有的多义词，然后看看他的performance ？做一个中文、英文的case展示
        (2.2) 无对齐语料的情况下的多义词performance
            找到测试中的多义词，然后通过单语找到自己语言的词，再通过字典或者vector的方式找到对应的另一种语言的词
    (3) 单语言状态下句子长度与performance的变化
        现在的一个问题是没有这个数据？

"""

import os
import json
from fastNLP.io import Loader, DataBundle
from fastNLP import Instance, DataSet


class CnLoader(Loader):
    """
    会读取folder下面的
        desc.json
        dev.json
        question.json
        seen_test.json
        target_words.txt
        train.json
        unseen_test.json
    读入的数据
        word            definition
        测试            这是 一个 测试
    同时返回的DataBunlde还包含一个属性
        target_words: [word1, word2, ...]

    """
    def __init__(self):
        super().__init__()

    def load(self, folder):
        data_bundle = DataBundle()
        for name in ['desc.json', 'dev.json', 'question.json', 'seen_test.json', 'train.json',
                     'unseen_test.json']:
            path = os.path.join(folder, name)
            with open(path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                dataset = DataSet()
                for d in data:
                    word = d['word']
                    definition = d['definition']
                    ins = Instance(word=word, definition=definition)
                    dataset.append(ins)
                data_bundle.set_dataset(dataset, name=name.split('.')[0])
        # 读取target_words
        words = []
        with open(os.path.join(folder, 'target_words.txt'), 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    words.append(line)
        setattr(data_bundle, 'target_words', words)
        return data_bundle


class EnLoader(Loader):
    """
    读取folder下
        desc.json
        dev.json
        seen_test.json
        target_words.txt
        train.json
        unseen_test.json
    读取的数据
        word                definition
        forget              cannot remember ...
    包含一个target_words属性
        [w1, w2, ...]

    """
    def __init__(self):
        super().__init__()

    def load(self, folder):
        data_bundle = DataBundle()
        for name in ['desc.json', 'dev.json', 'seen.json', 'train.json', 'unseen.json']:
            path = os.path.join(folder, name)
            dataset = DataSet()
            with open(path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                for d in data:
                    word = d['word'].lower()
                    definition = d['definitions'].lower()
                    ins = Instance(word=word, definition=definition)
                    dataset.append(ins)
                data_bundle.set_dataset(dataset, name=name.split('.')[0])
        words = []
        with open(os.path.join(folder, 'target_words.txt'), 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    words.append(line)
        setattr(data_bundle, 'target_words', words)
        return data_bundle

