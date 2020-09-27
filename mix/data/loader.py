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


def read_dataset(path, lower, word_idx=1, def_idx=-1):
    ds = DataSet()
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                parts = line.split('\t')
                if lower:
                    ins = Instance(word=parts[word_idx].lower(), definition=parts[def_idx].lower())
                else:
                    ins = Instance(word=parts[word_idx], definition=parts[def_idx])
                ds.append(ins)
    return ds


class BiUnAlignLoader(Loader):
    """
    非监督的双语

    """
    def __init__(self, lg1_lg2, lower=True):
        # lg1是target_language
        super().__init__()
        assert lg1_lg2 in ('en_es', 'en_fr')
        self.lg1_lg2 = lg1_lg2
        self.lower = lower

    def load(self, folder):
        #  首先读取两个单语文件
        lg1, lg2 = self.lg1_lg2.split('_')
        fns ={
            'dev':'{}_dev.csv',
            # 'test':'{}_test500.csv'.format(self.lg1_lg2),
            'train': '{}_train500_10.csv'
        }
        data_bundle = DataBundle()
        words = {}
        for lg in [lg1, lg2]:
            for name, fn in fns.items():
                path = os.path.join(folder, fn.format(lg))
                ds = read_dataset(path, self.lower, 0)
                data_bundle.set_dataset(ds, name=f'{lg}_{name}')
            target_words = {}
            with open(os.path.join(folder, '{}.txt'.format(lg)), encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if line:
                        if self.lower:
                            line = line.lower()
                        target_words[line] = 1
            target_words = list(target_words.keys())
            words[lg] = target_words
        setattr(data_bundle, 'target_words_dict', words)

        # 读取bi的测试数据
        bi1 = f'{lg1}_{lg2}'
        bi2 = f'{lg2}_{lg1}'
        for bi in [bi1, bi2]:
            path = os.path.join(folder, '{}_test500.csv'.format(bi))
            ds = read_dataset(path, self.lower, 1)
            # ds = DataSet()
            # with open(path, 'r', encoding='utf-8') as f:
            #     for line in f:
            #         line = line.strip()
            #         if line:
            #             parts = line.split('\t')
            #             ins = Instance(word=parts[1].lower(), definition=parts[-1])
            #             ds.append(ins)
            data_bundle.set_dataset(ds, '{}_test'.format(bi))

        return data_bundle


class MixUnAlignLoader(Loader):
    #
    def __init__(self, lower=True):
        super().__init__()
        self.lower = lower

    def load(self, folder):
        data_bundle = DataBundle()
        fns ={
            'dev':'{}_dev.csv',
            # 'test':'{}_test500.csv'.format(self.lg1_lg2),
            'train': '{}_train500_10.csv'
        }
        data_bundle = DataBundle()
        words = {}
        for lg in ['en', 'es', 'fr']:
            for name, fn in fns.items():
                path = os.path.join(folder, fn.format(lg))
                ds = read_dataset(path, self.lower, 0)
                data_bundle.set_dataset(ds, name=f'{lg}_{name}')
            target_words = {}
            with open(os.path.join(folder, '{}.txt'.format(lg)), encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if line:
                        if self.lower:
                            line = line.lower()
                        target_words[line] = 1
            target_words = list(target_words.keys())
            words[lg] = target_words
        setattr(data_bundle, 'target_words_dict', words)

        for bi in ['en_fr', 'fr_en', 'en_es', 'es_en']:
            path = os.path.join(folder, '{}_test500.csv'.format(bi))
            ds = read_dataset(path, self.lower, 1)
            data_bundle.set_dataset(ds, '{}_test'.format(bi))

        return data_bundle


