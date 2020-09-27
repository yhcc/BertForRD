from fastNLP.io import Pipe
from fastNLP import DataSet
from transformers import BertTokenizer
from collections import defaultdict


class MixUnalignBertPipe(Pipe):
    """
    配合相应的loader, 可以实现bi和mix的情况

    """
    def __init__(self, bert_name, max_word_len=6):
        super().__init__()
        self.bert_name = bert_name
        self.max_word_len = max_word_len
        self.lower = True

    def process(self, data_bundle):
        """
        输入为
            word                definition
            测试                  这是 一个 测试
        包含一个target_words_dict属性
            {
                'lg1': [],
                'lg2': []
            }
        包含的dataset为
            lg1_train
            lg2_train
            lg1_dev
            lg2_dev
            lg1_lg2_test
            lg2_lg1_test

        :param data_bundle:
        :return:
        """
        data_bundle.apply_field(lambda x:x.replace('<concept>', '[MASK]'), field_name='definition', new_field_name='definition')

        tokenizer = BertTokenizer.from_pretrained(self.bert_name)
        tokenizer.do_basic_tokenize = False
        mask_id = tokenizer.convert_tokens_to_ids(['[MASK]'])[0]
        sep_id = tokenizer.convert_tokens_to_ids(['[SEP]'])[0]
        cls_id = tokenizer.convert_tokens_to_ids(['[CLS]'])[0]
        pad_id = tokenizer.convert_tokens_to_ids(['[PAD]'])[0]

        target_words_dict = data_bundle.target_words_dict  # dict
        lgs = list(target_words_dict.keys())

        # 首先搞定词表问题
        target_word2idx = defaultdict(dict)  # word的index
        target_shifts = [0]
        lg_dict = {}
        target_word2bpes = []

        train_word2bpes = []
        train_word2idx = defaultdict(dict)  # word的index
        train_shifts = [0]

        for lg in lgs:
            lg_dict[lg] = len(lg_dict)
            target_words = target_words_dict[lg]
            for word in target_words:
                if word not in target_word2idx[lg]:
                    target_word2idx[lg][word] = len(target_word2idx[lg])
                    bpes = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(word))
                    target_word2bpes.append(bpes)
                    train_word2idx[lg][word] = len(train_word2idx[lg])
                    train_word2bpes.append(bpes)

            # 把train中加入进去
            target_shifts.append(len(target_word2bpes))
            names = [name for name in data_bundle.get_dataset_names() if name.startswith(lg) and 'test' not in name]
            for name in names:  # 部分单语的word可能不被包含在target_words中
                _ds = data_bundle.get_dataset(name)
                for ins in _ds:
                    word = ins['word']
                    if word not in train_word2idx[lg]:
                        bpes = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(word))
                        train_word2idx[lg][word] = len(train_word2idx[lg])
                        train_word2bpes.append(bpes)
            train_shifts.append(len(train_word2bpes))

        # 然后将数据进行index
        for name in data_bundle.get_dataset_names():
            # (1) 需要tokenize word列
            # (2) 需要tokenize definition列
            ds = data_bundle.get_dataset(name)
            if 'test' in name:
                lgs = [name[:2], name[3:5]]
            else:
                lgs = [name[:2], name[:2]]
            new_ds = DataSet()
            lg = lgs[0]
            pre_lg_ids = [lg_dict[lg]] * (self.max_word_len + 2)
            if 'test' not in name:
                word2idx = train_word2idx
                word2bpes = train_word2bpes
                shifts = train_shifts
            else:
                word2idx = target_word2idx
                word2bpes = target_word2bpes
                shifts = target_shifts
            for ins in ds:
                definition = ins['definition'].split()
                words = []
                for word in definition:
                    if self.lower:
                        word = word.lower()
                    word = tokenizer.tokenize(word)
                    word = tokenizer.convert_tokens_to_ids(word)
                    words.extend(word)
                input = [cls_id] + [mask_id] * self.max_word_len + \
                        [sep_id] + words
                input = input[:256]
                input.append(sep_id)
                ins['input'] = input
                lg_ids = pre_lg_ids + [lg_dict[lgs[-1]]] * (len(words)+1)
                ins['language_ids'] = lg_ids[:len(input)]
                # 因为所有word的混合在一起，有一定的shift
                word = ins['word']
                if word not in word2idx[lg]:
                    if 'train' in name:
                        continue
                    else:
                        bpes = [mask_id]*(self.max_word_len+1)
                else:
                    idx = word2idx[lg][word] + shifts[lg_dict[lg]]
                    bpes = word2bpes[idx]
                if len(bpes) <= self.max_word_len:
                    ins['target'] = idx
                    new_ds.append(ins)
                else:
                    if 'train' not in name:
                        ins['target'] = -1
                        new_ds.append(ins)
            data_bundle.set_dataset(new_ds, name)

        for word2bpes in [train_word2bpes, target_word2bpes]:
            for i in range(len(word2bpes)):
                bpes = word2bpes[i]
                bpes = bpes[:self.max_word_len] + [mask_id] * max(0, self.max_word_len - len(bpes))
                word2bpes[i] = bpes

        data_bundle.set_pad_val('input', pad_id)
        data_bundle.set_input('input', 'language_ids')
        data_bundle.set_target('target')
        setattr(data_bundle, 'train_word2bpes', train_word2bpes)
        setattr(data_bundle, 'target_word2bpes', target_word2bpes)
        setattr(data_bundle, 'pad_id', pad_id)
        setattr(data_bundle, 'lg_shift', target_shifts)
        setattr(data_bundle, 'train_lg_shift', train_shifts)
        setattr(data_bundle, 'lg_dict', lg_dict)
        setattr(data_bundle, 'source_word2idx', train_word2idx)
        setattr(data_bundle, 'target_word2idx', target_word2idx)
        return data_bundle

