from fastNLP.io import Pipe
from transformers import XLMRobertaTokenizer
from fastNLP import DataSet
from transformers import BertTokenizer
from collections import defaultdict


class BiAlignedBertPipe(Pipe):
    """
    输出为

        target          input
        int             [cls, xx]

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
        unk_id = tokenizer.convert_tokens_to_ids(['[UNK]'])[0]

        target_words = data_bundle.target_words
        word2bpes = []
        word2idx = {}
        for ins in data_bundle.get_dataset('train'):
            word = ins['word']
            if word not in word2idx:
                word2idx[word] = len(word2idx)
                bpes = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(word))
                word2bpes.append(bpes)
        number_word_in_train = len(word2idx)

        for word in target_words:
            if word not in word2idx:
                word2idx[word] = len(word2idx)
                bpes = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(word))
                word2bpes.append(bpes)

        for name in data_bundle.get_dataset_names():
            # (1) 需要tokenize word列
            # (2) 需要tokenize definition列
            ds = data_bundle.get_dataset(name)
            new_ds = DataSet()
            for ins in ds:
                word = ins['word']
                idx = word2idx[word]
                bpes = word2bpes[idx]
                definition = ins['definition'].split()
                words = []
                for word in definition:
                    if self.lower:
                        word = word.lower()
                    word = tokenizer.tokenize(word)
                    word = tokenizer.convert_tokens_to_ids(word)
                    words.extend(word)
                input = [cls_id] + [mask_id]*self.max_word_len + \
                    [sep_id] + words
                input = input[:256]

                input.append(sep_id)
                ins['input'] = input
                if unk_id in bpes:  # 如果有unk，要么扔掉，要么dev设置为-1
                    if name == 'train':
                        continue
                    else:
                        bpes = [0] * (self.max_word_len + 1)  # 使得设置target为-1
                if len(bpes)<=self.max_word_len:
                    ins['target'] = idx
                    new_ds.append(ins)
                else:
                    if name != 'train':
                        ins['target'] = -1
                        new_ds.append(ins)
            data_bundle.set_dataset(new_ds, name)

        for i in range(len(word2bpes)):
            bpes = word2bpes[i]
            bpes = bpes[:self.max_word_len] + [mask_id]*max(0, self.max_word_len-len(bpes))
            word2bpes[i] = bpes

        data_bundle.set_pad_val('input', pad_id)
        data_bundle.set_input('input')
        data_bundle.set_target('target')
        setattr(data_bundle, 'word2bpes', word2bpes)
        setattr(data_bundle, 'pad_id', pad_id)
        setattr(data_bundle, 'number_word_in_train', number_word_in_train)
        return data_bundle


class JointAlignBertPipe(Pipe):
    def __init__(self, pre_name, max_word_len):
        super().__init__()
        self.pre_name = pre_name
        self.max_word_len = max_word_len
        self.lower = True

    def process(self, data_bundle):
        """
        包含
            {lg1_lg2}_train
            {lg1_lg2}_dev
            {lg1_lg2}_test

        :param data_bundle:
        :return:
        """
        tokenizer = BertTokenizer.from_pretrained(self.pre_name)
        mask_id = tokenizer.convert_tokens_to_ids(['[MASK]'])[0]
        sep_id = tokenizer.convert_tokens_to_ids(['[SEP]'])[0]
        cls_id = tokenizer.convert_tokens_to_ids(['[CLS]'])[0]
        pad_id = tokenizer.convert_tokens_to_ids(['[PAD]'])[0]
        unk_id = tokenizer.convert_tokens_to_ids((['[UNK]']))[0]
        data_bundle.apply_field(lambda x:x.replace('<concept>', '[MASK]'), field_name='definition', new_field_name='definition')

        # 需要处理word2bpes
        word2bpes = []
        word2idx = defaultdict(dict)
        lg_dict = {}
        shifts = [0]
        for name in ['en', 'es', 'fr']:
            lg_dict[name] = len(lg_dict)
            ds_names = [ds_name for ds_name in data_bundle.get_dataset_names() if ds_name.startswith(name)]
            for ds_name in ds_names:  # 部分单语的word可能不被包含在target_words中
                _ds = data_bundle.get_dataset(ds_name)
                for ins in _ds:
                    word = ins['word']
                    if word not in word2idx[lg_dict[name]]:
                        word2idx[lg_dict[name]][word] = len(word2idx[lg_dict[name]])
                        bpes = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(word))
                        word2bpes.append(bpes)

            target_words = getattr(data_bundle, f'{name}_target_words')
            for word in target_words:
                if word not in word2idx[lg_dict[name]]:
                    word2idx[lg_dict[name]][word] = len(word2idx[lg_dict[name]])
                    word2bpes.append(tokenizer.convert_tokens_to_ids(tokenizer.tokenize(word)))
            shifts.append(len(word2bpes))

        #  将数据index
        for name in data_bundle.get_dataset_names():
            # (1) 需要tokenize word列
            # (2) 需要tokenize definition列
            ds = data_bundle.get_dataset(name)
            target_lg = name[:2]
            source_lg = name[3:5]
            new_ds = DataSet()
            pre_lg_ids = [lg_dict[target_lg]]*(self.max_word_len+2)
            for ins in ds:
                word = ins['word']
                idx = word2idx[lg_dict[target_lg]][word] + shifts[lg_dict[target_lg]]
                bpes = word2bpes[idx]
                definition = ins['definition'].split()
                words = []
                for word in definition:
                    if self.lower:
                        word = word.lower()
                    word = tokenizer.tokenize(word)
                    word = tokenizer.convert_tokens_to_ids(word)
                    words.extend(word)
                input = [cls_id] + [mask_id]*self.max_word_len + \
                    [sep_id] + words
                input = input[:256]
                input.append(sep_id)
                ins['input'] = input
                lg_ids = pre_lg_ids + [lg_dict[source_lg]]*(len(words)+1)
                ins['language_ids'] = lg_ids[:len(input)]

                if unk_id in bpes:  # 如果有unk，要么扔掉，要么dev设置为-1
                    if name == 'train':
                        continue
                    else:
                        bpes = [0] * (self.max_word_len + 1)  # 使得设置target为-1
                if len(bpes)<=self.max_word_len:
                    ins['target'] = idx  # 因为所有word的混合在一起，有一定的shift
                    new_ds.append(ins)
                else:
                    if 'train' not in name:
                        ins['target'] = -1
                        new_ds.append(ins)
            data_bundle.set_dataset(new_ds, name)

        for i in range(len(word2bpes)):
            bpes = word2bpes[i]
            bpes = bpes[:self.max_word_len] + [mask_id]*max(0, self.max_word_len-len(bpes))
            word2bpes[i] = bpes

        data_bundle.set_pad_val('input', pad_id)
        data_bundle.set_input('input', 'language_ids')
        data_bundle.set_target('target')
        setattr(data_bundle, 'word2bpes', word2bpes)
        setattr(data_bundle, 'pad_id', pad_id)
        setattr(data_bundle, 'lg_shift', shifts)
        setattr(data_bundle, 'lg_dict', lg_dict)

        return data_bundle

