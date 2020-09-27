"""



"""
from fastNLP.io import  DataBundle, Pipe
from fastNLP import DataSet
from transformers.tokenization_bert import BertTokenizer
from .loader import CnLoader, EnLoader
from transformers import RobertaTokenizer


def _prepare_data_bundle(tokenizer, data_bundle, max_word_len):
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
            definition = []
            for word in ins['definition'].split():
                definition.extend(tokenizer.tokenize(word))
            definition = tokenizer.convert_tokens_to_ids(definition)
            input = [cls_id] + [mask_id] * max_word_len + \
                    [sep_id] + definition
            input = input[:256]
            input.append(sep_id)
            ins['input'] = input
            if unk_id in bpes:
                if name == 'train':
                    continue
                else:
                    bpes = [0] * (max_word_len + 1)  # 使得设置target为-1
            if len(bpes) <= max_word_len:
                ins['target'] = idx
                new_ds.append(ins)
            else:
                if name != 'train':
                    ins['target'] = -1
                    new_ds.append(ins)
        data_bundle.set_dataset(new_ds, name)
    # 中文就不考虑start和middle bpe的区别了
    for i in range(len(word2bpes)):
        bpes = word2bpes[i]
        bpes = bpes[:max_word_len] + [mask_id] * max(0, max_word_len - len(bpes))
        word2bpes[i] = bpes

    data_bundle.set_pad_val('input', pad_id)
    data_bundle.set_input('input')
    data_bundle.set_target('target')
    setattr(data_bundle, 'word2bpes', word2bpes)
    setattr(data_bundle, 'pad_id', pad_id)
    setattr(data_bundle, 'number_word_in_train', number_word_in_train)
    setattr(data_bundle, 'word2idx', word2idx)

    return data_bundle


class CNBertPipe(Pipe):
    """
    由于中文roberta使用的是hit的，所以直接用这个bert load就好了
    """
    def __init__(self, bert_name, max_word_len=6):
        super().__init__()
        self.bert_name = bert_name
        self.max_word_len = max_word_len

    def process(self, data_bundle):
        """
        输入为
            word                definition
            测试                  这是 一个 测试

        :param data_bundle:
        :return:
        """
        tokenizer = BertTokenizer.from_pretrained(self.bert_name)
        tokenizer.do_basic_tokenize = True
        return _prepare_data_bundle(tokenizer, data_bundle, self.max_word_len)

    def process_from_file(self, paths) -> DataBundle:
        data_bundle = CnLoader().load(paths)
        return self.process(data_bundle)


class ENBertPipe(Pipe):
    """
    输出为

        target          input
        int             [cls, xx]
    并包含两个attr
        first_word2bpe: 出现在首位的bpe
        middle_word2bpe: 出现在中间的bpe

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
        tokenizer = BertTokenizer.from_pretrained(self.bert_name)
        tokenizer.do_basic_tokenize = False
        return _prepare_data_bundle(tokenizer, data_bundle, self.max_word_len)

    def process_from_file(self, paths) -> DataBundle:
        data_bundle = EnLoader().load(paths)
        return self.process(data_bundle)


class ENRobertaPipe(Pipe):
    def __init__(self, roberta_name, max_word_len):
        super().__init__()
        self.roberta_name = roberta_name
        self.max_word_len = max_word_len

    def process(self, data_bundle):
        tokenizer = RobertaTokenizer.from_pretrained(self.roberta_name)
        mask_id = tokenizer.convert_tokens_to_ids(['<mask>'])[0]
        sep_id = tokenizer.convert_tokens_to_ids(['</s>'])[0]
        cls_id = tokenizer.convert_tokens_to_ids(['<s>'])[0]
        pad_id = tokenizer.convert_tokens_to_ids(['<pad>'])[0]
        unk_id = tokenizer.convert_tokens_to_ids(['<unk>'])[0]

        target_words = data_bundle.target_words
        word2bpes = []
        word2idx = {}
        for ins in data_bundle.get_dataset('train'):
            word = ins['word']
            if word not in word2idx:
                word2idx[word] = len(word2idx)
                bpes = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(' ' + word))
                word2bpes.append(bpes)

        number_word_in_train = len(word2idx)

        for word in target_words:
            if word not in word2idx:
                word2idx[word] = len(word2idx)
                bpes = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(' ' + word))
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
                    word = tokenizer.tokenize(' ' + word)
                    word = tokenizer.convert_tokens_to_ids(word)
                    words.extend(word)
                input = [cls_id] + [mask_id] * self.max_word_len + \
                        [sep_id]*2 + words
                input = input[:256]
                input.append(sep_id)
                ins['input'] = input
                if unk_id in bpes:
                    if name == 'train':
                        continue
                    else:
                        bpes = [0] * (self.max_word_len+1)  # 使得设置target为-1
                if len(bpes) <= self.max_word_len:
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
        setattr(data_bundle, 'word2idx', word2idx)
        return data_bundle

    def process_from_file(self, paths) -> DataBundle:
        data_bundle = EnLoader().load(paths)
        return self.process(data_bundle)


if __name__ == '__main__':
    path = '../../../data/cn'
    pipe = CNBertPipe('cn')
    data_bundle = pipe.process_from_file(path)
    print(data_bundle)
