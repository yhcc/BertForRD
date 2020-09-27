

"""
check一下，是否所有的bpe都可以分

"""

def is_all_target_words_bpe_speratable():
    from transformers import BertTokenizer
    import os

    tokenizer = BertTokenizer.from_pretrained(os.path.expanduser('~/.fastNLP/embedding/bert-base-multilingual-uncased'))
    for name in ['en', 'es', 'fr']:
        unk_count = 0
        total = 0
        with open(os.path.join('../../../data/mix', f'{name}.txt'), 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    tokens = tokenizer.tokenize(line)
                    if '[UNK]' in tokens:
                        unk_count+=1
                    total += 1
        print(f"for {name}:{unk_count}/{total}")
    """
    bert-base-multilingual-cased输出为
        for en:12/45566
        for es:6/35798
        for fr:7/44508
    bert-base-multilingual-uncased输出为
        for en:9/45566
        for es:6/35798
        for fr:7/44508
    """


def is_target_word_unique():
    import os

    lower = True
    for name in ['en', 'es', 'fr']:
        total = 0
        counter = {}
        with open(os.path.join('../../../data/mix', f'{name}.txt'), 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    if lower:
                        line = line.lower()
                    counter[line] = counter.get(line, 0) + 1
                    total += 1
        print(f"{name}:unique {len(counter)}/total {total}, max repeat:{max(counter.values())}")
    """
    lower=False 输出为
        en:unique 45566/total 45566
        es:unique 35798/total 35798
        fr:unique 44508/total 44508
    lower=True 输出为
        en:unique 41723/total 45566
        es:unique 33341/total 35798
        fr:unique 37871/total 44508
    """

def is_target_word_contain_all_word():
    import os

    for pair in ['en_fr', 'fr_en', 'en_es', 'es_en']:
        name = pair[:2]
        counter = {}
        with open(os.path.join('../../../data/mix', f'{name}.txt'), 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    line = line.lower()
                    counter[line] = counter.get(line, 0) + 1

        for split in ['{}_dev.csv', '{}_test500.csv', '{}_train500_10.csv']:
            split = split.format(pair)
            not_found = 0
            with open(os.path.join('../../../data/mix', split), 'r', encoding='utf-8') as f:
                for line in f:
                    if line:
                        if line.split('\t')[1] not in counter:
                            not_found += 1
            print(f"{pair}:{split} not found {not_found}")

def read_target_words(path):
    words = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                words.append(line)
    return words

def number_recognizied_word():
    from transformers import BertTokenizer
    import os
    tokenizer = BertTokenizer.from_pretrained(os.path.expanduser('~/.fastNLP/embedding/bert-base-multilingual-cased'))
    for lg in ['en', 'es', 'fr']:
        unk_count = 0
        path = os.path.join('../../../data/mix', f'{lg}.txt')
        lowered_target_words = list(set([word.lower() for word in read_target_words(path)]))
        for word in lowered_target_words:
            tokens = tokenizer.tokenize(word)
            if '[UNK]' in tokens:
                unk_count += 1
        print(f"{lg}:{unk_count}/{len(lowered_target_words)}")
        """
        en:12/41723
        es:6/33341
        fr:7/37871
        """
    return

    from transformers import XLMRobertaTokenizer
    tokenizer = XLMRobertaTokenizer.from_pretrained(os.path.expanduser('~/.fastNLP/embedding/xlm-roberta-base'))
    for lg in ['en', 'es', 'fr']:
        unk_count = 0
        path = os.path.join('../../../data/mix', f'{lg}.txt')
        lowered_target_words = list(set([word.lower() for word in read_target_words(path)]))
        for word in lowered_target_words:
            tokens = tokenizer.tokenize(word)
            if '<unk>' in tokens:
                unk_count += 1
        print(f"{lg}:{unk_count}/{len(lowered_target_words)}")
        """
        en:0/41723
        es:0/33341
        fr:0/37871
        """
if __name__ == '__main__':
    # is_all_target_words_bpe_speratable()
    # is_target_word_unique()
    # is_target_word_contain_all_word()
    number_recognizied_word()

    import torch
    from torch import nn
    from fairseq.models.roberta import XLMRModel, RobertaLMHead
    import os

    # xlmr = XLMRModel.from_pretrained('../../../../release/weights/xlmr.base', checkpoint_file='model.pt')
    #
    # class LMFairseqXLM(nn.Module):
    #     def __init__(self, pre_name):
    #         super().__init__()
    #         self.model = XLMRModel.from_pretrained(pre_name, checkpoint_file='model.pt')
    #         # states = torch.load(os.path.join(pre_name, 'model.pt'), map_location='cpu')
    #         # states = {key[len('decoder.lm_head.'):]:value for key,value in states['model'].items() if key.startswith('decoder.lm_head')}
    #         # self.cls = RobertaLMHead(embed_dim=states['weight'].size(1)
    #         #                          , output_dim=states['bias'].size(0), activation_fn='gelu')
    #         # self.cls.load_state_dict(states)
    #
    #     def set_start_end(self, start=1, end=5):
    #         self.start = start
    #         self.end = end
    #
    #     def forward(self, input):
    #         feats = self.model.extract_features(input)
    #         feats = feats[:, self.start:self.end]
    #         feats = self.cls(feats)
    #         return feats
    #
    # model = LMFairseqXLM('../../../../release/weights/xlmr.base')
    #
    #
    # states = torch.load('../../../../release/weights/xlmr.base/model.pt', map_location=torch.device('cpu'))
    # print(states.keys())