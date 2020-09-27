

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
    # tokenizer = BertTokenizer.from_pretrained(os.path.expanduser('~/.fastNLP/embedding/bert-base-cased'))
    # path = os.path.join('../../../data/en', 'target_words.txt')
    # unk_count = 0
    # total = 0
    # for word in read_target_words(path):
    #     tokens = tokenizer.tokenize(word)
    #     if '[UNK]' in tokens:
    #         unk_count += 1
    #     total += 1
    # print(f"{unk_count}/{total}")   # 打印为0/50477

    tokenizer = BertTokenizer.from_pretrained(os.path.expanduser('~/.fastNLP/embedding/bert-chinese-wwm'))
    tokenizer.do_basic_tokenize = True
    path = os.path.join('../../../data/cn', 'target_words.txt')
    unk_count = 0
    total = 0
    for word in read_target_words(path):
        tokens = tokenizer.tokenize(word)
        if '[UNK]' in tokens:
            unk_count += 1
        total += 1
    print(f"{unk_count}/{total}")  # 3614/58491,


if __name__ == '__main__':
    number_recognizied_word()