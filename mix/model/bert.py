from torch import nn
import torch
from joint.model.bert import LMBertForMaskedLM, LMBertModel
import torch.nn.functional as F

def reverse_grad_hook(grad):
    return -grad

from fastNLP.modules import MLP

class JointBertReverseDict(nn.Module):
    def __init__(self, pre_name, train_word2bpes, target_word2bpes, pad_id, num_languages):
        # word2bpes:

        super().__init__()
        # self.model = LMBertForMaskedLM.from_pretrained(pre_name)
        # self.model.bert.add_language_embedding(num_languages)
        # self.model.set_start_end(1, 1+len(train_word2bpes[0]))

        self.model = LMBertModel.from_pretrained(pre_name)
        # self.model.add_language_embedding(num_languages)

        self.max_word_len = len(train_word2bpes[0])
        word2bpes = torch.LongTensor(train_word2bpes).transpose(0, 1).unsqueeze(0)
        self.register_buffer('train_word2bpes', word2bpes)
        word2bpes = torch.LongTensor(target_word2bpes).transpose(0, 1).unsqueeze(0)
        self.register_buffer('target_word2bpes', word2bpes)

        self.lg_fc = MLP([768, 1024, num_languages], activation='relu', dropout=0.3)
        self.pad_id = pad_id
        self.use_train_bpe = False

    def forward(self, input, language_ids):
        # 返回值各是各的？还是可以直接一把soha？
        # language_ids: bsz x max_len
        # sep应该是type=0
        type_ids = input.flip(dims=[1]).cumsum(dim=-1).flip(dims=[1]).eq(0).long()
        attention_mask = input.ne(self.pad_id)

        #  batch_size x max_len x vocab_size
        # bpe_reps, encode_bpes = self.model(input_ids=input, language_ids=language_ids, token_type_ids=type_ids,
        #                                  attention_mask=attention_mask)

        bpe_reps = self.model(input_ids=input, language_ids=language_ids, token_type_ids=type_ids,
                                         attention_mask=attention_mask)[0]
        encode_bpes = bpe_reps
        bpe_reps = torch.matmul(bpe_reps[:, 1:self.max_word_len+1],
                                self.model.embeddings.word_embeddings.weight.t())

        # bsz x max_word_len x word_vocab_size
        if language_ids[0, 0]==language_ids[0, self.max_word_len+3] or self.use_train_bpe:
            word2bpes = self.train_word2bpes.repeat(bpe_reps.size(0), 1, 1)
        else:
            word2bpes = self.target_word2bpes.repeat(bpe_reps.size(0), 1, 1)
        word_scores = bpe_reps.gather(dim=-1, index=word2bpes)  # bsz x max_word_len x word_vocab_size

        word_scores = word_scores.sum(dim=1)

        # 计算language的分数
        if self.training and False:
            lg_bpe_reps = F.dropout(encode_bpes, 0.3, training=self.training)
            lg_bpe_reps.register_hook(reverse_grad_hook)
            lg_scores = self.lg_fc(lg_bpe_reps)
            language_ids = language_ids.masked_fill(attention_mask.eq(0), -100)
            return {'pred': word_scores, 'lg_pred': lg_scores, 'language_ids':language_ids}
        else:
            return {'pred': word_scores, 'language_ids':language_ids}
