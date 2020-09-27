
"""
需要支持aligned版本的MTL，所以需要添加language_embedding

"""

from transformers import BertModel, BertForMaskedLM
from mono.model.bert import RDBertForMaskedLM
from torch import nn
import torch
import numpy as np
from fastNLP.modules import MLP
import torch.nn.functional as F
def reverse_grad_hook(grad):
    return -grad


class BiBertReverseDict(nn.Module):
    def __init__(self, pre_name, word2bpes, pad_id, number_word_in_train=None):
        super().__init__()
        self.model = BertModel.from_pretrained(pre_name)
        # self.model = RDBertForMaskedLM.from_pretrained(pre_name)
        # self.model.set_start_end(1, 1+len(word2bpes[0]))
        self.max_word_len = len(word2bpes[0])
        word2bpes = torch.LongTensor(word2bpes).transpose(0, 1).unsqueeze(0)
        self.register_buffer('word2bpes', word2bpes)
        self.pad_id = pad_id
        self.number_word_in_train = number_word_in_train

    def forward(self, input):
        type_ids = input.flip(dims=[1]).cumsum(dim=-1).flip(dims=[1]).eq(0).long()
        attention_mask = input.ne(self.pad_id)

        #  batch_size x max_len x vocab_size
        bpe_reps = self.model(input_ids=input, token_type_ids=type_ids,
                                         attention_mask=attention_mask)[0]
        bpe_reps = torch.matmul(bpe_reps[:, 1:self.max_word_len+1],
                             self.model.embeddings.word_embeddings.weight.t())

        # bsz x max_word_len x word_vocab_size
        word2bpes = self.word2bpes.repeat(bpe_reps.size(0), 1, 1)
        word_scores = bpe_reps.gather(dim=-1, index=word2bpes)   # bsz x max_word_len x word_vocab_size

        if self.training and self.number_word_in_train:
            word_scores = word_scores[:, :self.number_word_in_train]

        word_scores = word_scores.sum(dim=1)

        return {'pred':word_scores}


class JointBertReverseDict(nn.Module):
    def __init__(self, pre_name, word2bpes, pad_id, num_languages):
        # word2bpes:

        super().__init__()
        self.model = LMBertForMaskedLM.from_pretrained(pre_name)
        self.model.bert.add_language_embedding(num_languages)
        self.model.set_start_end(1, 1+len(word2bpes[0]))

        # self.model = LMBertModel.from_pretrained(pre_name)
        # self.model.add_language_embedding(num_languages)

        self.max_word_len = len(word2bpes[0])
        word2bpes = torch.LongTensor(word2bpes).transpose(0, 1).unsqueeze(0)
        self.register_buffer('word2bpes', word2bpes)

        self.lg_fc = MLP([768, 1024, num_languages], activation='relu', dropout=0.3)
        self.pad_id = pad_id

    def forward(self, input, language_ids):
        # 返回值各是各的？还是可以直接一把soha？
        # language_ids: bsz x max_len
        # sep应该是type=0
        type_ids = input.flip(dims=[1]).cumsum(dim=-1).flip(dims=[1]).eq(0).long()
        attention_mask = input.ne(self.pad_id)

        #  batch_size x max_len x vocab_size
        bpe_reps, encode_bpes = self.model(input_ids=input, language_ids=language_ids, token_type_ids=type_ids,
                                         attention_mask=attention_mask)

        # bpe_reps = self.model(input_ids=input, language_ids=language_ids, token_type_ids=type_ids,
        #                                  attention_mask=attention_mask)[0]
        # encode_bpes = bpe_reps
        # bpe_reps = torch.matmul(bpe_reps[:, 1:self.max_word_len+1],
        #                         self.model.embeddings.word_embeddings.weight.t())

        # bsz x max_word_len x word_vocab_size
        word2bpes = self.word2bpes.repeat(bpe_reps.size(0), 1, 1)
        word_scores = bpe_reps.gather(dim=-1, index=word2bpes)  # bsz x max_word_len x word_vocab_size

        word_scores = word_scores.sum(dim=1)

        # 计算language的分数
        if self.training:
            lg_bpe_reps = F.dropout(encode_bpes, 0.3, training=self.training)
            lg_bpe_reps.register_hook(reverse_grad_hook)
            lg_scores = self.lg_fc(lg_bpe_reps)
            language_ids = language_ids.masked_fill(attention_mask.eq(0), -100)
            return {'pred': word_scores, 'lg_pred': lg_scores, 'language_ids':language_ids}
        else:
            return {'pred': word_scores}


class LMBertForMaskedLM(BertForMaskedLM):
    def __init__(self, config):
        super().__init__(config)

        self.bert = LMBertModel(config)
        self.init_weights()

    def set_start_end(self, start=1, end=5):
        self.start = start
        self.end = end

    def forward(
        self,
        input_ids=None,
        language_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        masked_lm_labels=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        lm_labels=None,
    ):
        outputs = self.bert(
            input_ids,
            language_ids=language_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
        )

        sequence_output = outputs[0]
        prediction_scores = self.cls(sequence_output[:, self.start:self.end])

        outputs = (prediction_scores, sequence_output)  # Add hidden states and attention if they are here

        return outputs  # (ltr_lm_loss), (masked_lm_loss), prediction_scores, (hidden_states), (attentions)


class LMBertModel(BertModel):
    def add_language_embedding(self, num_language):
        self.num_language = num_language
        self.language_embedding = nn.Embedding(
            num_embeddings=self.num_language, embedding_dim=self.config.hidden_size)
        nn.init.uniform_(self.language_embedding.weight.data, a=-np.sqrt(3 / self.language_embedding.weight.data.size(1)),
                         b=np.sqrt(3 / self.language_embedding.weight.data.size(1)))
        # nn.init.zeros_(self.language_embedding.weight.data)

    def forward(
        self,
        input_ids=None,
        language_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
    ):
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            input_shape = input_ids.size()
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        device = input_ids.device if input_ids is not None else inputs_embeds.device

        if attention_mask is None:
            attention_mask = torch.ones(input_shape, device=device)
        if token_type_ids is None:
            token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=device)

        # We can provide a self-attention mask of dimensions [batch_size, from_seq_length, to_seq_length]
        # ourselves in which case we just need to make it broadcastable to all heads.
        extended_attention_mask: torch.Tensor = self.get_extended_attention_mask(
            attention_mask, input_shape, self.device
        )

        # If a 2D ou 3D attention mask is provided for the cross-attention
        # we need to make broadcastabe to [batch_size, num_heads, seq_length, seq_length]
        if self.config.is_decoder and encoder_hidden_states is not None:
            encoder_batch_size, encoder_sequence_length, _ = encoder_hidden_states.size()
            encoder_hidden_shape = (encoder_batch_size, encoder_sequence_length)
            if encoder_attention_mask is None:
                encoder_attention_mask = torch.ones(encoder_hidden_shape, device=device)
            encoder_extended_attention_mask = self.invert_attention_mask(encoder_attention_mask)
        else:
            encoder_extended_attention_mask = None

        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape bsz x n_heads x N x N
        # input head_mask has shape [num_heads] or [num_hidden_layers x num_heads]
        # and head_mask is converted to shape [num_hidden_layers x batch x num_heads x seq_length x seq_length]
        head_mask = self.get_head_mask(head_mask, self.config.num_hidden_layers)

        embedding_output = self.embeddings(
            input_ids=input_ids, position_ids=position_ids, token_type_ids=token_type_ids, inputs_embeds=inputs_embeds
        )
        if hasattr(self, 'language_embedding'):
            language = self.language_embedding(language_ids)
            embedding_output = embedding_output + language
        encoder_outputs = self.encoder(
            embedding_output,
            attention_mask=extended_attention_mask,
            head_mask=head_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_extended_attention_mask,
        )
        sequence_output = encoder_outputs[0]
        pooled_output = self.pooler(sequence_output)

        outputs = (sequence_output, pooled_output,) + encoder_outputs[
            1:
        ]  # add hidden_states and attentions if they are here
        return outputs  # sequence_output, pooled_output, (hidden_states), (attentions)

