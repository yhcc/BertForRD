
from torch import nn
import torch
from transformers import RobertaForMaskedLM
from transformers import BertForMaskedLM

class RDBertForMaskedLM(BertForMaskedLM):
    def set_start_end(self, start=1, end=5):
        self.start = start
        self.end = end

    def forward(
        self,
        input_ids=None,
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
        r"""
        masked_lm_labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`, defaults to :obj:`None`):
            Labels for computing the masked language modeling loss.
            Indices should be in ``[-100, 0, ..., config.vocab_size]`` (see ``input_ids`` docstring)
            Tokens with indices set to ``-100`` are ignored (masked), the loss is only computed for the tokens with labels
            in ``[0, ..., config.vocab_size]``
        lm_labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`, defaults to :obj:`None`):
            Labels for computing the left-to-right language modeling loss (next word prediction).
            Indices should be in ``[-100, 0, ..., config.vocab_size]`` (see ``input_ids`` docstring)
            Tokens with indices set to ``-100`` are ignored (masked), the loss is only computed for the tokens with labels
            in ``[0, ..., config.vocab_size]``

    Returns:
        :obj:`tuple(torch.FloatTensor)` comprising various elements depending on the configuration (:class:`~transformers.BertConfig`) and inputs:
        masked_lm_loss (`optional`, returned when ``masked_lm_labels`` is provided) ``torch.FloatTensor`` of shape ``(1,)``:
            Masked language modeling loss.
        ltr_lm_loss (:obj:`torch.FloatTensor` of shape :obj:`(1,)`, `optional`, returned when :obj:`lm_labels` is provided):
                Next token prediction loss.
        prediction_scores (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length, config.vocab_size)`)
            Prediction scores of the language modeling head (scores for each vocabulary token before SoftMax).
        hidden_states (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``config.output_hidden_states=True``):
            Tuple of :obj:`torch.FloatTensor` (one for the output of the embeddings + one for the output of each layer)
            of shape :obj:`(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
        attentions (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``config.output_attentions=True``):
            Tuple of :obj:`torch.FloatTensor` (one for each layer) of shape
            :obj:`(batch_size, num_heads, sequence_length, sequence_length)`.

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.

        Examples::

            from transformers import BertTokenizer, BertForMaskedLM
            import torch

            tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
            model = BertForMaskedLM.from_pretrained('bert-base-uncased')

            input_ids = torch.tensor(tokenizer.encode("Hello, my dog is cute", add_special_tokens=True)).unsqueeze(0)  # Batch size 1
            outputs = model(input_ids, masked_lm_labels=input_ids)

            loss, prediction_scores = outputs[:2]

        """

        outputs = self.bert(
            input_ids,
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

        outputs = (prediction_scores,) + outputs[2:]  # Add hidden states and attention if they are here

        return outputs  # (ltr_lm_loss), (masked_lm_loss), prediction_scores, (hidden_states), (attentions)



class CNBertReverseDict(nn.Module):
    def __init__(self, pre_name, word2bpes, pad_id, number_word_in_train):
        super().__init__()
        self.bert_model = RDBertForMaskedLM.from_pretrained(pre_name)
        self.bert_model.set_start_end(1, 1+len(word2bpes[0]))
        self.max_word_len = len(word2bpes[0])
        # 1 x max_len x vocab_size
        word2bpes = torch.LongTensor(word2bpes).transpose(0, 1).unsqueeze(0)
        self.register_buffer('word2bpes', word2bpes)
        self.pad_id = pad_id
        self.number_word_in_train = number_word_in_train

    def forward(self, input):
        """
        每个input的形式cls + mask + sep_id + definition

        :param input: batch_size x max_len
        :return:
        """
        # sep应该是type=0
        type_ids = input.flip(dims=[1]).cumsum(dim=-1).flip(dims=[1]).eq(0).long()
        attention_mask = input.ne(self.pad_id)

        #  batch_size x max_len x vocab_size
        bpe_reps = self.bert_model(input_ids=input, token_type_ids=type_ids,
                                                        attention_mask=attention_mask)[0]
        # bsz x max_word_len x word_vocab_size
        word2bpes = self.word2bpes.repeat(bpe_reps.size(0), 1, 1)
        word_scores = bpe_reps.gather(dim=-1, index=word2bpes)   # bsz x max_word_len x word_vocab_size

        word_scores = word_scores.sum(dim=1)
        if self.training:
            word_scores = word_scores[:, :self.number_word_in_train]

        return {'pred': word_scores}


class ENBertReverseDict(nn.Module):
    def __init__(self, pre_name, word2bpes, pad_id, number_word_in_train):
        super().__init__()
        self.bert_model = RDBertForMaskedLM.from_pretrained(pre_name)
        self.bert_model.set_start_end(1, 1+len(word2bpes[0]))
        self.max_word_len = len(word2bpes[0])
        word2bpes = torch.LongTensor(word2bpes).transpose(0, 1).unsqueeze(0)
        self.register_buffer('word2bpes', word2bpes)
        self.number_word_in_train = number_word_in_train
        self.pad_id = pad_id

    def forward(self, input):
        """
        每个input的形式cls + mask + sep_id + definition

        :param input: batch_size x max_len
        :return:
        """
        # sep应该是type=0
        type_ids = input.flip(dims=[1]).cumsum(dim=-1).flip(dims=[1]).eq(0).long()
        attention_mask = input.ne(self.pad_id)

        #  batch_size x max_len x vocab_size
        bpe_reps = self.bert_model(input_ids=input, token_type_ids=type_ids,
                                                        attention_mask=attention_mask)[0]

        # bsz x max_word_len x word_vocab_size
        word2bpes = self.word2bpes.repeat(bpe_reps.size(0), 1, 1)
        word_scores = bpe_reps.gather(dim=-1, index=word2bpes)   # bsz x max_word_len x word_vocab_size

        word_scores = word_scores.sum(dim=1)
        if self.training:
            word_scores = word_scores[:, :self.number_word_in_train]

        return {'pred': word_scores}


class RDRobertaForMaskedLM(RobertaForMaskedLM):
    def set_start_end(self, start=1, end=5):
        self.start = start
        self.end = end

    def forward(
            self,
            input_ids=None,
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
        outputs = self.roberta(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
        )

        sequence_output = outputs[0]
        prediction_scores = self.lm_head(sequence_output[:, self.start:self.end])

        outputs = (prediction_scores,) + outputs[2:]  # Add hidden states and attention if they are here

        return outputs  # (ltr_lm_loss), (masked_lm_loss), prediction_scores, (hidden_states), (attentions)


class ENRobertaReverseDict(nn.Module):
    def __init__(self, pre_name, word2bpes, pad_id, number_word_in_train):
        super().__init__()
        self.roberta_model = RDRobertaForMaskedLM.from_pretrained(pre_name)
        self.roberta_model.set_start_end(1, end=1+len(word2bpes[0]))
        self.max_word_len = len(word2bpes[0])
        # 1 x 1 x vocab_size
        word2bpes = torch.LongTensor(word2bpes).transpose(0, 1).unsqueeze(0)
        self.register_buffer('word2bpes', word2bpes)
        self.number_word_in_train = number_word_in_train
        self.pad_id = pad_id

    def forward(self, input):
        """
        每个input的形式cls + mask + sep_id + definition

        :param input: batch_size x max_len
        :return:
        """
        # sep应该是type=0
        # type_ids = input.flip(dims=[1]).cumsum(dim=-1).flip(dims=[1]).eq(0).long()
        attention_mask = input.ne(self.pad_id)

        #  batch_size x max_len x vocab_size
        bpe_reps = self.roberta_model(input_ids=input, token_type_ids=None,
                                         attention_mask=attention_mask)[0]

        # bsz x max_word_len x word_vocab_size
        word2bpes = self.word2bpes.repeat(bpe_reps.size(0), 1, 1)
        word_scores = bpe_reps.gather(dim=-1, index=word2bpes)   # bsz x max_word_len x word_vocab_size

        word_scores = word_scores.sum(dim=1)
        if self.training and self.number_word_in_train is not None:
            word_scores = word_scores[:, :self.number_word_in_train]

        return {'pred': word_scores}

