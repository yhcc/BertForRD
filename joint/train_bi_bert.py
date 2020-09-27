import sys
sys.path.append('../')
import os
if 'p' in os.environ:
    os.environ['CUDA_VISIBLE_DEVICES']=os.environ['p']

import torch
from torch import optim
from fastNLP import Trainer, CrossEntropyLoss
from fastNLP import BucketSampler, cache_results, WarmupCallback, GradientClipCallback
from joint.data.pipe import BiAlignedBertPipe
from joint.data.loader import BiAlignLoader
# from V1.model.bert import ENBertReverseDict
from joint.model.bert import BiBertReverseDict
import fitlog
from joint.model.metrics import BiMetric
from joint.model.callback import FitlogCallback
from joint.data.utils import clip_max_length
# fitlog.debug()
fitlog.set_log_dir('logs')
fitlog.add_hyper_in_file(__file__)
fitlog.add_other(name='note', value='use embedding')

paths = '../data/mix'

#######hyper
pair = 'es_en'
model_name = 'bert'
max_word_len = 5
lr = 1e-4
batch_size = 80
n_epochs = 20
#######hyper
pre_name = 'bert-base-multilingual-cased'
# transformers中bert-base-multilingual-cased


@cache_results('caches/{}_{}_{}.pkl'.format(pair, pre_name.split('/')[-1], max_word_len), _refresh=False)
def get_data():
    data_bundle = BiAlignLoader(pair).load(paths)
    data_bundle = BiAlignedBertPipe(pre_name, max_word_len).process(data_bundle)
    return data_bundle

data_bundle = get_data()
print(data_bundle)
word2bpes = data_bundle.word2bpes
print(f"In total {len(word2bpes)} target words")
pad_id = data_bundle.pad_id

model = BiBertReverseDict(pre_name, word2bpes, pad_id=pad_id,
                          number_word_in_train=data_bundle.number_word_in_train)

if torch.cuda.is_available():
    model.cuda()

optimizer = optim.AdamW(model.parameters(), lr=lr)

data = {}
data['test'] = data_bundle.get_dataset('test')

callbacks = [GradientClipCallback(clip_type='value'), WarmupCallback(warmup=0.1, schedule='linear')]
callbacks.append(FitlogCallback(data=data, verbose=1))
train_data = data_bundle.get_dataset('train')
train_data.add_seq_len('input')

# from collections import Counter
# print(Counter(train_data.get_field('seq_len').content))
# exit(0)

sampler = BucketSampler()
clip_max_length(train_data, data_bundle, max_sent_len=50)

trainer = Trainer(train_data=train_data, model=model,
                  optimizer=optimizer, loss=CrossEntropyLoss(),
                 batch_size=batch_size, sampler=sampler, drop_last=False, update_every=1,
                 num_workers=1, n_epochs=n_epochs, print_every=5,
                 dev_data=data_bundle.get_dataset('dev'), metrics=BiMetric(),
                 metric_key='t10',
                 validate_every=-1, save_path=None, use_tqdm=True, device=None,
                 callbacks=callbacks, check_code_level=0)
trainer.train()
