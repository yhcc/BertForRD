import sys
sys.path.append('../')
import os
if 'p' in os.environ:
    os.environ['CUDA_VISIBLE_DEVICES']=os.environ['p']

# os.environ['CUDA_VISIBLE_DEVICES'] = '1'

import torch
from torch import optim
from fastNLP import Trainer
from fastNLP import BucketSampler, cache_results, WarmupCallback, GradientClipCallback
from joint.data.pipe import JointAlignBertPipe
from joint.data.loader import JointAlignLoader
from joint.model.bert import JointBertReverseDict
import fitlog
from joint.model.metrics import JointMetric, SummaryMetric
from fastNLP import DataSet, Tester
from joint.model.callback import FitlogCallback
from mix.model.loss import MixLoss
from joint.data.utils import clip_max_length
# fitlog.debug()
fitlog.set_log_dir('logs')
fitlog.add_hyper_in_file(__file__)
fitlog.add_other(name='note', value='lm head')

paths = '../data/mix'
#######hyper
model_name = 'bert'
max_word_len = 5
batch_size = 80
n_epochs = 20
lg_lambda = 0.0
lr = 5e-5
#######hyper
pre_name = 'bert-base-multilingual-cased'
# transformers中xlm-roberta-base


@cache_results('caches/joint_{}_{}.pkl'.format(pre_name.split('/')[-1], max_word_len), _refresh=False)
def get_data():
    data_bundle = JointAlignLoader().load(paths)
    data_bundle = JointAlignBertPipe(pre_name, max_word_len).process(data_bundle)
    return data_bundle

data_bundle = get_data()
print(data_bundle)
word2bpes = data_bundle.word2bpes
pad_id = data_bundle.pad_id
print(f"In total {len(word2bpes)} target words")
lg_dict = getattr(data_bundle, 'lg_dict')
lg_shifts = getattr(data_bundle, 'lg_shift')

model = JointBertReverseDict(pre_name, word2bpes, pad_id=pad_id, num_languages=3)

summary_ms = []
train_data = DataSet()
testers = {}

for name in ['en_fr', 'en_es', 'fr_en', 'es_en']:
    for ins in data_bundle.get_dataset('{}_train'.format(name)):
        train_data.append(ins)
    # dev
    data = data_bundle.get_dataset(f'{name}_dev')
    _metric = JointMetric(lg_shifts[lg_dict[name[:2]]], lg_shifts[lg_dict[name[:2]]+1])
    tester = Tester(data, model, _metric, batch_size=120, num_workers=1, device=None, verbose=1, use_tqdm=True)
    testers[f'{name}_dev'] = tester
    summary_ms.append(_metric)
    # test
    data = data_bundle.get_dataset(f'{name}_test')
    _metric = JointMetric(lg_shifts[lg_dict[name[:2]]], lg_shifts[lg_dict[name[:2]]+1])
    tester = Tester(data, model, _metric, batch_size=120, num_workers=1, device=None, verbose=1, use_tqdm=True)
    testers[f'{name}_test'] = tester
metric = SummaryMetric(summary_ms)

clip_max_length(train_data, data_bundle, max_sent_len=50)
train_data.add_seq_len('input')
train_data.set_input('input', 'language_ids')
train_data.set_target('target')
train_data.set_pad_val('input', pad_id)

if torch.cuda.is_available():
    model.cuda()

optimizer = optim.AdamW(model.parameters(), lr=lr)

callbacks = [GradientClipCallback(clip_type='value'), WarmupCallback(warmup=0.01, schedule='linear')]
callbacks.append(FitlogCallback(tester=testers, verbose=1))
sampler = BucketSampler()

dev_data = data_bundle.get_dataset('en_fr_dev')[:2]

trainer = Trainer(train_data=train_data, model=model,
                  optimizer=optimizer, loss=MixLoss(lg_shifts, lg_lambda=lg_lambda),
                 batch_size=batch_size, sampler=sampler, drop_last=False, update_every=1,
                 num_workers=1, n_epochs=n_epochs, print_every=5,
                 dev_data=dev_data, metrics=metric,
                 metric_key='t10',
                 validate_every=-1, save_path=None, use_tqdm=True, device=None,
                 callbacks=callbacks, check_code_level=-1)
trainer.train(load_best_model=False)



