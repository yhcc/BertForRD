import sys
sys.path.append('../')
import os
if 'p' in os.environ:
    os.environ['CUDA_VISIBLE_DEVICES']=os.environ['p']

import torch
from torch import optim
from fastNLP import Trainer
from fastNLP import BucketSampler, cache_results, WarmupCallback, GradientClipCallback
from mix.data.pipe import MixUnalignBertPipe
from mix.data.loader import MixUnAlignLoader
# from V1.model.bert import ENBertReverseDict
from mix.model.bert import JointBertReverseDict
from fastNLP import DataSet, Tester
import fitlog
from mix.model.loss import MixLoss
from joint.model.metrics import JointMetric, SummaryMetric
from joint.data.utils import clip_max_length
from mix.model.metrics import SaveMetric
from joint.model.callback import FitlogCallback
# fitlog.debug()
fitlog.set_log_dir('logs')
fitlog.add_hyper_in_file(__file__)
fitlog.add_other('no lg loss', name='note')


paths = '../data/mix'
#######hyper
model_name = 'bert'
max_word_len = 5
lr = 5e-5
lg_lambda = 0.0
batch_size = 80
n_epochs = 20
#######hyper

pre_name = 'bert-base-multilingual-cased'
# transformers中bert-base-multilingual-cased


@cache_results('caches/mix_{}_{}.pkl'.format(pre_name.split('/')[-1], max_word_len), _refresh=False)
def get_data():
    data_bundle = MixUnAlignLoader().load(paths)
    data_bundle = MixUnalignBertPipe(pre_name, max_word_len).process(data_bundle)
    return data_bundle

data_bundle = get_data()
print(data_bundle)
train_word2bpes = data_bundle.train_word2bpes
target_word2bpes = data_bundle.target_word2bpes
print(f"In total {len(target_word2bpes)} target words, {len(train_word2bpes)} words.")


pad_id = data_bundle.pad_id
lg_dict = getattr(data_bundle, 'lg_dict')
lg_shifts = getattr(data_bundle, 'lg_shift')
train_lg_shifts = getattr(data_bundle, 'train_lg_shift')


train_data = DataSet()
for name, ds in data_bundle.iter_datasets():
    if 'train' in name:
        for ins in ds:
            train_data.append(ins)

train_data.add_seq_len('input')
train_data.set_input('input', 'language_ids')
train_data.set_target('target')
train_data.set_pad_val('input', pad_id)

clip_max_length(train_data, data_bundle, max_sent_len=50)

model = JointBertReverseDict(pre_name, train_word2bpes, target_word2bpes, pad_id=pad_id, num_languages=3)

if torch.cuda.is_available():
    model.cuda()

optimizer = optim.AdamW(model.parameters(), lr=lr)

data = {}
summary_ms = []
for name,ds in data_bundle.iter_datasets():
    if 'test' in name:
        _metric = JointMetric(lg_shifts[lg_dict[name[:2]]], lg_shifts[lg_dict[name[:2]]+1])
        tester = Tester(ds, model, _metric, batch_size=120, num_workers=1, device=None, verbose=1, use_tqdm=True)
        data[name] = tester
    elif 'dev' in name:
        _metric = JointMetric(train_lg_shifts[lg_dict[name[:2]]], train_lg_shifts[lg_dict[name[:2]]+1])
        tester = Tester(ds, model, _metric, batch_size=120, num_workers=1, device=None, verbose=1, use_tqdm=True)
        data[name] = tester
        summary_ms.append(_metric)

dev_data = data_bundle.get_dataset('en_dev')
metric = SummaryMetric(summary_ms)

callbacks = [GradientClipCallback(clip_type='value'), WarmupCallback(warmup=0.1, schedule='linear')]
callbacks.append(FitlogCallback(tester=data, verbose=1))

# from collections import Counter
# print(Counter(train_data.get_field('seq_len').content))
# exit(0)

sampler = BucketSampler()
trainer = Trainer(train_data=train_data, model=model,
                  optimizer=optimizer, loss=MixLoss(train_lg_shifts, lg_lambda=lg_lambda),
                 batch_size=batch_size, sampler=None, drop_last=False, update_every=1,
                 num_workers=1, n_epochs=n_epochs, print_every=5,
                 dev_data=dev_data[:2], metrics=metric,
                 metric_key='t1',
                 validate_every=-1, save_path='save_models/', use_tqdm=True, device=None,
                 callbacks=callbacks, check_code_level=-1)
trainer.train(load_best_model=True)
fitlog.add_other(trainer.start_time, name='start_time')
