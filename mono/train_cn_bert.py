import sys
sys.path.append('../')
import os
if 'p' in os.environ:
    os.environ['CUDA_VISIBLE_DEVICES']=os.environ['p']

import torch
from torch import optim
from fastNLP import Trainer, CrossEntropyLoss
from fastNLP import BucketSampler, cache_results, WarmupCallback, GradientClipCallback, FitlogCallback
from mono.data.pipe import CNBertPipe
from mono.model.bert import CNBertReverseDict
import fitlog
from mono.model.metrics import MonoMetric
from joint.data.utils import clip_max_length
# fitlog.debug()
fitlog.set_log_dir('cn_logs')
fitlog.add_hyper_in_file(__file__)


paths = '../data/cn'
#########hyper
model_name = 'bert'  # change this to roberta, you can run the roberta version
max_word_len = 4
lr = 5e-5
batch_size = 80
n_epochs = 20
#########hyper
if model_name == 'bert':
    pre_name = 'bert-chinese-wwm'
elif model_name=='roberta':
    pre_name = 'chinese_roberta_wwm_ext'
else:
    raise Exception()


@cache_results('caches/cn_{}.pkl'.format(pre_name.split('/')[-1], max_word_len), _refresh=False)
def get_data():
    data_bundle = CNBertPipe(pre_name, max_word_len).process_from_file(paths)
    return data_bundle

data_bundle = get_data()
print(data_bundle)
word2bpes = data_bundle.word2bpes
print(f"In total {len(word2bpes)} target words")
pad_id = data_bundle.pad_id

model = CNBertReverseDict(pre_name, word2bpes, pad_id=pad_id,
                          number_word_in_train=data_bundle.number_word_in_train)

if torch.cuda.is_available():
    model.cuda()

optimizer = optim.AdamW(model.parameters(), lr=lr)

data = {}
for name in ['desc', 'question', 'seen_test', 'unseen_test']:
    data[name] = data_bundle.get_dataset(name)

callbacks = [GradientClipCallback(clip_type='value'), WarmupCallback(warmup=0.1, schedule='linear')]
callbacks.append(FitlogCallback(data=data, verbose=1))
train_data = data_bundle.get_dataset('train')
train_data.add_seq_len('input')
sampler = BucketSampler()

clip_max_length(train_data, data_bundle)

trainer = Trainer(train_data=train_data, model=model,
                  optimizer=optimizer, loss=CrossEntropyLoss(),
                 batch_size=batch_size, sampler=sampler, drop_last=False, update_every=1,
                 num_workers=1, n_epochs=n_epochs, print_every=5,
                 dev_data=data_bundle.get_dataset('dev'), metrics=MonoMetric(),
                 metric_key='t1',
                 validate_every=-1, save_path='save_models/', use_tqdm=True, device=None,
                 callbacks=callbacks, check_code_level=0)
trainer.train(load_best_model=False)
fitlog.add_other(trainer.start_time, name='start_time')

