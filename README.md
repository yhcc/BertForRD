This is the code for ``BERT for Monolingual and Cross-Lingual Reverse Dictionary``. 如果您发现github网速较慢，您也可以在https://gitee.com/yh_cc/BertForRD 下载代码和数据。

python package requirements
```python
transformers
fastNLP
torch
```

The meaning of each folder is, you can run any `train_*.py` file with `python train_*.py`
```
joint/  # this folder includes the code to tackle the superversied multilingual sceneraio
    - train_bi_bert.py  # you can directly run `python train_bi_bert.py` to run
    - train_joint_bert.py 
mix/ # this folder includes the code to tackle the unaligned multilingual sceneraio
    - train_mix_bert.py  # This is the code to run three languages simutaneously.
    - train_pair_bert.py  # This is the code to run one pair of unaligned languages 
mono/ # this folder contains the code to run the monolingual reverse dictionayr scenerio
    - train_cn_bert.py  # This contains code for Chinese BERT and Roberta model 
    - train_en_bert.py
    - train_en_roberta.py
```


To ease the reproduction, we copied the data from https://github.com/thunlp/MultiRD and https://github.com/muhaochen/bilingual_dictionaries
 to the 'data.zip' file, unzip it will have the following folders
 ```
- cn  # Chinese reverse dictionary dataset released in `Multi-channel Reverse Dictionary Model, AAAI 2020`
    - desc.json  # Contains 200 word-description pairs give by Chinese native speaker
    - seen_test.json  # Contains 2000 seen words during training
    - unseen_test.jon  # Contains 2000 which is not presented in the training set
    - question.json  # Contains 272 real-world Chinese exam question-answers of writing the right word give a description from the Internet
    - train.json  # Contains 84694 word-definition pairs
    - target_words.txt  # The target word list

- en  # English reverse dictionary dataset collected in ` Learning to understand phrases by embedding the dictionary, 2016 TACL` 
    - desc.json # Contains 200 human-written word-description pairs.
    - seen.json # Contains 500 word-definition pairs which are seen during the training.
    - unseen.json # Contains 500 word-definition pairs which are not seen during the training.
    - training.json  # The training dataset
    - dev.json # The development set which includes both seen and unseen word-definition pairs.
    - target_words.txt  # the target word list

- mix  # Multilingual reverse dictionary data from `Learning to Represent Bilingual Dictionaries, CoNLL 2018`
    # This folder includes both monolingual and bilingual reverse dictionary 
    # {lg}.txt is the target word list
    # {lg}_test500.csv: The monolingual test set
    # {lg}_train500_10.csv: The monolingua train set
    # {lg}_dev.txt is the monolingual development set
    # {lg1}_{lg2}_dev.csv: The definition is in {lg2} and the target word is in {lg1}
    # {lg1}_{lg2}_test500.csv: 
    # {lg1}_{lg2}_train500_10.csv
```
