from fastNLP import MetricBase
import numpy as np


class MonoMetric(MetricBase):
    def __init__(self):
        super().__init__()
        self.total = 0
        self.ranks = []
        self.top1 = 0
        self.top10 = 0
        self.top100 = 0

    def evaluate(self, pred, target):
        """

        :param pred: batch x vocab_size
        :param target: batch_size, LongTensor
        :return:
        """
        self.total += pred.size(0)
        pred, indices = pred.topk(k=1000, dim=-1, largest=True, sorted=True)
        indices = indices.tolist()
        target_word1 = target.tolist()
        for pred_i, target_word in zip(indices, target_word1):
            try:
                self.ranks.append(pred_i.index(target_word))
            except:
                self.ranks.append(1000)
            if target_word in pred_i[:100]:
                self.top100 += 1
                if target_word in pred_i[:10]:
                    self.top10 += 1
                    if target_word == pred_i[0]:
                        self.top1 += 1

    def get_metric(self, reset=True):
        res = {'t1':round(self.top1/self.total, 3), 't10':round(self.top10/self.total, 3),
               't100':round(self.top100/self.total, 3),
               'rank': np.median(self.ranks), 'var': round(np.sqrt(np.var(self.ranks)), 2)
               }
        if reset:
            self.total = 0
            self.ranks = []
            self.top1 = 0
            self.top10 = 0
            self.top100 = 0
        return res