from fastNLP import MetricBase
import numpy as np


class BiMetric(MetricBase):
    """
    两种语言的评测
    """
    def __init__(self):
        super().__init__()
        self.total = 0
        self.ranks = []
        self.top1 = 0
        self.top10 = 0

    def evaluate(self, pred, target):
        """

        :param pred: batch x vocab_size
        :param target: batch_size x num_repeat', LongTensor, 由于大小写的原因
        :return:
        """
        self.total += pred.size(0)
        pred, indices = pred.sort(dim=-1, descending=True)
        _indices = indices[:, :1000].tolist()
        target_word1 = target.tolist()
        for index,(pred_i, target_word) in enumerate(zip(_indices, target_word1)):
            try:
                self.ranks.append(1/(pred_i.index(target_word)+1))
            except:
                try:
                    self.ranks.append(1/(indices[index][1000:].tolist().index(target_word)+1000))
                except:
                    self.ranks.append(1/pred.size(1))
            if target_word in pred_i[:10]:
                self.top10 += 1
                if target_word == pred_i[0]:
                    self.top1 += 1

    def get_metric(self, reset=True):
        res = {'t1':round(self.top1/self.total, 3), 't10':round(self.top10/self.total, 3),
               'rank': round(np.mean(self.ranks), 3)
               }
        self.last_metrics = res.copy()
        if reset:
            self.total = 0
            self.ranks = []
            self.top1 = 0
            self.top10 = 0

        return res


class JointMetric(MetricBase):
    def __init__(self, start, end):
        """

        :param int start: 预测中词表的开头
        :param int end: 预测中词表的结尾
        """
        super().__init__()
        self.total = 0
        self.ranks = []
        self.top1 = 0
        self.top10 = 0
        self.start = start
        self.end = end

    def evaluate(self, pred, target):
        """

        :param pred: batch x vocab_size
        :param target: batch_size, LongTensor
        :return:
        """
        pred = pred[:, self.start:self.end]
        target = target - self.start
        self.total += pred.size(0)
        pred, indices = pred.sort(dim=-1, descending=True)
        _indices = indices[:, :1000].tolist()
        target_word1 = target.tolist()
        for index, (pred_i, target_word) in enumerate(zip(_indices, target_word1)):
            try:
                self.ranks.append(1 / (pred_i.index(target_word) + 1))
            except:
                try:
                    self.ranks.append(1 / (indices[index][1000:].tolist().index(target_word) + 1000))
                except:
                    self.ranks.append(1 / pred.size(1))
            if target_word in pred_i[:10]:
                self.top10 += 1
                if target_word == pred_i[0]:
                    self.top1 += 1

    def get_metric(self, reset=True):
        res = {'t1': round(self.top1 / self.total, 3), 't10': round(self.top10 / self.total, 3),
               'rank': round(np.mean(self.ranks), 3)
               }
        self.last_metrics = res.copy()
        if reset:
            self.total = 0
            self.ranks = []
            self.top1 = 0
            self.top10 = 0

        return res


class SummaryMetric(MetricBase):
    def __init__(self, metrics):
        super().__init__()
        self.metrics = metrics

    def evaluate(self, *args, **kwargs):
        pass

    def get_metric(self, reset=True):
        res = dict()
        for metric in self.metrics:
            for key, value in metric.last_metrics.items():
                if key in res:
                    res[key].append(value)
                else:
                    res[key] = [value]
        assert 1 == len(set(map(len, res.values())))
        _res = {}
        for key, value in res.items():
            _res[key] = round(np.mean(value), 3)
        return _res
