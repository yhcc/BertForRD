from fastNLP import MetricBase


class SaveMetric(MetricBase):
    #  记录预测结果, 第一列是需要预测的结果, 第二列之后是预测的单语
    def __init__(self, fp, lg2_dict, start, end, keep_number=10):
        """

        :param fp: 存结果的路径，存格式为target source_pred1 source_pred2
        :param lg2_dict: key为word, value是idx，reverse一下可以得到idx到word的转换，这是source的词表
        :param start: source语言的开头index
        :param end: source语言的结尾的index
        """
        super().__init__()
        self.fp = fp
        self.lg2_dict = {value:key for key,value in lg2_dict.items()}
        self.start = start
        self.end = end
        self.keep_number = keep_number
        self.target_preds = []
        self.source_preds = []

    def evaluate(self, pred, word):
        pred = pred[:, self.start:self.end]
        _, indices = pred.topk(dim=-1, k=self.keep_number, sorted=True, largest=True)
        indices = indices.tolist()
        target_word1 = word.tolist()
        for target, pred in zip(target_word1, indices):
            self.target_preds.append(target)
            self.source_preds.append([self.lg2_dict[p] for p in pred])

    def get_metric(self, reset=True):
        with open(self.fp, 'w', encoding='utf-8') as f:
            for t, ps in zip(self.target_preds, self.source_preds):
                f.write(t + ' ' + ' '.join(ps) + '\n')

        return {'top10': 0}


