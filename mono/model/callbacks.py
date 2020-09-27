from fastNLP import Callback
from copy import deepcopy
from fastNLP import Tester
from fastNLP import DataSet
import fitlog


class FitlogCallback(Callback):
    r"""
    该callback可将loss和progress写入到fitlog中; 如果Trainer有dev的数据，将自动把dev的结果写入到log中; 同时还支持传入
    一个(或多个)test数据集进行测试(只有在trainer具有dev时才能使用)，每次在dev上evaluate之后会在这些数据集上验证一下。
    并将验证结果写入到fitlog中。这些数据集的结果是根据dev上最好的结果报道的，即如果dev在第3个epoch取得了最佳，则
    fitlog中记录的关于这些数据集的结果就是来自第三个epoch的结果。
    """

    def __init__(self, data=None, tester=None, log_loss_every=0, verbose=0, log_exception=False,
                 separate_max=False):
        r"""

        :param ~fastNLP.DataSet,Dict[~fastNLP.DataSet] data: 传入DataSet对象，会使用多个Trainer中的metric对数据进行验证。如果需要
            传入多个DataSet请通过dict的方式传入，dict的key将作为对应dataset的name传递给fitlog。data的结果的名称以'data'开头。
        :param ~fastNLP.Tester,Dict[~fastNLP.Tester] tester: Tester对象，将在on_valid_end时调用。tester的结果的名称以'tester'开头
        :param int log_loss_every: 多少个step记录一次loss(记录的是这几个batch的loss平均值)，如果数据集较大建议将该值设置得
            大一些，不然会导致log文件巨大。默认为0, 即不要记录loss。
        :param int verbose: 是否在终端打印evaluation的结果，0不打印。
        :param bool log_exception: fitlog是否记录发生的exception信息
        :param bool separate_max: 是否单独记录metric每项的最大dev以及其对应的测试值，如果为True，则只允许通过data传入数据。对于类似
            F值类似的metric，改值不能为True，否则导致记录错误
        """
        super().__init__()
        self.datasets = {}
        self.testers = {}
        self._log_exception = log_exception
        assert isinstance(log_loss_every, int) and log_loss_every >= 0
        if tester is not None:
            assert separate_max is False, "No tester is allowed when `separate_max` is True."
            if isinstance(tester, dict):
                for name, test in tester.items():
                    if not isinstance(test, Tester):
                        raise TypeError(f"{name} in tester is not a valid fastNLP.Tester.")
                    self.testers['tester-' + name] = test
            if isinstance(tester, Tester):
                self.testers['tester-test'] = tester
            for tester in self.testers.values():
                setattr(tester, 'verbose', 0)

        if isinstance(data, dict):
            for key, value in data.items():
                assert isinstance(value, DataSet), f"Only DataSet object is allowed, not {type(value)}."
            for key, value in data.items():
                self.datasets['data-' + key] = value
        elif isinstance(data, DataSet):
            self.datasets['data-test'] = data
        elif data is not None:
            raise TypeError("data receives dict[DataSet] or DataSet object.")

        self.verbose = verbose
        self._log_loss_every = log_loss_every
        self._avg_loss = 0
        self.separate_max = separate_max

    def on_train_begin(self):
        if (len(self.datasets) > 0 or len(self.testers) > 0) and self.trainer.dev_data is None:
            raise RuntimeError("Trainer has no dev data, you cannot pass extra data to do evaluation.")

        if len(self.datasets) > 0:
            for key, data in self.datasets.items():
                tester = Tester(data=data, model=self.model,
                                batch_size=self.trainer.kwargs.get('dev_batch_size', self.batch_size),
                                metrics=self.trainer.metrics,
                                verbose=0,
                                use_tqdm=self.trainer.test_use_tqdm)
                self.testers[key] = tester
        fitlog.add_progress(total_steps=self.n_steps)

    def on_backward_begin(self, loss):
        if self._log_loss_every > 0:
            self._avg_loss += loss.item()
            if self.step % self._log_loss_every == 0:
                fitlog.add_loss(self._avg_loss / self._log_loss_every * self.update_every, name='loss', step=self.step,
                                epoch=self.epoch)
                self._avg_loss = 0

    def on_valid_end(self, eval_result, metric_key, optimizer, better_result):
        if better_result:
            eval_result = deepcopy(eval_result)
            eval_result['step'] = self.step
            eval_result['epoch'] = self.epoch
            fitlog.add_best_metric(eval_result)
        fitlog.add_metric(eval_result, step=self.step, epoch=self.epoch)
        if len(self.testers) > 0:
            for key, tester in self.testers.items():
                try:
                    eval_result = tester.test()
                    if self.verbose != 0:
                        self.pbar.write("FitlogCallback evaluation on {}:".format(key))
                        self.pbar.write(tester._format_eval_results(eval_result))
                    fitlog.add_metric(eval_result, name=key, step=self.step, epoch=self.epoch)
                    if better_result:
                        fitlog.add_best_metric(eval_result, name=key)
                except Exception as e:
                    self.pbar.write("Exception happens when evaluate on DataSet named `{}`.".format(key))
                    raise e

    def on_train_end(self):
        fitlog.finish()

    def on_exception(self, exception):
        fitlog.finish(status=1)
        if self._log_exception:
            fitlog.add_other(repr(exception), name='except_info')
