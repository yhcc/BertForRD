from fastNLP import LossBase
import torch.nn.functional as F


class MixLoss(LossBase):
    def __init__(self, shifts, lg_lambda=0.0001):
        super().__init__()
        self.lg_lambda = lg_lambda
        self.shifts = shifts
        self.num_language = len(shifts)-1

    def get_loss(self, pred, target, lg_pred=None, language_ids=None):
        loss = 0
        language_id = language_ids[:, 0]  #
        for i in range(self.num_language):
            language_mask = language_id.eq(i)  # bsz
            if language_mask.sum().item()>0:
                start, end = self.shifts[i], self.shifts[i+1]
                tmp_target = target.masked_select(language_mask) - start  # 取出当前语言的
                tmp_pred = pred[:, start:end].masked_select(language_mask[:, None]).view(-1, end-start)  # bsz x vocab_size
                loss = loss + F.cross_entropy(tmp_pred, tmp_target, reduction='sum')

        # loss1 = F.cross_entropy(pred, target)
        if lg_pred is not None:
            loss2 = F.cross_entropy(lg_pred.transpose(1, 2), language_ids)
            loss = loss/pred.size(0)+self.lg_lambda*loss2
        else:
            loss = loss / pred.size(0)

        return loss

