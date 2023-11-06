import torch
from torch import Tensor
from torch.nn import functional as F
from torch import distributed as dist
from torch import nn
import numpy as np


class SimpleContrastiveLoss:

    def __call__(self, x: Tensor, y: Tensor, target: Tensor = None, reduction: str = 'mean'):
        if target is None:
            target_per_qry = y.size(0) // x.size(0)
            target = torch.arange(
                0, x.size(0) * target_per_qry, target_per_qry, device=x.device, dtype=torch.long)
        logits = torch.matmul(x, y.transpose(0, 1))
        return F.cross_entropy(logits, target, reduction=reduction)


class DistributedContrastiveLoss(SimpleContrastiveLoss):
    def __init__(self, n_target: int = 0, scale_loss: bool = True):
        assert dist.is_initialized(), "Distributed training has not been properly initialized."
        super().__init__()
        self.word_size = dist.get_world_size()
        self.rank = dist.get_rank()
        self.scale_loss = scale_loss

    def __call__(self, x: Tensor, y: Tensor, **kwargs):
        dist_x = self.gather_tensor(x)
        dist_y = self.gather_tensor(y)
        loss = super().__call__(dist_x, dist_y, **kwargs)
        if self.scale_loss:
            loss = loss * self.word_size
        return loss

    def gather_tensor(self, t):
        gathered = [torch.empty_like(t) for _ in range(self.word_size)]
        dist.all_gather(gathered, t)
        gathered[self.rank] = t
        return torch.cat(gathered, dim=0)


def compute_aspect_loss(scores, target, loss_type='softmax',
                        pos_aspect_label=None, zero_id=None):
    """
    scores: [bs, class_num]
    target: [bs]
    pos_aspect_label: if exists, [bs, class_num]
    zero_id: empty aspect label id
    """
    if loss_type == 'softmax':
        cross_entropy = nn.CrossEntropyLoss(reduction='mean')
        return cross_entropy(scores, target)
    elif loss_type == 'multi-softmax':
        cross_entropy_batch = nn.CrossEntropyLoss(reduction='none')
        # 如果没有正例，算出来的loss本身为0
        # target: [bs, class_num], 作为每个类别的概率，要float
        loss = cross_entropy_batch(scores, target.float())  # [128]
        pos_per_sample = target.sum(axis=1)  # 每一行有多少个正例，可能为0
        loss = loss * (pos_per_sample != 0).int()  # 对无正例的样本loss置0
        # pos_per_sample[pos_per_sample < 1e-8] = 1.0  # 每个样本内根据pos数量取均值
        all_ones = torch.ones(pos_per_sample.shape,
                              device=pos_per_sample.device)
        pos_per_sample = torch.where(
            pos_per_sample == 0, all_ones, pos_per_sample)
        loss = loss / pos_per_sample

        if (loss != 0.0).int().sum() == 0:  # batch内都是0
            return 0 * loss.sum()
        else:
            loss = loss.sum() / (loss != 0.0).int().sum()  # 针对样本数量取均值
            return loss
    elif loss_type == 'softmax-ignore':
        # 忽略掉 aspect 为空的
        cross_entropy_batch = nn.CrossEntropyLoss(reduction='none')
        batch_loss = cross_entropy_batch(scores, target)  # [bs]
        batch_num = (batch_loss != 0.0).int()
        if zero_id is None:
            raise ValueError('zero_id is None!')
        batch_mask = (target != zero_id).float()
        batch_num = (batch_num * batch_mask).sum()
        batch_loss = (batch_loss * batch_mask).sum()
        if batch_num == 0:
            new_loss = 0 * batch_loss
        else:
            new_loss = torch.div(batch_loss, batch_num)
        return new_loss


class MultiLabelCircleLoss(nn.Module):
    def __init__(self, reduction="mean", have_weight=0, inf=1e12):
        """CircleLoss of MultiLabel, 多个目标类的多标签分类场景，希望“每个目标类得分都不小于每个非目标类的得分”
        多标签分类的交叉熵(softmax+crossentropy推广, N选K问题), LSE函数的梯度恰好是softmax函数
        让同类相似度与非同类相似度之间拉开一定的margin。
          - 使同类相似度比最大的非同类相似度更大。
          - 使最小的同类相似度比最大的非同类相似度更大。
          - 所有同类相似度都比所有非同类相似度更大。
        urls: [将“softmax+交叉熵”推广到多标签分类问题](https://spaces.ac.cn/archives/7359)
        args:
            reduction: str, Specifies the reduction to apply to the output, 输出形式. 
                            eg.``'none'`` | ``'mean'`` | ``'sum'``. ``'none'``
            inf: float, Minimum of maths, 无穷大.  eg. 1e12
            have_weight: bool, 如果为true，代表输入的labels非0/1，数值越大代表该label越重要
        returns:
            Tensor of loss.
        examples:
            >>> label, logits = [[1, 1, 1, 1], [0, 0, 0, 1]], [[0, 1, 1, 0], [1, 0, 0, 1],]
            >>> label, logits = torch.tensor(label).float(), torch.tensor(logits).float()
            >>> loss = MultiLabelCircleLoss()(logits, label)
        """
        super(MultiLabelCircleLoss, self).__init__()
        self.reduction = reduction
        self.inf = inf  # 无穷大
        self.have_weight = have_weight
    
    def logsumexp_weight(self, input, dim, weight=None, keepdim=False):
        """
        input 是输入张量，dim 是指定的维度，keepdim 用于决定是否保持维度
        避免数值不稳定性
        """
        max_val, _ = input.max(dim, keepdim=True)  # max_val 是输入张量 input 在维度 dim 上的最大值，_ 表示不需要最大值的位置
        output = (input - max_val).exp()
        if weight is not None:
            output = output * weight
        output = output.sum(dim, keepdim=True)
        output = output.log() + max_val
        if not keepdim:
            output = output.squeeze(dim)
        return output

    def forward(self, logits, ori_labels):
        """多标签分类的交叉熵
        说明：logits和labels的shape一致，labels的元素非0即1，
             1表示对应的类为目标类，0表示对应的类为非目标类。
        警告：请保证logits的值域是全体实数，换言之一般情况下logits
             不用加激活函数，尤其是不能加sigmoid或者softmax！预测
             阶段则输出y_pred大于0的类。
        """
        if self.have_weight:
            all_ones = torch.ones(ori_labels.size()).to(ori_labels.device)
            all_zeros = torch.zeros(ori_labels.size()).to(ori_labels.device)
            labels = torch.where(ori_labels > 0, all_ones, all_zeros)  # 转为01的
        else:
            labels = ori_labels
        logits = (1 - 2 * labels) * logits  # 把pos的乘-1
        logits_neg = logits - labels * self.inf  # 把pos的位置变成负无穷
        logits_pos = logits - (1 - labels) * self.inf  # 把neg的位置变成正无穷
        zeros = torch.zeros_like(logits[..., :1])

        logits_neg = torch.cat([logits_neg, zeros], dim=-1)
        logits_pos = torch.cat([logits_pos, zeros], dim=-1)
        neg_loss = torch.logsumexp(logits_neg, dim=-1)  #  neg 不用动

        if self.have_weight:
            # pos: log sum exp(-sj) => log sum w_j exp(-sj)
            weight = torch.where(ori_labels == 0, all_ones, ori_labels)  # 大家的weight至少为1
            weight = torch.cat([ori_labels, torch.ones_like(weight[..., :1])], dim=-1)
            pos_loss = self.logsumexp_weight(logits_pos, dim=-1, weight=weight)
        else:
            # torch.logsumexp(logits_pos, dim=-1)[:5]
            # self.logsumexp_weight(logits_pos, dim=-1, weight=torch.ones_like(logits_pos))[:5]
            pos_loss = torch.logsumexp(logits_pos, dim=-1)

        loss = neg_loss + pos_loss
        example_mask = (ori_labels.sum(1) > 0).int()  # 1 代表至少有一个正类，没有正类的样本不算loss
        new_loss = loss * example_mask  # 把对应loss置0
        if "mean" == self.reduction:
            loss = new_loss.sum() / example_mask.sum()
        else:
            loss = new_loss.sum()
        return loss