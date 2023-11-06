import copy
import json
import os
from dataclasses import dataclass
from typing import Dict, Optional
import numpy as np

import torch
from torch import nn, Tensor
import torch.distributed as dist
from transformers import PreTrainedModel, AutoModel
from transformers.file_utils import ModelOutput

from tevatron.arguments import ModelArguments, \
    TevatronTrainingArguments as TrainingArguments

import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


# def compute_aspect_loss(scores, target, loss_type='softmax',
#                         pos_aspect_label=None, zero_id=None):
#     """
#     scores: [bs, class_num]
#     target: [bs]
#     pos_aspect_label: if exists, [bs, class_num]
#     zero_id: empty aspect label id
#     """
#     if loss_type == 'softmax':
#         cross_entropy = nn.CrossEntropyLoss(reduction='mean')
#         return cross_entropy(scores, target)
#     elif loss_type == 'softmax-ignore':
#         # 忽略掉 aspect 为空的
#         cross_entropy_batch = nn.CrossEntropyLoss(reduction='none')
#         batch_loss = cross_entropy_batch(scores, target)  # [bs]
#         batch_num = (batch_loss != 0.0).int()
#         if zero_id is None:
#             raise ValueError('zero_id is None!')
#         batch_mask = (target != zero_id).float()
#         batch_num = (batch_num * batch_mask).sum()
#         batch_loss = (batch_loss * batch_mask).sum()
#         if batch_num == 0:
#             new_loss = 0 * batch_loss
#         else:
#             new_loss = torch.div(batch_loss, batch_num)
#         return new_loss
#     elif loss_type == 'multi-softmax':
#         cross_entropy_batch = nn.CrossEntropyLoss(reduction='none')
#         # 如果没有正例，算出来的loss本身为0
#         # target: [bs, class_num], 作为每个类别的概率，要float
#         loss = cross_entropy_batch(scores, target.float())  # [128]
#         pos_per_sample = target.sum(axis=1)  # 每一行有多少个正例，可能为0
#         loss = loss * (pos_per_sample != 0).int()  # 对无正例的样本loss置0
#         # pos_per_sample[pos_per_sample < 1e-8] = 1.0  # 每个样本内根据pos数量取均值
#         all_ones = torch.ones(pos_per_sample.shape,
#                               device=pos_per_sample.device)
#         pos_per_sample = torch.where(
#             pos_per_sample == 0, all_ones, pos_per_sample)
#         loss = loss / pos_per_sample
#         loss = loss.sum() / (loss != 0.0).int().sum()  # 针对样本数量取均值
#         return loss


@dataclass
class EncoderOutput(ModelOutput):
    q_reps: Optional[Tensor] = None
    p_reps: Optional[Tensor] = None
    loss: Optional[Tensor] = None
    scores: Optional[Tensor] = None


class EncoderPooler(nn.Module):
    def __init__(self, **kwargs):
        super(EncoderPooler, self).__init__()
        self._config = {}

    def forward(self, q_reps, p_reps):
        raise NotImplementedError('EncoderPooler is an abstract class')

    def load(self, model_dir: str):
        pooler_path = os.path.join(model_dir, 'pooler.pt')
        if pooler_path is not None:
            if os.path.exists(pooler_path):
                logger.info(f'Loading Pooler from {pooler_path}')
                state_dict = torch.load(pooler_path, map_location='cpu')
                self.load_state_dict(state_dict)
                # self.load_state_dict_with_missing_keys_report(self, state_dict)
                # self.load_state_dict(state_dict, strict=False)
                return
        logger.info("Training Pooler from scratch")
        return
    
    def load_state_dict_with_missing_keys_report(self, model, state_dict):
        model_keys = set(model.state_dict().keys())
        loaded_keys = set(state_dict.keys())

        missing_keys = model_keys - loaded_keys
        unexpected_keys = loaded_keys - model_keys

        if missing_keys:
            print('missing pooler keys in state_dict:', missing_keys)

        if unexpected_keys:
            print('unexpected pooler keys in state_dict:', unexpected_keys)

    def save_pooler(self, save_path):
        torch.save(self.state_dict(), os.path.join(save_path, 'pooler.pt'))
        with open(os.path.join(save_path, 'pooler_config.json'), 'w') as f:
            json.dump(self._config, f)


class EncoderModel(nn.Module):
    TRANSFORMER_CLS = AutoModel

    def __init__(self,
                 lm_q: PreTrainedModel,
                 lm_p: PreTrainedModel,
                 pooler: nn.Module = None,
                 untie_encoder: bool = False,
                 negatives_x_device: bool = False
                 ):
        super().__init__()
        self.lm_q = lm_q
        self.lm_p = lm_p
        self.pooler = pooler
        self.cross_entropy = nn.CrossEntropyLoss(reduction='mean')
        self.negatives_x_device = negatives_x_device
        self.untie_encoder = untie_encoder
        if self.negatives_x_device:
            if not dist.is_initialized():
                raise ValueError(
                    'Distributed training has not been initialized for representation all gather.')
            self.process_rank = dist.get_rank()
            self.world_size = dist.get_world_size()

    def forward(self, query: Dict[str, Tensor] = None, passage: Dict[str, Tensor] = None,
                brand_id=None, color_id=None):
        """
        query: {"input_ids": [bs, q_seq_len], "attention_mask": [bs, seq_len]}, eg, [32, 32]
        passage: {"input_ids": [bs*train_n_passages, p_seq_len], "attention_mask": [bs*train_n_passages, p_seq_len]} , eg, [64, 156]
        """
        q_reps = self.encode_query(query)  # [bs, 768]
        p_reps = self.encode_passage(passage)  # [bs*train_n_passages, 768]

        # for inference, infer query 或者 passage
        if q_reps is None or p_reps is None:
            return EncoderOutput(
                q_reps=q_reps,
                p_reps=p_reps
            )

        # for training
        if self.training:
            if self.negatives_x_device:
                q_reps = self._dist_gather_tensor(q_reps)
                p_reps = self._dist_gather_tensor(p_reps)

            scores = self.compute_similarity(q_reps, p_reps)  # [32*4, 64*4]
            scores = scores.view(q_reps.size(0), -1)

            # [0,1,2,..., 32*4-1], [32*4,]
            target = torch.arange(scores.size(
                0), device=scores.device, dtype=torch.long)
            # * ( (64*4) // 32*4 ) = 2 = train_n_passages
            target = target * (p_reps.size(0) // q_reps.size(0))
            # [0, 2, 4, 6, ...] 这些是pos passage的 idx

            loss = self.compute_loss(scores, target)
            if self.negatives_x_device:
                loss = loss * self.world_size  # counter average weight reduction
        # for eval
        else:
            scores = self.compute_similarity(q_reps, p_reps)
            loss = None
        return EncoderOutput(
            loss=loss,
            scores=scores,
            q_reps=q_reps,
            p_reps=p_reps,
        )

    def cal_acc_for_mul(self, true_label, pred_logits, top_k=3, is_recall=False, is_train=False):
        """
        每个样本topk的正确预测true_label的数量/true_label的总数量
        true_label: [bs, class_num] 可能有多个1
        pred_label: [bs, class_num]  每个类别的数目
        """
        topk_res = torch.topk(pred_logits, k=top_k, dim=1)  # topk的结果
        topk_class = topk_res.indices  # topk的位置 [bs, top_k]
        # 全0，根据topk构造和true_label对应的pred_label
        pred_label = np.zeros(pred_logits.shape, dtype=np.int64)

        index = np.arange(pred_logits.shape[0]).repeat(
            top_k)  # [0,0,0,1,1,1,..,bs-1,bs-1,bs-1]
        topk_class = topk_class.reshape(-1).cpu().numpy()  # [bs*top_k]
        pred_label[index, topk_class] = 1  # [bs, class_num] 索引，将topk设1
        if is_recall:
            same_num = np.sum((true_label.numpy() == pred_label).astype(
                int) * true_label.numpy())
            all_num = np.sum((true_label.numpy() == 1).astype(int))
            if all_num == 0:
                return -1
            topk_acc = same_num / all_num
            return [topk_acc]
        else:
            # 每个样本的判断结果是否正确
            # topk里是否有对的，有的话，> 0
            is_correct = (np.logical_and(pred_label, true_label).sum(1) > 0).int()
            is_correct = is_correct[torch.where((true_label.sum(1)>0))]
            if is_train:
                return is_correct.sum()/is_correct.shape[0]
            return is_correct

    @staticmethod
    def build_pooler(model_args):
        return None

    @staticmethod
    def load_pooler(weights, **config):
        return None

    def encode_passage(self, psg):
        raise NotImplementedError('EncoderModel is an abstract class')

    def encode_query(self, qry):
        raise NotImplementedError('EncoderModel is an abstract class')

    def compute_similarity(self, q_reps, p_reps):
        return torch.matmul(q_reps, p_reps.transpose(0, 1))

    def compute_loss(self, scores, target):
        return self.cross_entropy(scores, target)

    def _dist_gather_tensor(self, t: Optional[torch.Tensor]):
        """
        http://www.manongjc.com/detail/31-fyszsjjqtkjzfch.html
        https://zhuanlan.zhihu.com/p/518802196
        """
        if t is None:
            return None
        t = t.contiguous()

        all_tensors = [torch.empty_like(t) for _ in range(self.world_size)]
        # all_tensors: [tensor([1, 2]), tensor([3, 4])] # Rank 0
        dist.all_gather(all_tensors, t)

        all_tensors[self.process_rank] = t
        all_tensors = torch.cat(all_tensors, dim=0)

        return all_tensors

    @classmethod
    def build(
            cls,
            model_args: ModelArguments,
            train_args: TrainingArguments,
            **hf_kwargs,
    ):
        # load local
        if os.path.isdir(model_args.model_name_or_path):
            if model_args.untie_encoder:  # 不共享bert
                _qry_model_path = os.path.join(
                    model_args.model_name_or_path, 'query_model')
                _psg_model_path = os.path.join(
                    model_args.model_name_or_path, 'passage_model')
                if not os.path.exists(_qry_model_path):
                    _qry_model_path = model_args.model_name_or_path
                    _psg_model_path = model_args.model_name_or_path
                logger.info(
                    f'loading query model weight from {_qry_model_path}')
                lm_q = cls.TRANSFORMER_CLS.from_pretrained(
                    _qry_model_path,
                    **hf_kwargs
                )
                logger.info(
                    f'loading passage model weight from {_psg_model_path}')
                lm_p = cls.TRANSFORMER_CLS.from_pretrained(
                    _psg_model_path,
                    **hf_kwargs
                )
            else:  # 默认为false，共享bert
                lm_q = cls.TRANSFORMER_CLS.from_pretrained(
                    model_args.model_name_or_path, **hf_kwargs)
                lm_p = lm_q
        # load pre-trained
        else:
            lm_q = cls.TRANSFORMER_CLS.from_pretrained(
                model_args.model_name_or_path, **hf_kwargs)
            lm_p = copy.deepcopy(lm_q) if model_args.untie_encoder else lm_q

        if model_args.add_pooler:
            pooler = cls.build_pooler(model_args)
        else:
            pooler = None

        model = cls(
            lm_q=lm_q,
            lm_p=lm_p,
            pooler=pooler,
            negatives_x_device=train_args.negatives_x_device,
            untie_encoder=model_args.untie_encoder
        )
        return model

    @classmethod
    def load(
            cls,
            model_name_or_path,
            model_args=None,
            train_args=None,
            **hf_kwargs,
    ):
        # load local
        untie_encoder = True
        if os.path.isdir(model_name_or_path):
            _qry_model_path = os.path.join(model_name_or_path, 'query_model')
            _psg_model_path = os.path.join(model_name_or_path, 'passage_model')
            if os.path.exists(_qry_model_path):
                logger.info(
                    f'found separate weight for query/passage encoders')
                logger.info(
                    f'loading query model weight from {_qry_model_path}')
                lm_q = cls.TRANSFORMER_CLS.from_pretrained(
                    _qry_model_path,
                    **hf_kwargs
                )
                logger.info(
                    f'loading passage model weight from {_psg_model_path}')
                lm_p = cls.TRANSFORMER_CLS.from_pretrained(
                    _psg_model_path,
                    **hf_kwargs
                )
                untie_encoder = False
            else:
                logger.info(f'try loading tied weight')
                logger.info(f'loading model weight from {model_name_or_path}')
                lm_q = cls.TRANSFORMER_CLS.from_pretrained(
                    model_name_or_path, **hf_kwargs)
                lm_p = lm_q
        else:
            logger.info(f'try loading tied weight')
            logger.info(f'loading model weight from {model_name_or_path}')
            lm_q = cls.TRANSFORMER_CLS.from_pretrained(
                model_name_or_path, **hf_kwargs)
            lm_p = lm_q

        pooler_weights = os.path.join(model_name_or_path, 'pooler.pt')
        pooler_config = os.path.join(model_name_or_path, 'pooler_config.json')
        if os.path.exists(pooler_weights) and os.path.exists(pooler_config):
            logger.info(f'found pooler weight and configuration')
            with open(pooler_config) as f:
                pooler_config_dict = json.load(f)
            pooler = cls.load_pooler(model_name_or_path, **pooler_config_dict)
        else:
            pooler = None

        model = cls(
            lm_q=lm_q,
            lm_p=lm_p,
            pooler=pooler,
            untie_encoder=untie_encoder
        )
        return model

    def save(self, output_dir: str):
        if self.untie_encoder:
            os.makedirs(os.path.join(output_dir, 'query_model'))
            os.makedirs(os.path.join(output_dir, 'passage_model'))
            self.lm_q.save_pretrained(os.path.join(output_dir, 'query_model'))
            self.lm_p.save_pretrained(
                os.path.join(output_dir, 'passage_model'))
        else:
            self.lm_q.save_pretrained(output_dir)
        if self.pooler:
            self.pooler.save_pooler(output_dir)
