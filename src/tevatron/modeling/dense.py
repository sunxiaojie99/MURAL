import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
import logging
from .encoder import EncoderPooler, EncoderModel, EncoderOutput

import os
import copy
import logging
import json
from typing import Dict, Optional
import math

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


class DensePooler(EncoderPooler):
    def __init__(self, input_dim: int = 768, output_dim: int = 768, tied=True):
        super(DensePooler, self).__init__()
        self.linear_q = nn.Linear(input_dim, output_dim)
        if tied:
            self.linear_p = self.linear_q
        else:
            self.linear_p = nn.Linear(input_dim, output_dim)
        self._config = {'input_dim': input_dim,
                        'output_dim': output_dim, 'tied': tied}

    def forward(self, q: Tensor = None, p: Tensor = None, **kwargs):
        if q is not None:
            return self.linear_q(q[:, 0])
        elif p is not None:
            return self.linear_p(p[:, 0])
        else:
            raise ValueError


class DenseModel(EncoderModel):
    def encode_passage(self, psg):
        if psg is None:
            return None
        psg_out = self.lm_p(**psg, return_dict=True)
        p_hidden = psg_out.last_hidden_state
        if self.pooler is not None:
            p_reps = self.pooler(p=p_hidden)  # D * d
        else:
            p_reps = p_hidden[:, 0]
        return p_reps

    def encode_query(self, qry):
        if qry is None:
            return None
        qry_out = self.lm_q(**qry, return_dict=True)
        q_hidden = qry_out.last_hidden_state  # [bs, seq_len, dim]
        if self.pooler is not None:
            q_reps = self.pooler(q=q_hidden)
        else:
            q_reps = q_hidden[:, 0]
        return q_reps

    def compute_similarity(self, q_reps, p_reps):
        return torch.matmul(q_reps, p_reps.transpose(0, 1))

    @staticmethod
    def load_pooler(model_weights_file, **config):
        pooler = DensePooler(**config)
        pooler.load(model_weights_file)
        return pooler

    @staticmethod
    def build_pooler(model_args):
        pooler = DensePooler(
            model_args.projection_in_dim,
            model_args.projection_out_dim,
            tied=not model_args.untie_encoder
        )  # 如果前面共享的encoder，那么投影层就不共享；如果是两个encoder，就需要相同的投影层来到一个空间
        pooler.load(model_args.model_name_or_path)
        return pooler

    def forward(self, query: Dict[str, Tensor] = None, passage: Dict[str, Tensor] = None):
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

            loss = self.compute_loss(scores, target)  # l2 norm搭配温度系数使用
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
