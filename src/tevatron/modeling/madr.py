import torch
import torch.nn as nn
from torch import Tensor
import torch.nn.functional as F
import torch.distributed as dist
from transformers import PreTrainedModel
from dataclasses import dataclass
from transformers.file_utils import ModelOutput

import os
import copy
import logging
import json
from typing import Dict, Optional
import math
import numpy as np
from sklearn.metrics import accuracy_score


from .encoder import EncoderPooler, EncoderModel
from tevatron.arguments import ModelArguments, \
    TevatronTrainingArguments as TrainingArguments
from tevatron.loss import compute_aspect_loss

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


@dataclass
class EncoderOutput(ModelOutput):
    q_reps: Optional[Tensor] = None
    p_reps: Optional[Tensor] = None
    scores: Optional[Tensor] = None
    doc_aspect_weight: Optional[Tensor] = None
    query_aspect_weight: Optional[Tensor] = None
    loss: Optional[Tensor] = None
    ori_loss: Optional[Tensor] = None
    doc_loss_dict: Optional[Tensor] = None
    doc_acc_dict: Optional[Tensor] = None
    # for encode
    doc_label_dict: Optional[Tensor] = None
    # for train
    doc_need_aspect_alpha: Optional[Tensor] = None


class MADRPooler(EncoderPooler):
    def __init__(self, input_dim: int = 768, out_dim: int = 768,
                 q_pool_type='att', doc_pool_type='att', tied=True, aspect_num=None, begin_id=0):
        super(MADRPooler, self).__init__()

        self.q_pool_type = q_pool_type
        self.doc_pool_type = doc_pool_type
        self.aspect_num = aspect_num
        self.begin_id = begin_id

        print('============Extra k embeddings(except cls), add other')

        if q_pool_type == 'att':
            # ====attempt
            self.att_query_q = nn.Parameter(torch.ones(
                self.aspect_num - 1, input_dim).normal_(mean=0.0, std=0.02))  # other 是 cls本身

            self.key_w_q = nn.Linear(input_dim, input_dim)
            self.value_w_q = nn.Linear(input_dim, input_dim)
            self.cls_linear_q = nn.Linear(input_dim, self.aspect_num)

            self.attn_q = nn.MultiheadAttention(input_dim, 1)
        else:
            self.linear_q = nn.Linear(input_dim, out_dim)

        if doc_pool_type == 'att':
            if q_pool_type == 'att':  # 对于madr，pooler共享
                self.att_query_d = self.att_query_q
                self.key_w_d = self.key_w_q
                self.value_w_d = self.value_w_q
                self.cls_linear_d = self.cls_linear_q
                self.attn_d = self.attn_q

            else:
                self.att_query_d = nn.Parameter(torch.ones(
                    self.aspect_num - 1, input_dim).normal_(mean=0.0, std=0.02))

                self.key_w_d = nn.Linear(input_dim, input_dim)
                self.value_w_d = nn.Linear(input_dim, input_dim)
                self.cls_linear_d = nn.Linear(input_dim, self.aspect_num)

                self.attn_d = nn.MultiheadAttention(input_dim, 1)
        else:
            self.linear_doc = nn.Linear(input_dim, out_dim)

        if tied and doc_pool_type == 'cls' and q_pool_type == 'cls':
            self.linear_doc = self.linear_q

        self._config = {'input_dim': input_dim,
                        'out_dim': out_dim,
                        'q_pool_type': q_pool_type,
                        'doc_pool_type': doc_pool_type,
                        'tied': tied,
                        'aspect_num': aspect_num,
                        'begin_id': begin_id,
                        }  # 输入参数

    def forward(self, q: Tensor = None, doc: Tensor = None,
                doc_attention_mask=None, q_attention_mask=None, doc_hidden_states=None, **kwargs):
        """
        q: [bs, seq_len, dim]
        doc: [bs, seq_len, dim]
        doc_attention_mask: [bs, seq_len]
        doc_hidden_states: 13个[bs, seq_len, dim]的tensor


        res: [128, 768]
        aspect_emb: [128, 3, 768]
        """

        if q is not None:
            if self.q_pool_type == 'cls':
                return self.linear_q(q[:, 0]), None, None
            elif self.q_pool_type == 'att':
                # 1. 拆分
                q_token_seq = q[:, 1:]  # [128, 155, 768]
                q_token_mask = q_attention_mask[:, 1:]  # [128, 155]
                query = self.att_query_q  # [3, 768]
                bs = q_token_seq.shape[0]

                # === 用官方api
                aspect_emb, _ = self.attn_q(query.unsqueeze(0).repeat(bs, 1, 1).transpose(0, 1),
                                            q_token_seq.transpose(0, 1), q_token_seq.transpose(0, 1), key_padding_mask=q_token_mask == 0)
                aspect_emb = aspect_emb.transpose(
                    0, 1)  # [bs, aspect_num, 768]

                # 2.进入融合操作
                q_cls = q[:, 0]
                q_att = self.cls_linear_q(q_cls)  # [128, 3] 线性层变换到aspect大小维度
                q_att = torch.softmax(
                    q_att, dim=-1).unsqueeze(1)  # [128, 1, 3]
                
                # ====attempt, 额外添加cls
                aspect_emb = torch.cat([aspect_emb, q_cls.unsqueeze(1)], dim=1)  # [bs, aspect, 768], cls 当作other

                res = torch.bmm(q_att, aspect_emb).squeeze(
                    1)  # [128, 1, 768] -> [128, 768]

                return res, aspect_emb, q_att
            else:
                print('====== not valid q_pool_type!')
                raise ValueError
        elif doc is not None:
            if self.doc_pool_type == 'cls':
                return self.linear_doc(doc[:, 0]), None, None
            elif self.doc_pool_type == 'att':
                # 1. 拆分
                # 方式1: 用自定义的3个aspect query，对token seq emb注意力加权，得到3个aspect emb
                doc_token_seq = doc[:, 1:]  # [128, 155, 768]
                doc_token_mask = doc_attention_mask[:, 1:]  # [128, 155]
                query = self.att_query_d  # [3, 768]
                bs = doc_token_seq.shape[0]

                aspect_emb, _ = self.attn_d(query.unsqueeze(0).repeat(bs, 1, 1).transpose(0, 1),
                                            doc_token_seq.transpose(0, 1), doc_token_seq.transpose(0, 1), key_padding_mask=doc_token_mask == 0)
                aspect_emb = aspect_emb.transpose(
                    0, 1)  # [bs, aspect_num, 768]

                # 2.进入融合操作，
                # 方式1:直接对cls变换到3维
                doc_cls = doc[:, 0]
                # [128, 3] 线性层变换到aspect大小维度
                doc_att = self.cls_linear_d(doc_cls)
                doc_att = torch.softmax(
                    doc_att, dim=-1).unsqueeze(1)  # [128, 1, 3]
                
                # ====attempt
                aspect_emb = torch.cat([aspect_emb, doc_cls.unsqueeze(1)], dim=1)  # [bs, aspect, 768], cls 当作other

                res = torch.bmm(doc_att, aspect_emb).squeeze(
                    1)  # [128, 1, 768] -> [128, 768]

                return res, aspect_emb, doc_att

            else:
                print('====== not valid doc_pool_type!')
                raise ValueError
        else:
            raise ValueError

    def load(self, model_dir: str):
        pooler_path = os.path.join(model_dir, 'pooler.pt')
        if pooler_path is not None:
            if os.path.exists(pooler_path):
                logger.info(f'Loading Pooler from {pooler_path}')
                state_dict = torch.load(pooler_path, map_location='cpu')
                # self.load_state_dict(state_dict)
                self.load_state_dict_with_missing_keys_report(self, state_dict)
                self.load_state_dict(state_dict, strict=False)
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


class MADRModel(EncoderModel):
    def __init__(self,
                 lm_q: PreTrainedModel,
                 lm_p: PreTrainedModel,
                 aspect_fc_dict,
                 device,
                 model_args=None,
                 pooler: nn.Module = None,
                 untie_encoder: bool = False,
                 negatives_x_device: bool = False
                 ):
        nn.Module.__init__(self)
        self.lm_q = lm_q
        self.lm_p = lm_p
        self.brand_fc = aspect_fc_dict['brand_fc']
        self.color_fc = aspect_fc_dict['color_fc']
        self.cate1_fc = aspect_fc_dict['cate1_fc']
        self.cate2_fc = aspect_fc_dict['cate2_fc']
        self.cate3_fc = aspect_fc_dict['cate3_fc']
        self.cate4_fc = aspect_fc_dict['cate4_fc']
        self.cate5_fc = aspect_fc_dict['cate5_fc']
        self.cate_mul_fc = aspect_fc_dict['cate_mul_fc']

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
        self.device = device
        if model_args is not None:
            # if yes, return aspect label when encode
            self.is_eval_aspect = model_args.is_eval_aspect
            self.is_pretrain = model_args.is_pretrain

        gating_type = model_args.gating_type
        self.gating_type = gating_type

        self.bce = nn.BCEWithLogitsLoss()

    def encode_passage(self, psg, need_aspect_alpha=None):
        if psg is None:
            return None, None, None
        # 128=bs*train_n_passages
        # last_hidden_state:[128, 156, 768] , pooler_output: [128, 768]
        psg_out = self.lm_p(**psg, return_dict=True, output_hidden_states=True)
        p_hidden = psg_out.last_hidden_state
        if self.pooler is not None:
            if need_aspect_alpha:
                p_reps, aspect_emb, aspect_weight = self.pooler(
                    doc=p_hidden, doc_attention_mask=psg.attention_mask,
                    doc_hidden_states=psg_out.hidden_states,
                    need_aspect_alpha=need_aspect_alpha,
                    gating_type=self.gating_type)  # D * d
            else:
                p_reps, aspect_emb, aspect_weight = self.pooler(
                    doc=p_hidden, doc_attention_mask=psg.attention_mask,
                    doc_hidden_states=psg_out.hidden_states)  # D * d
        else:
            p_reps = p_hidden[:, 0]
            aspect_emb = None
            aspect_weight = None
        return p_reps, aspect_emb, aspect_weight

    def encode_query(self, qry, need_aspect_alpha=None):
        if qry is None:
            return None, None, None
        qry_out = self.lm_q(**qry, return_dict=True)
        q_hidden = qry_out.last_hidden_state  # [bs, seq_len, dim]
        if self.pooler is not None:
            if need_aspect_alpha:
                q_reps, aspect_emb, aspect_weight = self.pooler(
                    q=q_hidden, q_attention_mask=qry.attention_mask,
                    need_aspect_alpha=need_aspect_alpha,
                    gating_type=self.gating_type)
            else:
                q_reps, aspect_emb, aspect_weight = self.pooler(
                    q=q_hidden, q_attention_mask=qry.attention_mask)
        else:
            q_reps = q_hidden[:, 0]
            aspect_emb = None
            aspect_weight = None
        return q_reps, aspect_emb, aspect_weight

    def compute_similarity(self, q_reps, p_reps):
        return torch.matmul(q_reps, p_reps.transpose(0, 1))


    @staticmethod
    def load_pooler(model_weights_file, **config):
        pooler = MADRPooler(**config)
        pooler.load(model_weights_file)
        return pooler

    @staticmethod
    def build_pooler(model_args):

        pooler = MADRPooler(
            model_args.projection_in_dim,
            model_args.projection_out_dim,
            q_pool_type=model_args.q_pool_type,
            doc_pool_type=model_args.doc_pool_type,
            tied=not model_args.untie_encoder,
            aspect_num=model_args.aspect_num,
            begin_id=model_args.begin_id,
        )  # 如果前面共享的encoder，那么投影层就不共享；如果是两个encoder，就需要相同的投影层来到一个空间

        # 不加载pre-train的
        pooler.load(model_args.model_name_or_path)
        # logger.info("not load madr pooler from pre-trained model")
        return pooler

    @classmethod
    def build(
            cls,
            model_args: ModelArguments,
            train_args: TrainingArguments,
            **hf_kwargs,
    ):
        cls.forward_step = 0
        cls.finetune_aspect_alpha = train_args.finetune_aspect_alpha
        cls.aspect_loss_type = model_args.aspect_loss_type
        cls.zeroid_dict = {
            'p_brand': train_args.brand_zero_id,
            'p_color': train_args.color_zero_id,
            'p_cate1': train_args.cat1_zero_id,
            'p_cate2': train_args.cat2_zero_id,
            'p_cate3': train_args.cat3_zero_id,
            'p_cate4': train_args.cat4_zero_id,
            'p_cate5': train_args.cat5_zero_id,
            'p_cate_mul': train_args.cat_mul_zero_id,
        }
        # load local
        if os.path.isdir(model_args.model_name_or_path):
            if model_args.untie_encoder:  # 不共享bert
                if model_args.query_model_name_or_path != 'no':
                    _qry_model_path = model_args.query_model_name_or_path
                else:
                    _qry_model_path = os.path.join(
                        model_args.model_name_or_path, 'query_model')
                    if not os.path.exists(_qry_model_path):
                        _qry_model_path = model_args.model_name_or_path
                if model_args.doc_model_name_or_path != 'no':
                    _psg_model_path = model_args.doc_model_name_or_path
                else:
                    _psg_model_path = os.path.join(
                        model_args.model_name_or_path, 'passage_model')
                    if not os.path.exists(_psg_model_path):
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

        brand_fc = nn.Linear(model_args.projection_out_dim,
                             train_args.brand2id_dict_len)
        color_fc = nn.Linear(model_args.projection_out_dim,
                             train_args.color2id_dict_len)
        cate1_fc = nn.Linear(model_args.projection_out_dim,
                             train_args.cat1_2id_dict_len)
        cate2_fc = nn.Linear(model_args.projection_out_dim,
                             train_args.cat2_2id_dict_len)
        cate3_fc = nn.Linear(model_args.projection_out_dim,
                             train_args.cat3_2id_dict_len)
        cate4_fc = nn.Linear(model_args.projection_out_dim,
                             train_args.cat4_2id_dict_len)
        cate5_fc = nn.Linear(model_args.projection_out_dim,
                             train_args.cat5_2id_dict_len)
        cate_mul_fc = nn.Linear(model_args.projection_out_dim,
                                train_args.cat_mul2id_dict_len)

        model_name_or_path = model_args.model_name_or_path
        brand_fc_path = os.path.join(model_name_or_path, 'brand_fc.pt')
        color_fc_path = os.path.join(model_name_or_path, 'color_fc.pt')
        cate1_fc_path = os.path.join(model_name_or_path, 'cate1_fc.pt')
        cate2_fc_path = os.path.join(model_name_or_path, 'cate2_fc.pt')
        cate3_fc_path = os.path.join(model_name_or_path, 'cate3_fc.pt')
        cate4_fc_path = os.path.join(model_name_or_path, 'cate4_fc.pt')
        cate5_fc_path = os.path.join(model_name_or_path, 'cate5_fc.pt')
        cate_mul_fc_path = os.path.join(model_name_or_path, 'cate_mul_fc.pt')

        if os.path.exists(brand_fc_path):
            logger.info(f'found brand_fc')
            brand_fc.load_state_dict(torch.load(
                brand_fc_path, map_location='cpu'))
            color_fc.load_state_dict(torch.load(
                color_fc_path, map_location='cpu'))
            cate1_fc.load_state_dict(torch.load(
                cate1_fc_path, map_location='cpu'))
            cate2_fc.load_state_dict(torch.load(
                cate2_fc_path, map_location='cpu'))
            cate3_fc.load_state_dict(torch.load(
                cate3_fc_path, map_location='cpu'))
            cate4_fc.load_state_dict(torch.load(
                cate4_fc_path, map_location='cpu'))
            cate5_fc.load_state_dict(torch.load(
                cate5_fc_path, map_location='cpu'))
            if os.path.exists(cate_mul_fc_path):
                cate_mul_fc.load_state_dict(torch.load(
                    cate_mul_fc_path, map_location='cpu'))
            else:
                logger.info(f'not found cate_mul_fc!')
        else:
            logger.info(f'not found brand_fc!')

        aspect_fc_dict = {
            'brand_fc': brand_fc,
            'color_fc': color_fc,
            'cate1_fc': cate1_fc,
            'cate2_fc': cate2_fc,
            'cate3_fc': cate3_fc,
            'cate4_fc': cate4_fc,
            'cate5_fc': cate5_fc,
            'cate_mul_fc': cate_mul_fc,
        }

        model = cls(
            lm_q=lm_q,
            lm_p=lm_p,
            aspect_fc_dict=aspect_fc_dict,
            model_args=model_args,
            device=train_args.device,
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

        brand_fc_path = os.path.join(model_name_or_path, 'brand_fc.pt')
        color_fc_path = os.path.join(model_name_or_path, 'color_fc.pt')
        cate1_fc_path = os.path.join(model_name_or_path, 'cate1_fc.pt')
        cate2_fc_path = os.path.join(model_name_or_path, 'cate2_fc.pt')
        cate3_fc_path = os.path.join(model_name_or_path, 'cate3_fc.pt')
        cate4_fc_path = os.path.join(model_name_or_path, 'cate4_fc.pt')
        cate5_fc_path = os.path.join(model_name_or_path, 'cate5_fc.pt')
        cate_mul_fc_path = os.path.join(model_name_or_path, 'cate_mul_fc.pt')

        brand_fc = nn.Linear(model_args.projection_out_dim,
                             train_args.brand2id_dict_len)
        color_fc = nn.Linear(model_args.projection_out_dim,
                             train_args.color2id_dict_len)
        cate1_fc = nn.Linear(model_args.projection_out_dim,
                             train_args.cat1_2id_dict_len)
        cate2_fc = nn.Linear(model_args.projection_out_dim,
                             train_args.cat2_2id_dict_len)
        cate3_fc = nn.Linear(model_args.projection_out_dim,
                             train_args.cat3_2id_dict_len)
        cate4_fc = nn.Linear(model_args.projection_out_dim,
                             train_args.cat4_2id_dict_len)
        cate5_fc = nn.Linear(model_args.projection_out_dim,
                             train_args.cat5_2id_dict_len)
        cate_mul_fc = nn.Linear(model_args.projection_out_dim,
                                train_args.cat_mul2id_dict_len)

        if os.path.exists(brand_fc_path):
            logger.info(f'found brand_fc')
            brand_fc.load_state_dict(torch.load(
                brand_fc_path, map_location='cpu'))
            color_fc.load_state_dict(torch.load(
                color_fc_path, map_location='cpu'))
            cate1_fc.load_state_dict(torch.load(
                cate1_fc_path, map_location='cpu'))
            cate2_fc.load_state_dict(torch.load(
                cate2_fc_path, map_location='cpu'))
            cate3_fc.load_state_dict(torch.load(
                cate3_fc_path, map_location='cpu'))
            cate4_fc.load_state_dict(torch.load(
                cate4_fc_path, map_location='cpu'))
            cate5_fc.load_state_dict(torch.load(
                cate5_fc_path, map_location='cpu'))
            if os.path.exists(cate_mul_fc_path):
                cate_mul_fc.load_state_dict(torch.load(
                    cate_mul_fc_path, map_location='cpu'))
            else:
                logger.info(f'not found cate_mul_fc!')
        else:
            logger.info(f'not found brand_fc!')
            raise ValueError('not found brand_fc!')

        aspect_fc_dict = {
            'brand_fc': brand_fc,
            'color_fc': color_fc,
            'cate1_fc': cate1_fc,
            'cate2_fc': cate2_fc,
            'cate3_fc': cate3_fc,
            'cate4_fc': cate4_fc,
            'cate5_fc': cate5_fc,
            'cate_mul_fc': cate_mul_fc,
        }

        model = cls(
            lm_q=lm_q,
            lm_p=lm_p,
            model_args=model_args,
            aspect_fc_dict=aspect_fc_dict,
            device=None,
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

        aspect_fc_dict = {
            'brand_fc': self.brand_fc,
            'color_fc': self.color_fc,
            'cate1_fc': self.cate1_fc,
            'cate2_fc': self.cate2_fc,
            'cate3_fc': self.cate3_fc,
            'cate4_fc': self.cate4_fc,
            'cate5_fc': self.cate5_fc,
            'cate_mul_fc': self.cate_mul_fc,
        }
        for fc_name, fc_layer in aspect_fc_dict.items():
            if fc_layer:
                torch.save(fc_layer.state_dict(),
                           os.path.join(output_dir, '{}.pt'.format(fc_name)))

    def sum_loss_dict(self, loss_dict):
        all_loss = None
        for k, v in loss_dict.items():
            if all_loss is not None:
                all_loss += v
            else:
                all_loss = v.clone()
        return all_loss

    def forward(self, query: Dict[str, Tensor] = None, passage: Dict[str, Tensor] = None,
                aspect_id=None, doc_need_aspect_alpha=None):
        """
        query: {"input_ids": [bs, q_seq_len], "attention_mask": [bs, seq_len]}, eg, [32, 32]
        passage: {"input_ids": [bs*train_n_passages, p_seq_len], "attention_mask": [bs*train_n_passages, p_seq_len]} , eg, [64, 156]
        brand_id: [1, 11, 1672,..], len=bs*train_n_passages, eg, 64
        color_id: [5, 403, 5, 5, 14, 5,..], len=bs*train_n_passages, eg, 64
        """
        doc_need_aspect_alpha = {
            'p_brand': 1,
            'p_color': 2,
            'p_cate1': 0,
            'p_cate2': 0,
            'p_cate3': 0,
            'p_cate4': 0,
            'p_cate5': 0,
            'p_cate_mul': 3,
        }
        q_reps, query_aspect_emb, query_aspect_weight = self.encode_query(
            query, need_aspect_alpha=doc_need_aspect_alpha)  # [bs, 768]
        p_reps, doc_aspect_emb, doc_aspect_weight = self.encode_passage(
            passage, need_aspect_alpha=doc_need_aspect_alpha)  # [bs*train_n_passages, 768]

        fc_dict = {
            'p_brand': self.brand_fc,
            'p_color': self.color_fc,
            'p_cate1': self.cate1_fc,
            'p_cate2': self.cate2_fc,
            'p_cate3': self.cate3_fc,
            'p_cate4': self.cate4_fc,
            'p_cate5': self.cate5_fc,
            'p_cate_mul': self.cate_mul_fc,
        }
        if doc_aspect_emb is not None:  # doc 的池化方式是 att
            p_reps_for_aspect = {}
            for aspect_name, alpha in doc_need_aspect_alpha.items():
                if alpha != 0:
                    # 'p_brand': self.brand_fc(p_reps)
                    p_reps_for_aspect[aspect_name] = fc_dict[aspect_name](
                        doc_aspect_emb[:, alpha-1])
            assert len(p_reps_for_aspect) == (
                doc_aspect_emb.shape[1]-1)  # 有一个是other的

        # for inference, infer query 或者 passage, 对于pretrain阶段也是只有p_reps有值
        if (q_reps is None or p_reps is None) and self.is_pretrain == 'no':
            is_eval_final = False
            if is_eval_final:
                print('=====acc eval: final')
                if doc_aspect_emb is not None:
                    p_reps_for_aspect = {}
                    for aspect_name, alpha in doc_need_aspect_alpha.items():
                        if alpha != 0:
                            p_reps_for_aspect[aspect_name] = fc_dict[aspect_name](
                                p_reps)
            else:
                print('=====acc eval: separate')

            if self.is_eval_aspect == 'yes':
                doc_label_dict = {}
                doc_acc_dict = {}
                if doc_aspect_emb is not None:
                    for doc_aspect_name, alpha in doc_need_aspect_alpha.items():
                        if alpha != 0:
                            if doc_aspect_name == 'p_cate_mul':
                                doc_acc_dict[doc_aspect_name] = self.cal_acc_for_mul(aspect_id[doc_aspect_name].cpu(),
                                                                                     p_reps_for_aspect[doc_aspect_name].cpu().float(), is_recall=False, top_k=3)
                            else:
                                doc_label_dict[doc_aspect_name] = p_reps_for_aspect[doc_aspect_name].argmax(
                                    1).tolist()  # [doc_num]
                return EncoderOutput(
                    q_reps=q_reps,
                    p_reps=p_reps,
                    doc_label_dict=doc_label_dict,
                    doc_acc_dict=doc_acc_dict,
                )
            else:
                return EncoderOutput(
                    q_reps=q_reps,
                    p_reps=p_reps
                )

        # for training
        if self.training:
            for aspect_name, aspect_id_list in aspect_id.items():
                aspect_id[aspect_name] = torch.tensor(
                    aspect_id_list).to(self.device)

            if self.negatives_x_device:
                q_reps = self._dist_gather_tensor(q_reps)
                p_reps = self._dist_gather_tensor(p_reps)

                for aspect_name, aspect_id_tensor in aspect_id.items():  # 聚合有监督label
                    aspect_id[aspect_name] = self._dist_gather_tensor(
                        aspect_id_tensor)

                if doc_aspect_emb is not None:
                    for aspect_name, doc_aspect in p_reps_for_aspect.items():  # 聚合passage的表示
                        p_reps_for_aspect[aspect_name] = self._dist_gather_tensor(
                            doc_aspect)

            scores = self.compute_similarity(q_reps, p_reps)  # [32*4, 64*4]
            scores = scores.view(q_reps.size(0), -1)

            # [0,1,2,..., 32*4-1], [32*4,]
            target = torch.arange(scores.size(
                0), device=scores.device, dtype=torch.long)
            # * ( (64*4) // 32*4 ) = 2 = train_n_passages
            target = target * (p_reps.size(0) // q_reps.size(0))
            # [0, 2, 4, 6, ...] 这些是pos passage的 idx
            ori_loss = self.compute_loss(scores, target)

            new_loss = ori_loss.clone()
            aspect_loss_alpha = self.finetune_aspect_alpha  # aspect loss 整体的系数

            doc_app_loss_dict = {}
            doc_loss_dict = {}
            doc_acc_dict = {}
            doc_not_empty_aspect = 0
            if doc_aspect_emb is not None:
                for aspect_name, doc_aspect in p_reps_for_aspect.items():
                    p_reps_for_aspect[aspect_name] = doc_aspect.view(
                        p_reps.size(0), -1)

                for doc_aspect_name, alpha in doc_need_aspect_alpha.items():
                    if alpha != 0:
                        doc_not_empty_aspect += 1
                        aspect_loss_type = self.aspect_loss_type
                        if doc_aspect_name == 'p_cate_mul':
                            aspect_loss_type = 'multi-softmax'

                        doc_loss_dict[doc_aspect_name] = compute_aspect_loss(
                            p_reps_for_aspect[doc_aspect_name], aspect_id[doc_aspect_name],
                            aspect_loss_type, zero_id=self.zeroid_dict[doc_aspect_name])
                        if doc_aspect_name == 'p_cate_mul':
                            doc_acc_dict[doc_aspect_name] = self.cal_acc_for_mul(aspect_id[doc_aspect_name].cpu(),
                                                                                 p_reps_for_aspect[doc_aspect_name].cpu().float(), is_recall=False, top_k=3, is_train=True)
                        else:
                            doc_acc_dict[doc_aspect_name] = accuracy_score(
                                aspect_id[doc_aspect_name].cpu(), p_reps_for_aspect[doc_aspect_name].argmax(1).cpu().tolist())

                if doc_not_empty_aspect == 0:  # 针对aspect 数量取平均，防止被aspect 数量干扰alpha
                    doc_not_empty_aspect = 1

                all_doc_loss = self.sum_loss_dict(doc_loss_dict)
                doc_app_loss = self.sum_loss_dict(doc_app_loss_dict)

                if all_doc_loss is not None:
                    all_doc_loss = all_doc_loss / doc_not_empty_aspect
                    new_loss += aspect_loss_alpha * all_doc_loss
                    doc_loss_dict['avg_doc_loss'] = all_doc_loss
                
                if doc_app_loss is not None and aspect_loss_alpha != 0.0:
                    doc_app_loss = doc_app_loss / doc_not_empty_aspect
                    new_loss += aspect_loss_alpha * doc_app_loss
                    doc_loss_dict['app_loss'] = doc_app_loss

            if self.negatives_x_device:
                new_loss = new_loss * self.world_size
                self.forward_step += 1
        # for eval
        else:
            scores = self.compute_similarity(q_reps, p_reps)
            new_loss = None
            ori_loss = None
            doc_loss_dict = {}
            doc_acc_dict = {}

        return EncoderOutput(
            loss=new_loss,
            scores=scores,
            q_reps=q_reps,
            p_reps=p_reps,
            doc_loss_dict=doc_loss_dict,
            doc_acc_dict=doc_acc_dict,
            ori_loss=ori_loss,
            query_aspect_weight=query_aspect_weight,
            doc_aspect_weight=doc_aspect_weight,
            doc_need_aspect_alpha=doc_need_aspect_alpha,
        )
