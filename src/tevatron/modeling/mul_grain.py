import torch
import torch.nn as nn
from torch import Tensor
import torch.nn.functional as F
import torch.distributed as dist
from transformers import PreTrainedModel, AutoModel
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


class MULGRAIN_Pooler(EncoderPooler):
    def __init__(self, input_dim: int = 768, out_dim: int = 768,
                 q_pool_type='att', doc_pool_type='att', tied=True, aspect_num=None, begin_id=0):
        super(MULGRAIN_Pooler, self).__init__()

        self.q_pool_type = q_pool_type
        self.doc_pool_type = doc_pool_type
        self.aspect_num = aspect_num
        self.begin_id = begin_id

        if q_pool_type == 'att':
            self.cls_linear_q = nn.Linear(input_dim, self.aspect_num)

        else:
            self.linear_q = nn.Linear(input_dim, out_dim)

        if doc_pool_type == 'att':
            if q_pool_type == 'att':  # 对于madr，pooler共享
                self.cls_linear_d = self.cls_linear_q
            else:
                self.cls_linear_d = nn.Linear(input_dim, self.aspect_num)
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
                # 1. 拆分 直接取前3个token emb
                begin_id = self.begin_id  # 从哪个tokenid开始取，0是cls
                aspect_emb = q[:, begin_id:begin_id+self.aspect_num]  # 直接选取 前3个token输出

                # 2.进入融合操作
                q_cls = q[:, 0]
                q_att = self.cls_linear_q(q_cls)  # [128, 3] 线性层变换到aspect大小维度
                q_att = torch.softmax(
                    q_att, dim=-1).unsqueeze(1)  # [128, 1, 3]
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
                # 1. 直接取前3个token emb
                begin_id = self.begin_id  # 从哪个tokenid开始取，0是cls
                aspect_emb = doc[:, begin_id:begin_id+self.aspect_num]  # 直接选取 前3个token输出

                # 2.进入融合操作，
                # 方式1:直接对cls变换到3维
                doc_cls = doc[:, 0]
                # [128, 3] 线性层变换到aspect大小维度
                doc_att = self.cls_linear_d(doc_cls)
                doc_att = torch.softmax(
                    doc_att, dim=-1).unsqueeze(1)  # [128, 1, 3]
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
                self.load_state_dict(state_dict)
                return
        logger.info("Training Pooler from scratch")
        return

    def save_pooler(self, save_path):
        torch.save(self.state_dict(), os.path.join(save_path, 'pooler.pt'))
        with open(os.path.join(save_path, 'pooler_config.json'), 'w') as f:
            json.dump(self._config, f)


class MULGRAIN_Model(EncoderModel):

    # TRANSFORMER_CLS = MGRAN_Model

    def __init__(self,
                 lm_q: PreTrainedModel,
                 lm_p: PreTrainedModel,
                 device,
                 model_args=None,
                 pooler: nn.Module = None,
                 untie_encoder: bool = False,
                 negatives_x_device: bool = False
                 ):
        nn.Module.__init__(self)
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
        self.device = device
        self.aspect_emb_dict = None
        if model_args is not None:
            # if yes, return aspect label when encode
            self.is_eval_aspect = model_args.is_eval_aspect
            if hasattr(model_args, 'aspect_emb_dict'):
                self.aspect_emb_dict = model_args.aspect_emb_dict
            

    def encode_passage(self, psg):
        if psg is None:
            return None, None, None
        # 128=bs*train_n_passages
        # last_hidden_state:[128, 156, 768] , pooler_output: [128, 768]
        psg_out = self.lm_p(**psg, return_dict=True, output_hidden_states=True)
        p_hidden = psg_out.last_hidden_state
        if self.pooler is not None:
            p_reps, aspect_emb, aspect_weight = self.pooler(
                doc=p_hidden, doc_attention_mask=psg.attention_mask,
                doc_hidden_states=psg_out.hidden_states)  # D * d
        else:
            p_reps = p_hidden[:, 0]
            aspect_emb = None
            aspect_weight = None
        return p_reps, aspect_emb, aspect_weight

    def encode_query(self, qry):
        if qry is None:
            return None, None, None
        qry_out = self.lm_q(**qry, return_dict=True)
        q_hidden = qry_out.last_hidden_state  # [bs, seq_len, dim]
        if self.pooler is not None:
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
        pooler = MULGRAIN_Pooler(**config)
        pooler.load(model_weights_file)
        return pooler

    @staticmethod
    def build_pooler(model_args):

        pooler = MULGRAIN_Pooler(
            model_args.projection_in_dim,
            model_args.projection_out_dim,
            q_pool_type=model_args.q_pool_type,
            doc_pool_type=model_args.doc_pool_type,
            tied=not model_args.untie_encoder,
            aspect_num=model_args.aspect_num,
            begin_id=model_args.begin_id,
        )  # 如果前面共享的encoder，那么投影层就不共享；如果是两个encoder，就需要相同的投影层来到一个空间

        # 不加载pre-train的
        # pooler.load(model_args.model_name_or_path)
        logger.info("not load madr pooler from pre-trained model")
        return pooler

    @classmethod
    def build(
            cls,
            model_args: ModelArguments,
            train_args: TrainingArguments,
            **hf_kwargs,
    ):
        cls.forward_step = 0
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

        model = cls(
            lm_q=lm_q,
            lm_p=lm_p,
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


        model = cls(
            lm_q=lm_q,
            lm_p=lm_p,
            model_args=model_args,
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

        q_reps, query_aspect_emb, query_aspect_weight = self.encode_query(
            query)  # [bs, 768]
        p_reps, doc_aspect_emb, doc_aspect_weight = self.encode_passage(
            passage)  # [bs*train_n_passages, 768]


        # for inference, infer query 或者 passage, 对于pretrain阶段也是只有p_reps有值
        if (q_reps is None or p_reps is None):
            doc_acc_dict = {}
            doc_label_dict = {}
            self.aspect_emb_dict = None  # == not eval
            if self.aspect_emb_dict is not None and p_reps is not None:

                is_eval_final = False
                if is_eval_final is False:
                    print('=====acc eval: different')
                    source_reps_dict = {
                        'p_cate_wholeword': doc_aspect_emb[:,1,:],
                        'p_brand_wholeword': doc_aspect_emb[:,1,:],
                        'p_color_wholeword': doc_aspect_emb[:,1,:],

                        'p_cate_word': doc_aspect_emb[:,2,:],
                        'p_brand_word': doc_aspect_emb[:,2,:],
                        'p_color_word': doc_aspect_emb[:,2,:],

                        'p_cate_wordpiece': doc_aspect_emb[:,3,:],
                        'p_brand_wordpiece': doc_aspect_emb[:,3,:],
                        'p_color_wordpiece': doc_aspect_emb[:,3,:],
                    }
                else:
                    print('=====acc eval: final')
                    source_reps_dict = {
                        'p_cate_wholeword': p_reps,
                        'p_brand_wholeword': p_reps,
                        'p_color_wholeword': p_reps,

                        'p_cate_word': p_reps,
                        'p_brand_word': p_reps,
                        'p_color_word': p_reps,

                        'p_cate_wordpiece': p_reps,
                        'p_brand_wordpiece': p_reps,
                        'p_color_wordpiece': p_reps,
                    }

                aspect_names = self.aspect_emb_dict.keys()
                for _name in aspect_names:
                    aspect_rep = self.aspect_emb_dict[_name]  # [xxx, 768]
                    sim_score = torch.matmul(source_reps_dict[_name], aspect_rep.T)
                    labels = aspect_id[_name]
                    
                    if _name in ['p_brand_wholeword', 'p_color_wholeword']:
                        # doc_acc_dict[_name] = accuracy_score(
                        #         labels, sim_score.argmax(1).cpu().tolist())
                        doc_label_dict[_name] = sim_score.argmax(1).cpu().tolist()  # [doc_num]
                    else:
                        assert sim_score.shape == labels.shape, print(sim_score.shape, labels.shape, _name)
                        labels = torch.tensor(labels)
                        doc_acc_dict[_name] = self.cal_acc_for_mul(labels, sim_score.cpu().float(), is_recall=False, top_k=3)
            
            return EncoderOutput(
                q_reps=q_reps,
                p_reps=p_reps,
                doc_acc_dict=doc_acc_dict,
                doc_label_dict=doc_label_dict
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
            ori_loss = self.compute_loss(scores, target)

            new_loss = ori_loss

            doc_loss_dict = {}
            doc_acc_dict = {}

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
        )
