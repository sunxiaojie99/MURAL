import logging
from .loss import SimpleContrastiveLoss, DistributedContrastiveLoss
import os
from itertools import repeat
from typing import Dict, List, Tuple, Optional, Any, Union

from transformers.trainer import Trainer, unwrap_model, MODEL_FOR_CAUSAL_LM_MAPPING_NAMES, is_torch_tpu_available, PREFIX_CHECKPOINT_DIR

import torch
from torch.utils.data import DataLoader
import torch.distributed as dist
# torch.autograd.set_detect_anomaly(True)


logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

try:
    from grad_cache import GradCache
    _grad_cache_available = True
except ModuleNotFoundError:
    _grad_cache_available = False


class TevatronTrainer(Trainer):
    """
    https://github.com/huggingface/transformers/blob/main/src/transformers/trainer.py
    """

    def __init__(self, *args, ttf_writer=None, **kwargs):
        super(TevatronTrainer, self).__init__(*args, **kwargs)
        self._dist_loss_scale_factor = dist.get_world_size(
        ) if self.args.negatives_x_device else 1
        self.ttf_writer = ttf_writer

    def _save(self, output_dir: Optional[str] = None):
        output_dir = output_dir if output_dir is not None else self.args.output_dir
        os.makedirs(output_dir, exist_ok=True)
        logger.info("Saving model checkpoint to %s", output_dir)
        self.model.save(output_dir)

    def _prepare_inputs(
            self,
            inputs: Tuple[Dict[str, Union[torch.Tensor, Any]], ...]
    ) -> List[Dict[str, Union[torch.Tensor, Any]]]:
        prepared = []
        for x in inputs:
            if isinstance(x, torch.Tensor):
                prepared.append(x.to(self.args.device))
            else:
                prepared.append(super()._prepare_inputs(x))
        return prepared

    def get_train_dataloader(self) -> DataLoader:
        if self.train_dataset is None:
            raise ValueError("Trainer: training requires a train_dataset.")
        train_sampler = self._get_train_sampler()

        return DataLoader(
            self.train_dataset,
            batch_size=self.args.train_batch_size,
            sampler=train_sampler,
            collate_fn=self.data_collator,
            drop_last=True,
            num_workers=self.args.dataloader_num_workers,
        )

    def compute_loss(self, model, inputs):
        # att_id: torch.Size([bs*train_n_passages, max_p_len])
        query, passage, brand_id, color_id,\
            cate1_id, cate2_id, cate3_id, cate4_id, cate5_id, cat_mul_id = inputs
        aspect_id = {
            'p_brand': brand_id,
            'p_color': color_id,
            'p_cate1': cate1_id,
            'p_cate2': cate2_id,
            'p_cate3': cate3_id,
            'p_cate4': cate4_id,
            'p_cate5': cate5_id,
            'p_cate_mul': cat_mul_id,
        }
        if self.args.model_type == 'mtbert':
            out = model(query=query, passage=passage,
                        aspect_id=aspect_id)

            if out.loss.device == torch.device('cuda:0') and self.ttf_writer is not None:
                if out.loss is not None:  # 原始的
                    self.ttf_writer.add_scalar("final_loss", out.loss.detach(
                    ) / self._dist_loss_scale_factor, self.state.global_step)
                if out.ori_loss is not None:  # 原始的
                    self.ttf_writer.add_scalar(
                        "ori_loss", out.ori_loss.detach(), self.state.global_step)
                for k, v in out.doc_loss_dict.items():  # 原始的
                    self.ttf_writer.add_scalar(
                        'doc_loss_'+k, v.detach(), self.state.global_step)
                for k, v in out.doc_acc_dict.items():
                    self.ttf_writer.add_scalar(
                        'doc_acc_'+k, v, self.state.global_step)

            return out.loss
        elif self.args.model_type == 'bibert':
            out = model(query=query, passage=passage)
            if out.loss.device == torch.device('cuda:0') and self.ttf_writer is not None:
                self.ttf_writer.add_scalar("final_loss", out.loss.detach(
                ) / self._dist_loss_scale_factor, self.state.global_step)
            return out.loss
        elif self.args.model_type == 'madr':

            out = model(query=query, passage=passage,
                        aspect_id=aspect_id
                        )
            # alpha 为0的不计算loss, 如果alpha=2，代表其对应的是第2个aspect emb
            doc_need_aspect_alpha = out.doc_need_aspect_alpha
            if out.loss.device == torch.device('cuda:0') and self.ttf_writer is not None:
                if out.loss is not None:  # 原始的
                    self.ttf_writer.add_scalar("final_loss", out.loss.detach(
                    ) / self._dist_loss_scale_factor, self.state.global_step)
                if out.ori_loss is not None:  # 原始的
                    self.ttf_writer.add_scalar(
                        "ori_loss", out.ori_loss.detach(), self.state.global_step)
                for k, v in out.doc_loss_dict.items():  # 原始的
                    self.ttf_writer.add_scalar(
                        k, v.detach(), self.state.global_step)
                for k, v in out.doc_acc_dict.items():
                    self.ttf_writer.add_scalar(
                        'doc_acc_'+k, v, self.state.global_step)
                if out.doc_aspect_weight is not None:
                    for aspect_name, alpha in doc_need_aspect_alpha.items():
                        if alpha != 0:
                            self.ttf_writer.add_histogram(
                                'doc_{}_aspect_weight_{}'.format(aspect_name, alpha-1), out.doc_aspect_weight[:, 0, alpha-1])
                    self.ttf_writer.add_histogram(
                        'doc_{}_aspect_weight'.format('other'), out.doc_aspect_weight[:, 0, -1])
                if out.query_aspect_weight is not None:
                    for aspect_name, alpha in doc_need_aspect_alpha.items():
                        if alpha != 0:
                            self.ttf_writer.add_histogram(
                                'query_{}_aspect_weight_{}'.format(aspect_name, alpha-1), out.query_aspect_weight[:, 0, alpha-1])
                    self.ttf_writer.add_histogram(
                        'query_{}_aspect_weight'.format('other'), out.query_aspect_weight[:, 0, -1])

                # debug for loss up
                if out.scores is not None:
                    self.ttf_writer.add_scalar(
                        "scores", torch.mean(out.scores), self.state.global_step)
                if out.p_reps is not None:
                    self.ttf_writer.add_scalar(
                        "doc_average_norm", torch.mean(torch.norm(out.p_reps, dim=1)), self.state.global_step)
                if out.q_reps is not None:
                    self.ttf_writer.add_scalar(
                        "q_average_norm", torch.mean(torch.norm(out.q_reps, dim=1)), self.state.global_step)
            return out.loss
        elif self.args.model_type == 'mgrain':
            out = model(query=query, passage=passage,)

            if out.loss.device == torch.device('cuda:0') and self.ttf_writer is not None:
                if out.loss is not None:  # 原始的
                    self.ttf_writer.add_scalar("final_loss", out.loss.detach(
                    ) / self._dist_loss_scale_factor, self.state.global_step)
                if out.doc_aspect_weight is not None:
                    for alpha in range(out.doc_aspect_weight.shape[-1]):
                        self.ttf_writer.add_histogram(
                            'doc_aspect_weight_{}'.format(alpha), out.doc_aspect_weight[:, 0, alpha])

                if out.query_aspect_weight is not None:
                    for alpha in range(out.query_aspect_weight.shape[-1]):
                        self.ttf_writer.add_histogram(
                            'query_aspect_weight_{}'.format(alpha), out.query_aspect_weight[:, 0, alpha])

                # debug for loss up
                if out.scores is not None:
                    self.ttf_writer.add_scalar(
                        "scores", torch.mean(out.scores), self.state.global_step)
                if out.p_reps is not None:
                    self.ttf_writer.add_scalar(
                        "doc_average_norm", torch.mean(torch.norm(out.p_reps, dim=1)), self.state.global_step)
                if out.q_reps is not None:
                    self.ttf_writer.add_scalar(
                        "q_average_norm", torch.mean(torch.norm(out.q_reps, dim=1)), self.state.global_step)
            return out.loss
        else:
            assert 'not valid model_type!'
            raise ValueError

    def training_step(self, *args):
        # 实际上定义了 forward和loss的计算过程
        # with torch.autograd.detect_anomaly():
        #     out = super(TevatronTrainer, self).training_step(
        #         *args) / self._dist_loss_scale_factor
        out = super(TevatronTrainer, self).training_step(
            *args) / self._dist_loss_scale_factor
        # for name, parms in args[0].named_parameters():
        #     if parms.grad is not None:
        #         print('-->name:', name, '-->grad_requirs:', parms.requires_grad, '--weight', torch.mean(parms.data), ' -->grad_value:', torch.mean(parms.grad))
        for name, parms in args[0].named_parameters():
            if self.ttf_writer is not None:
                if name == 'module.pooler.att_query_q' and parms.grad is not None:
                    self.ttf_writer.add_histogram(
                        'pooler.att_query_q.grad', parms.grad)
                if name == 'module.pooler.att_query_d' and parms.grad is not None:
                    self.ttf_writer.add_histogram(
                        'pooler.att_query_d.grad', parms.grad)

        return out


def split_dense_inputs(model_input: dict, chunk_size: int):
    assert len(model_input) == 1
    arg_key = list(model_input.keys())[0]
    arg_val = model_input[arg_key]

    keys = list(arg_val.keys())
    chunked_tensors = [arg_val[k].split(chunk_size, dim=0) for k in keys]
    chunked_arg_val = [dict(zip(kk, tt))
                       for kk, tt in zip(repeat(keys), zip(*chunked_tensors))]

    return [{arg_key: c} for c in chunked_arg_val]


def get_dense_rep(x):
    if x.q_reps is None:
        return x.p_reps
    else:
        return x.q_reps


class GCTrainer(TevatronTrainer):
    def __init__(self, *args, **kwargs):
        logger.info('Initializing Gradient Cache Trainer')
        if not _grad_cache_available:
            raise ValueError(
                'Grad Cache package not available. You can obtain it from https://github.com/luyug/GradCache.')
        super(GCTrainer, self).__init__(*args, **kwargs)

        loss_fn_cls = DistributedContrastiveLoss if self.args.negatives_x_device else SimpleContrastiveLoss
        loss_fn = loss_fn_cls()

        self.gc = GradCache(
            models=[self.model, self.model],
            chunk_sizes=[self.args.gc_q_chunk_size, self.args.gc_p_chunk_size],
            loss_fn=loss_fn,
            split_input_fn=split_dense_inputs,
            get_rep_fn=get_dense_rep,
            fp16=self.args.fp16,
            scaler=self.scaler if self.args.fp16 else None
        )

    def training_step(self, model, inputs) -> torch.Tensor:
        model.train()
        queries, passages = self._prepare_inputs(inputs)
        queries, passages = {'query': queries}, {'passage': passages}

        _distributed = self.args.local_rank > -1
        self.gc.models = [model, model]
        loss = self.gc(queries, passages, no_sync_except_last=_distributed)

        return loss / self._dist_loss_scale_factor


class MLMTrainer(Trainer):
    def __init__(self, *args, ttf_writer=None, **kwargs):
        super(MLMTrainer, self).__init__(*args, **kwargs)
        self.ttf_writer = ttf_writer

    def _prepare_inputs(
            self,
            inputs):
        new_inputs = {}
        for x_name, x_t in inputs.items():
            if isinstance(x_t, torch.Tensor):
                new_inputs[x_name] = x_t.to(self.args.device)
            else:
                new_inputs[x_name] = x_t
        return new_inputs

    def _maybe_log_save_evaluate(self, tr_loss, model, trial, epoch, ignore_keys_for_eval):
        """
        https://github.com/huggingface/transformers/blob/main/src/transformers/trainer.py#L2103
        """
        if self.control.should_log:
            if is_torch_tpu_available():
                xm.mark_step()

            logs: Dict[str, float] = {}

            # all_gather + mean() to get average loss over all processes
            tr_loss_scalar = self._nested_gather(tr_loss).mean().item()

            # reset tr_loss to zero
            tr_loss -= tr_loss

            logs["loss"] = round(
                tr_loss_scalar / (self.state.global_step - self._globalstep_last_logged), 4)
            logs["learning_rate"] = self._get_learning_rate()

            self._total_loss_scalar += tr_loss_scalar
            self._globalstep_last_logged = self.state.global_step
            self.store_flos()

            self.log(logs)

        metrics = None
        if self.control.should_evaluate:
            if isinstance(self.eval_dataset, dict):
                for eval_dataset_name, eval_dataset in self.eval_dataset.items():
                    metrics = self.evaluate(
                        eval_dataset=eval_dataset,
                        ignore_keys=ignore_keys_for_eval,
                        metric_key_prefix=f"eval_{eval_dataset_name}",
                    )
            else:
                metrics = self.evaluate(ignore_keys=ignore_keys_for_eval)
            self._report_to_hp_search(trial, self.state.global_step, metrics)

        if self.control.should_save:
            self._save_checkpoint(model, trial, metrics=metrics)

            # add: Save additional model checkpoint
            if hasattr(model.module, 'additional_save'):
                checkpoint_folder = f"{PREFIX_CHECKPOINT_DIR}-{self.state.global_step}"
                run_dir = self._get_output_dir(trial=trial)
                output_dir = os.path.join(run_dir, checkpoint_folder)
                model.module.additional_save(
                    output_dir)  # for DataParallel model

            self.control = self.callback_handler.on_save(
                self.args, self.state, self.control)

    def compute_loss(self, model, inputs, return_outputs=False):
        """
        This is mostly copied from the Transformers trainer
        How the loss is computed by Trainer. By default, all models return the loss in the first element.
        Subclass and override for custom behavior.

        inputs: dict_keys(['input_ids', 'attention_mask', 'labels', 'aspect_dict'])
        """
        if self.label_smoother is not None and "labels" in inputs:
            labels = inputs.pop("labels")
        else:
            labels = None
        # for mlm
        if 'aspect_dict' not in inputs or inputs['aspect_dict'] is None:
            if 'aspect_dict' in inputs:
                inputs.pop("aspect_dict")
            outputs = model(**inputs)
        else:
            outputs = model(**inputs, step=self.state.global_step)
        if 'loss' in outputs and outputs.loss.device == torch.device('cuda:0'):
            if 'doc_loss_dict' in outputs:
                for k, v in outputs.doc_loss_dict.items():  # 原始的
                    if v is not None and v.shape != torch.Size([]):
                        if len(v) != 1:  # 多卡gather了
                            # https://github.com/huggingface/transformers/blob/main/src/transformers/trainer.py#L2556
                            v = v.mean()
                        self.ttf_writer.add_scalar(
                            'doc_loss_'+k, v.detach(), self.state.global_step)

        # Save past state if it exists
        # TODO: this needs to be fixed and made cleaner later.
        if self.args.past_index >= 0:
            self._past = outputs[self.args.past_index]

        if labels is not None:
            if unwrap_model(model)._get_name() in MODEL_FOR_CAUSAL_LM_MAPPING_NAMES.values():
                loss = self.label_smoother(outputs, labels, shift_labels=True)
            else:
                loss = self.label_smoother(outputs, labels)
        else:
            if isinstance(outputs, dict) and "loss" not in outputs:
                raise ValueError(
                    "The model did not return a loss from the inputs, only the following keys: "
                    f"{','.join(outputs.keys())}. For reference, the inputs it received are {','.join(inputs.keys())}."
                )
            # We don't use .loss here since the model may return tuples instead of ModelOutput.
            loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]

        return (loss, outputs) if return_outputs else loss
