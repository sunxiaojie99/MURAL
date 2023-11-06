#!/usr/bin/env python
# coding=utf-8
# Copyright 2020 The HuggingFace Team All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Fine-tuning the library models for masked language modeling (BERT, ALBERT, RoBERTa...) on a text file or a dataset.

Here is the full list of checkpoints on the hub that can be fine-tuned by this script:
https://huggingface.co/models?filter=fill-mask
"""
# You can also adapt this script on your own masked language modeling task. Pointers for this are left as comments.

import logging
import math
import os
import sys

import datasets
import torch

import evaluate
import transformers
from transformers import (
    CONFIG_MAPPING,
    AutoConfig,
    AutoModelForMaskedLM,
    AutoTokenizer,
    # DataCollatorForLanguageModeling,
    HfArgumentParser,
    Trainer,
    # TrainingArguments,
    is_torch_tpu_available,
    set_seed,
)
from transformers.trainer_utils import get_last_checkpoint
# from transformers.utils import check_min_version, send_example_telemetry
# from transformers.utils.versions import require_version

from tevatron.arguments import ModelArguments, DataArguments, \
    TevatronTrainingArguments as TrainingArguments
from tevatron.data import HFPretrainDataset, PretrainDataset, PreTrainCollator
from tevatron.trainer import MLMTrainer
from tevatron.modeling import BertForAspectMaskedLM, MtbertMaskLM, MadrMaskLM, \
    BertForMaskedLM_Same, CondenserForPretraining, \
    MGrain_MaskedLM
from torch.utils.tensorboard import SummaryWriter
from tevatron.config import get_cfg
from tevatron.data import load_dict, padding_for_max_len


# Will error if the minimal version of Transformers is not installed. Remove at your own risks.
# check_min_version("4.25.0.dev0")
# require_version("datasets>=1.8.0",
#                 "To fix: pip install -r examples/pytorch/language-modeling/requirements.txt")

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


def main():
    # See all possible arguments in src/transformers/training_args.py
    # or by passing the --help flag to this script.
    # We now keep distinct sets of args, for a cleaner separation of concerns.

    parser = HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        model_args, data_args, training_args = parser.parse_json_file(
            json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()
        model_args: ModelArguments
        data_args: DataArguments
        training_args: TrainingArguments

    if data_args.config_path is not None:
        cfg = get_cfg(data_args.config_path)
        need_cfg = [cfg.common, cfg.pretrain]
        for cfg_dict in need_cfg:
            for k, v in cfg_dict.items():
                if hasattr(model_args, k):
                    setattr(model_args, k, v)
                elif hasattr(data_args, k):
                    setattr(data_args, k, v)
                elif hasattr(training_args, k):
                    setattr(training_args, k, v)
                else:
                    raise ValueError('not find args!', k, v)
        data_args.__post_init__()

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    # Set the verbosity to info of the Transformers logger (on main process only):
    logger.info(f"Training/evaluation parameters {training_args}")
    logger.info("MODEL parameters %s", model_args)
    logger.info("Data parameters %s", data_args)

    # Detecting last checkpoint.
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir) and training_args.do_train and not training_args.overwrite_output_dir:
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )
        elif last_checkpoint is not None and training_args.resume_from_checkpoint is None:
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )

    set_seed(training_args.seed)

    # Load pretrained model and tokenizer
    # Distributed training:
    # The .from_pretrained methods guarantee that only one local process can concurrently
    # download model & vocab.
    config_kwargs = {
        "cache_dir": model_args.cache_dir,
        # "revision": model_args.model_revision,
    }
    if model_args.config_name:
        config = AutoConfig.from_pretrained(
            model_args.config_name, **config_kwargs)
    elif model_args.model_name_or_path:
        config = AutoConfig.from_pretrained(
            model_args.model_name_or_path, **config_kwargs)
    else:
        config = CONFIG_MAPPING[model_args.model_type]()
        logger.warning(
            "You are instantiating a new config instance from scratch.")
        if model_args.config_overrides is not None:
            logger.info(f"Overriding config: {model_args.config_overrides}")
            config.update_from_string(model_args.config_overrides)
            logger.info(f"New config: {config}")

    tokenizer_kwargs = {
        "cache_dir": model_args.cache_dir,
        "use_fast": model_args.use_fast_tokenizer,
    }
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path,
        **tokenizer_kwargs)
    tokenizer.add_special_tokens(
        {"additional_special_tokens": ["[unused0]", "[unused1]", "[unused2]", "[unused3]", "[SEP]", "[unused4]", \
            "[unused5]", "[unused6]", "[unused7]", "[unused8]", "[unused9]", "[unused10]", "[unused11]"]})
    

    if data_args.pretrain_type == 'mgrain' and model_args.is_shared_mgrain == 'yes':
        bert_vocab = tokenizer.get_vocab()

        # wordpiece
        wordpiece_dict = {
            'cate': data_args.cate_wordpiece_vocab_file,
            'brand': data_args.brand_wordpiece_vocab_file,
            'color': data_args.color_wordpiece_vocab_file,
        }
        wordpiece2token_id = {}
        for k in wordpiece_dict.keys():
            cate_wordpiece_vocab_dict = load_dict(wordpiece_dict[k])
            cate_wordpiece_dict_id2word = {v:k for k, v in cate_wordpiece_vocab_dict.items()}
            all_text_id = []  # 每个piece对应的wordid
            for word_id in range(len(cate_wordpiece_vocab_dict)):
                cate_text = cate_wordpiece_dict_id2word[word_id]
                all_text_id.append(bert_vocab[cate_text])
            
            wordpiece2token_id[k] = torch.tensor(all_text_id)
        data_args.cate_wordpiece2token_id = wordpiece2token_id['cate']
        data_args.brand_wordpiece2token_id = wordpiece2token_id['brand']
        data_args.color_wordpiece2token_id = wordpiece2token_id['color']

        # word
        word_dict = {
            'cate': data_args.cate_word_vocab_file,
            'brand': data_args.brand_word_vocab_file,
            'color': data_args.color_word_vocab_file,
        }
        word2token_ids = {}
        word_pad_mask = {}
        for k in word_dict.keys():
            cate_word_vocab_dict = load_dict(word_dict[k])
            cate_word_dict_id2word = {v:k for k, v in cate_word_vocab_dict.items()}
            all_text = []
            for word_id in range(len(cate_word_vocab_dict)):
                cate_text = cate_word_dict_id2word[word_id]
                all_text.append(cate_text)
            padding_out = tokenizer(all_text, padding='longest', return_tensors='pt', return_special_tokens_mask=True)  # dict_keys(['input_ids', 'token_type_ids', 'attention_mask', 'special_tokens_mask'])
            word2token_ids[k] = padding_out['input_ids']  # [7050, 8]
            word_pad_mask[k] = padding_out['attention_mask'] * (1-padding_out['special_tokens_mask'])  # 把padding和cls、sep去掉
        data_args.cate_word2token_ids = word2token_ids['cate']
        data_args.cate_word_pad_mask = word_pad_mask['cate']
        data_args.brand_word2token_ids = word2token_ids['brand']
        data_args.brand_word_pad_mask = word_pad_mask['brand']
        data_args.color_word2token_ids = word2token_ids['color']
        data_args.color_word_pad_mask = word_pad_mask['color']
        

        # wholeword
        wholeword_dict = {
            'cate': data_args.whole_cate_vocab_file,
            'brand': data_args.whole_brand_vocab_file,
            'color': data_args.whole_color_vocab_file,
        }
        wholeword2token_ids = {}
        wholeword_pad_mask = {}
        for k in wholeword_dict.keys():
            whole_cate_vocab_dict = load_dict(wholeword_dict[k])
            whole_cate_vocab_dict_id2word = {v:k for k, v in whole_cate_vocab_dict.items()}
            all_text = []
            for word_id in range(len(whole_cate_vocab_dict)):
                cate_text = whole_cate_vocab_dict_id2word[word_id]
                all_text.append(cate_text)
            padding_out = tokenizer(all_text, padding='longest', return_tensors='pt', return_special_tokens_mask=True)  # dict_keys(['input_ids', 'token_type_ids', 'attention_mask', 'special_tokens_mask'])
            wholeword2token_ids[k] = padding_out['input_ids']  # [12896, 18]
            wholeword_pad_mask[k] = padding_out['attention_mask'] * (1-padding_out['special_tokens_mask'])  # 把padding和cls、sep去掉
        data_args.cate_wholeword2token_ids = wholeword2token_ids['cate']
        data_args.cate_wholeword_pad_mask = wholeword_pad_mask['cate']
        data_args.brand_wholeword2token_ids = wholeword2token_ids['brand']
        data_args.brand_wholeword_pad_mask = wholeword_pad_mask['brand']
        data_args.color_wholeword2token_ids = wholeword2token_ids['color']
        data_args.color_wholeword_pad_mask = wholeword_pad_mask['color']
    
    elif data_args.pretrain_type == 'mgrain' and model_args.is_shared_mgrain == 'no':
        # 不同粒度的词表 是怎么对应bert中的token id的
        bert_vocab = tokenizer.get_vocab()
        # wordpiece
        wordpiece_dict = {
            'cate': data_args.cate_wordpiece_vocab_file,
            'brand': data_args.brand_wordpiece_vocab_file,
            'color': data_args.color_wordpiece_vocab_file,
        }
        wordpiece2token_id = {}
        for k in wordpiece_dict.keys():
            cate_wordpiece_vocab_dict = load_dict(wordpiece_dict[k])
            cate_wordpiece_dict_id2word = {v:k for k, v in cate_wordpiece_vocab_dict.items()}
            all_text_id = []  # 每个piece对应的wordid
            for word_id in range(len(cate_wordpiece_vocab_dict)):
                cate_text = cate_wordpiece_dict_id2word[word_id]
                all_text_id.append([bert_vocab[cate_text]])
            wordpiece2token_id[k] = torch.tensor(all_text_id)
        data_args.cate_wordpiece2token_id = wordpiece2token_id['cate']
        data_args.brand_wordpiece2token_id = wordpiece2token_id['brand']
        data_args.color_wordpiece2token_id = wordpiece2token_id['color']

        # word
        word_dict = {
            'cate': data_args.cate_word_vocab_file,
            'brand': data_args.brand_word_vocab_file,
            'color': data_args.color_word_vocab_file,
        }
        word2token_ids = {}
        for k in word_dict.keys():
            cate_word_vocab_dict = load_dict(word_dict[k])
            cate_word_vocab_dict_id2word = {v:k for k, v in cate_word_vocab_dict.items()}
            cate_word2token_ids = []
            for word_id in range(len(cate_word_vocab_dict)):
                cate_text = cate_word_vocab_dict_id2word[word_id]
                token_ids = tokenizer.encode(cate_text)[1:-1]
                cate_word2token_ids.append(token_ids)  # 把每个词对应的token id 组织成list，放进去
            word2token_ids[k] = cate_word2token_ids
        data_args.cate_word2token_ids = word2token_ids['cate']
        data_args.brand_word2token_ids = word2token_ids['brand']
        data_args.color_word2token_ids = word2token_ids['color']

        # wholeword
        wholeword_dict = {
            'cate': data_args.whole_cate_vocab_file,
            'brand': data_args.whole_brand_vocab_file,
            'color': data_args.whole_color_vocab_file,
        }
        wholeword2token_ids = {}
        for k in wholeword_dict.keys():
            cate_wholeword_vocab_dict = load_dict(wholeword_dict[k])
            cate_wholeword_vocab_dict_id2word = {v:k for k, v in cate_wholeword_vocab_dict.items()}
            cate_word2token_ids = []
            for word_id in range(len(cate_wholeword_vocab_dict_id2word)):
                cate_text = cate_wholeword_vocab_dict_id2word[word_id]
                token_ids = tokenizer.encode(cate_text)[1:-1]
                cate_word2token_ids.append(token_ids)  # 把每个词对应的token id 组织成list，放进去
            wholeword2token_ids[k] = cate_word2token_ids
        data_args.cate_wholeword2token_ids = wholeword2token_ids['cate']  # 12895
        data_args.brand_wholeword2token_ids = wholeword2token_ids['brand'] 
        data_args.color_wholeword2token_ids = wholeword2token_ids['color']

    if training_args.device == torch.device('cuda:0'):
        writer = SummaryWriter(model_args.tensorboard_output_dir)
    else:
        writer = None

    model_dict = {
        'mtbert': MtbertMaskLM,
        'madr': MadrMaskLM,
        'prompt': BertForAspectMaskedLM,
        'mlm': BertForMaskedLM_Same,
        'condenser': CondenserForPretraining,
        'mgrain': MGrain_MaskedLM,
    }

    if data_args.pretrain_type in model_dict:
        MaskLM = model_dict[data_args.pretrain_type]
    else:  # mlm
        MaskLM = AutoModelForMaskedLM

    if model_args.model_name_or_path:
        if data_args.pretrain_type in model_dict:
            model = MaskLM.from_pretrained(
                model_args.model_name_or_path,
                from_tf=bool(".ckpt" in model_args.model_name_or_path),
                config=config,
                cache_dir=model_args.cache_dir,
                data_args=data_args,
                training_args=training_args,
                model_args=model_args,
                tf_writer=writer,
                # revision=model_args.model_revision,
                # use_auth_token=True if model_args.use_auth_token else None,
            )
        else:
            model = MaskLM.from_pretrained(
                model_args.model_name_or_path,
                from_tf=bool(".ckpt" in model_args.model_name_or_path),
                config=config,
                cache_dir=model_args.cache_dir,
                # revision=model_args.model_revision,
                # use_auth_token=True if model_args.use_auth_token else None,
            )
        if hasattr(model, 'new_post_init'):
            print('====== new_post_init')
            model.new_post_init()
    else:
        logger.info("Training new model from scratch")
        model = MaskLM.from_config(config)
    
    if data_args.pretrain_type == 'wordpredict':
        # model.bert.get_input_embeddings().weight.size(): [30522, 768] -> [33610, 768]
        # model.bert.get_input_embeddings().weight[0][:5]
        model.bert.resize_token_embeddings(len(tokenizer))
        model.config.vocab_size = len(new_vocab_dict)
        model.post_init_emb()
    elif data_args.pretrain_type in ['wordpredict_no', 'multi_predict_not_shared', 'mgrain']:
        # print('====== not bert-init')
        model.post_init_emb()

    hfpretrain_dataset = HFPretrainDataset(
        tokenizer=tokenizer, data_args=data_args,
        cache_dir=data_args.data_cache_dir or model_args.cache_dir)
    eval_dataset = None
    if training_args.local_rank > 0:
        print("Waiting for main process to perform the mapping")
        torch.distributed.barrier()

    pretrain_dataset = PretrainDataset(
        data_args, hfpretrain_dataset.process(), tokenizer)
    if training_args.local_rank == 0:
        print("Loading results from main process")
        torch.distributed.barrier()

    # We resize the embeddings only when necessary to avoid index errors. If you are creating a model from scratch
    # on a small vocab and want a smaller embedding size, remove this test.
    embedding_size = model.get_input_embeddings().weight.shape[0]
    if len(tokenizer) > embedding_size:
        print('!!!!!!!!!resize_token_embeddings')
        model.resize_token_embeddings(len(tokenizer))

    # Data collator
    # This one will take care of randomly masking the tokens.
    pad_to_multiple_of_8 = training_args.fp16 and not data_args.pad_to_max_length
    data_collator = PreTrainCollator(data_args=data_args, tokenizer=tokenizer,
                                     pad_to_multiple_of=8 if pad_to_multiple_of_8 else None,)

    def preprocess_logits_for_metrics(logits, labels):
        if isinstance(logits, tuple):
            # Depending on the model and config, logits may contain extra tensors,
            # like past_key_values, but logits always come first
            logits = logits[0]
        return logits.argmax(dim=-1)

    metric = evaluate.load(
        "/path/downloads/tmp/evaluate-main/metrics/accuracy/accuracy.py")

    def compute_metrics(eval_preds):
        preds, labels = eval_preds
        # preds have the same shape as the labels, after the argmax(-1) has been calculated
        # by preprocess_logits_for_metrics
        labels = labels.reshape(-1)
        preds = preds.reshape(-1)
        mask = labels != -100
        labels = labels[mask]
        preds = preds[mask]
        return metric.compute(predictions=preds, references=labels)

    # Initialize our Trainer
    TTrainer = MLMTrainer

    trainer = TTrainer(
        model=model,
        args=training_args,
        train_dataset=pretrain_dataset if training_args.do_train else None,
        eval_dataset=eval_dataset if training_args.do_eval else None,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics if training_args.do_eval and not is_torch_tpu_available() else None,
        preprocess_logits_for_metrics=preprocess_logits_for_metrics
        if training_args.do_eval and not is_torch_tpu_available()
        else None,
        ttf_writer=writer,
    )

    # Training
    if training_args.do_train:
        checkpoint = None
        if training_args.resume_from_checkpoint is not None:
            checkpoint = training_args.resume_from_checkpoint
        elif last_checkpoint is not None:
            checkpoint = last_checkpoint
        train_result = trainer.train(resume_from_checkpoint=checkpoint)
        trainer.save_model()  # Saves the tokenizer too for easy upload
        metrics = train_result.metrics

        metrics["train_samples"] = len(pretrain_dataset)

        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()
        if trainer.is_world_process_zero():
            tokenizer.save_pretrained(training_args.output_dir)
            torch.save(trainer.optimizer.state_dict(), os.path.join(
                training_args.output_dir, "optimizer.pt"))
            torch.save(trainer.lr_scheduler.state_dict(), os.path.join(
                training_args.output_dir, "scheduler.pt"))

    # Evaluation
    if training_args.do_eval:
        logger.info("*** Evaluate ***")
        if eval_dataset is None:
            eval_dataset = pretrain_dataset
        metrics = trainer.evaluate()

        max_eval_samples = data_args.max_eval_samples if data_args.max_eval_samples is not None else len(
            eval_dataset)
        metrics["eval_samples"] = min(max_eval_samples, len(eval_dataset))
        try:
            perplexity = math.exp(metrics["eval_loss"])
        except OverflowError:
            perplexity = float("inf")
        metrics["perplexity"] = perplexity

        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)


def _mp_fn(index):
    # For xla_spawn (TPUs)
    main()


if __name__ == "__main__":
    main()
